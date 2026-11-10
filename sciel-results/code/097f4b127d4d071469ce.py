"""Combined program: independent-optimization DP seeded + joint refinement pass."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def _build_grids(k, n_coarse, n_fine):
    """Depth grid focused around 1/k where the optimum lives."""
    lo = 1e-4
    hi = max(10.0 / k, 2.0)
    grid = np.concatenate([
        np.linspace(lo, hi, n_coarse),
        np.linspace(lo, 3.0 / k, n_fine),
    ])
    grid = np.unique(np.clip(grid, lo, hi))
    return grid


def _build_joint_grid(k, n_pts):
    """Denser joint grid for refinement, focused near 1/k."""
    lo = 1e-4
    hi = 3.0 / k
    # Non-uniform: denser near 1/k (the typical optimum)
    uniform = np.linspace(lo, hi, n_pts)
    # Add extra points around 1/k
    center = 1.0 / k
    extra = np.linspace(center * 0.5, center * 2.0, n_pts // 2)
    grid = np.unique(np.clip(np.concatenate([uniform, extra]), lo, hi))
    return grid


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Two-stage DP over (time, inventory).
    Stage 1: independent bid/ask optimization (fast) to seed V.
    Stage 2: joint (db, da) grid refinement where time permits.
    """
    dt = T / steps
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    n_coarse, n_fine = 40, 60
    depth_grid = _build_grids(k, n_coarse, n_fine)
    # Joint refinement grid - vectorized, so we can afford more points
    jn = 45
    joint_grid = _build_joint_grid(k, jn)
    jn = len(joint_grid)

    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    joint_probs = 1.0 - np.exp(-A * np.exp(-k * joint_grid) * dt)

    V_next = np.zeros(2 * Q + 1)
    q_vals = np.arange(-Q, Q + 1)
    penalties = (gamma / 2.0) * sigma**2 * q_vals**2 * dt

    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)

    # Stage 1: fast independent optimization (always completed)
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(2 * Q + 1)
        for q_idx in range(2 * Q + 1):
            q = q_idx - Q
            penalty = penalties[q_idx]
            if q == Q:
                delta = V_next[q_idx - 1] - V_next[q_idx]
                vals = fill_probs * (depth_grid + delta)
                bi = int(np.argmax(vals))
                V_curr[q_idx] = vals[bi] - penalty + V_next[q_idx]
                opt_ask[n, q_idx] = depth_grid[bi]
                opt_bid[n, q_idx] = np.inf
            elif q == -Q:
                delta = V_next[q_idx + 1] - V_next[q_idx]
                vals = fill_probs * (depth_grid + delta)
                bi = int(np.argmax(vals))
                V_curr[q_idx] = vals[bi] - penalty + V_next[q_idx]
                opt_bid[n, q_idx] = depth_grid[bi]
                opt_ask[n, q_idx] = np.inf
            else:
                dp = V_next[q_idx + 1] - V_next[q_idx]
                dm = V_next[q_idx - 1] - V_next[q_idx]
                vb = fill_probs * (depth_grid + dp)
                va = fill_probs * (depth_grid + dm)
                bi = int(np.argmax(vb))
                ai = int(np.argmax(va))
                V_curr[q_idx] = vb[bi] + va[ai] - penalty + V_next[q_idx]
                opt_bid[n, q_idx] = depth_grid[bi]
                opt_ask[n, q_idx] = depth_grid[ai]
        V_next = V_curr

    # Stage 2: joint refinement from the end, fully vectorized across inventory
    V_next = np.zeros(2 * Q + 1)
    # Re-seed: we recompute backward from terminal, refining each step
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 4.0:
            break
        V_curr = np.zeros(2 * Q + 1)
        
        # Interior inventory levels: vectorized 2D search
        # For q in (-Q+1, ..., Q-1): indices 1..2Q-1
        interior = slice(1, 2 * Q)
        Vq = V_next[interior]
        Vp = V_next[interior + 1]
        Vm = V_next[interior - 1]
        dp = Vp - Vq
        dm = Vm - Vq
        cross = Vp - 2.0 * Vq + Vm
        pen = penalties[interior]
        
        # Shapes: (jn, jn, 2Q-1)
        pb = joint_probs[:, None, None]
        pa = joint_probs[None, :, None]
        db = joint_grid[:, None, None]
        da = joint_grid[None, :, None]
        next_val = Vq[None, None, :] + pb * dp[None, None, :] + pa * dm[None, None, :] - pb * pa * cross[None, None, :]
        reward = db * pb + da * pa
        total = reward - pen[None, None, :] + next_val
        
        flat = total.reshape(jn * jn, -1)
        bi_flat = np.argmax(flat, axis=0)  # shape (2Q-1,)
        V_curr[interior] = flat[bi_flat, np.arange(len(interior))]
        opt_bid[n, interior] = joint_grid[bi_flat // jn]
        opt_ask[n, interior] = joint_grid[bi_flat % jn]
        
        # Boundary: q = Q (index 2Q)
        delta_q = V_next[2 * Q - 1] - V_next[2 * Q]
        vals_q = joint_probs * (joint_grid + delta_q)
        bi_q = int(np.argmax(vals_q))
        V_curr[2 * Q] = vals_q[bi_q] - penalties[2 * Q] + V_next[2 * Q]
        opt_ask[n, 2 * Q] = joint_grid[bi_q]
        opt_bid[n, 2 * Q] = np.inf
        
        # Boundary: q = -Q (index 0)
        delta_m = V_next[1] - V_next[0]
        vals_m = joint_probs * (joint_grid + delta_m)
        bi_m = int(np.argmax(vals_m))
        V_curr[0] = vals_m[bi_m] - penalties[0] + V_next[0]
        opt_bid[n, 0] = joint_grid[bi_m]
        opt_ask[n, 0] = np.inf
        
        V_next = V_curr

    # Stage 3: 1D polish - refine each depth independently using the joint result as seed
    # This allows finer depth resolution at the final step
    if time.time() < _DEADLINE - 8.0:
        # Build a finer 1D grid for polishing
        n_polish = 120
        polish_grid = np.linspace(1e-4, 3.0 / k, n_polish)
        polish_probs = 1.0 - np.exp(-A * np.exp(-k * polish_grid) * dt)
        
        V_next = np.zeros(2 * Q + 1)
        for n in range(steps - 1, -1, -1):
            if time.time() > _DEADLINE - 4.0:
                break
            V_curr = np.zeros(2 * Q + 1)
            
            # Interior: independent 1D optimization with cross-term approximation
            # Use the joint solution's implied next-state structure
            interior = slice(1, 2 * Q)
            Vq = V_next[interior]
            Vp = V_next[interior + 1]
            Vm = V_next[interior - 1]
            dp = Vp - Vq
            dm = Vm - Vq
            pen = penalties[interior]
            
            # Bid polish: maximize pb*(db + dp) - penalty/2 + Vq
            # (splitting penalty evenly for the 1D approximation)
            vb = polish_probs * (polish_grid + dp[None, :])
            bi_idx = np.argmax(vb, axis=0)
            V_curr[interior] = vb[bi_idx, np.arange(len(interior))] - pen + V_next[interior]
            opt_bid[n, interior] = polish_grid[bi_idx]
            
            # Ask polish: maximize pa*(da + dm)
            va = polish_probs * (polish_grid + dm[None, :])
            ai_idx = np.argmax(va, axis=0)
            opt_ask[n, interior] = polish_grid[ai_idx]
            # Update V_curr to include both sides
            V_curr[interior] = vb[bi_idx, np.arange(len(interior))] + va[ai_idx, np.arange(len(interior))] - pen + V_next[interior]
            
            # Boundaries
            delta_q = V_next[2 * Q - 1] - V_next[2 * Q]
            vals_q = polish_probs * (polish_grid + delta_q)
            bi_q = int(np.argmax(vals_q))
            V_curr[2 * Q] = vals_q[bi_q] - penalties[2 * Q] + V_next[2 * Q]
            opt_ask[n, 2 * Q] = polish_grid[bi_q]
            opt_bid[n, 2 * Q] = np.inf
            
            delta_m = V_next[1] - V_next[0]
            vals_m = polish_probs * (polish_grid + delta_m)
            bi_m = int(np.argmax(vals_m))
            V_curr[0] = vals_m[bi_m] - penalties[0] + V_next[0]
            opt_bid[n, 0] = polish_grid[bi_m]
            opt_ask[n, 0] = np.inf
            
            V_next = V_curr
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return _POLICY_CACHE[key]


def _analytical_quote(t, inventory, mid, params):
    """Avellaneda-Stoikov style analytical fallback."""
    sigma = params["sigma"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    dt = T / steps
    remaining = T - t
    base = 1.0 / k
    skew_coeff = gamma * sigma**2 * remaining / 2.0
    c = skew_coeff / (2.0 * k)
    bid_depths = base + c * inventory
    ask_depths = base - c * inventory
    bid_depths = np.clip(bid_depths, 1e-4, 10.0 / k)
    ask_depths = np.clip(ask_depths, 1e-4, 10.0 / k)
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Quote using pre-computed optimal policy with analytical fallback."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    n = int(round(t / (T / steps)))
    n = min(max(n, 0), steps - 1)

    key = (sigma, A, k, gamma, T, steps, Q)

    if key not in _POLICY_CACHE:
        if time.time() < _DEADLINE - 12.0:
            try:
                _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
            except Exception:
                pass

    if key in _POLICY_CACHE:
        opt_bid, opt_ask = _POLICY_CACHE[key]
        q_idx = np.clip(inventory + Q, 0, 2 * Q)
        bid_depths = opt_bid[n, q_idx].copy()
        ask_depths = opt_ask[n, q_idx].copy()
        bid_depths = np.where(bid_depths < 1e-4, np.inf, bid_depths)
        ask_depths = np.where(ask_depths < 1e-4, np.inf, ask_depths)
        return (np.asarray(bid_depths, dtype=np.float64),
                np.asarray(ask_depths, dtype=np.float64))
    else:
        return _analytical_quote(t, inventory, mid, params)
# EVOLVE-BLOCK-END

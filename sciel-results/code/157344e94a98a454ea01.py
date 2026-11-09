import numpy as np
import time
from scipy.optimize import brentq

_cache = {}
_START_TIME = time.time()
# Keep a safety margin. The evaluator kills at 330s.
# We want to finish well before.
_DEADLINE = _START_TIME + 290.0

def _refine_depth(C, A, k, dt, x_init):
    """
    Refine the depth d given initial x = exp(-kd).
    Objective: f(x) = (1 - exp(-L x)) * (C - log(x)/k)
    where L = A * dt.
    We want to maximize f(x) for x in (0, 1].
    """
    L = A * dt
    inv_k = 1.0 / k
    
    # If C is very negative, we might not want to quote at all.
    # However, the grid search will have filtered out obviously bad spots.
    # We assume x_init is a good local minimum for the derivative.
    
    def h(x):
        # h(x) = f'(x)
        # f'(x) = L * exp(-Lx) * (C - log(x)/k) - (1 - exp(-Lx)) / (k*x)
        exp_term = np.exp(-L * x)
        term1 = L * exp_term * (C - inv_k * np.log(x))
        term2 = (1.0 - exp_term) * inv_k / x
        return term1 - term2

    # Check if we are already at the boundary x=1 (d=0)
    if x_init >= 1.0 - 1e-12:
        # If derivative at 1 is positive, max is at 1.
        # h(1) = L * C
        if L * C >= 0:
            return 1.0
        # Else, root is in (0, 1)
        
    # Check if we are at the lower bound
    if x_init <= 1e-12:
        return 1e-12

    try:
        # We need a bracket.
        # h(x) -> +inf as x->0 (if C is finite, -log(x) dominates? 
        # Let's check: L exp(-Lx)(C - log x/k) ~ L C + L log x / k.
        # As x->0, log x -> -inf. If C is fixed, term1 -> -inf?
        # Wait. f(x) = (1-e^{-Lx})(C - log x/k).
        # As x->0, 1-e^{-Lx} ~ Lx. f(x) ~ Lx(C - log x/k) -> 0 (since x log x -> 0).
        # f(0)=0. f(1)=0.
        # So max is inside (0,1).
        # h(x) sign change:
        # At x=1, h(1) = L*C.
        # At x->0, h(x) ~ L(C - log x/k) - 1/(kx).
        # -1/(kx) dominates -> -inf.
        # So h(0+) = -inf.
        # If h(1) > 0, there is a root in (0,1).
        # If h(1) <= 0, the function is decreasing at 1?
        # If h(1) < 0, and h(0) < 0, is there a root?
        # f(0)=0, f(1)=0. If f is concave-ish, max is inside.
        # Actually, if h(1) < 0, it means f is decreasing at x=1.
        # Since f(1)=0 and f(0)=0, if it decreases at 1, it must have increased before.
        # So there is a root where h goes from + to -.
        
        # We use the x_init from grid to help bracketing or just use it as start for Newton?
        # Brentq is safer. We need a bracket [a, b] where h(a)*h(b) < 0.
        
        # Let's sample a few points to find a bracket if x_init is not good enough.
        # But x_init is from grid max, so it should be close to the root.
        
        # Try brentq with a small window around x_init
        x_lo = max(1e-12, x_init * 0.1)
        x_hi = min(1.0, x_init * 10.0)
        
        # Ensure bracket
        f_lo = h(x_lo)
        f_hi = h(x_hi)
        
        if f_lo * f_hi > 0:
            # Expand or fallback to grid value
            # If signs are same, the root might be outside or we are at boundary
            if f_hi > 0:
                # Increasing at hi, so max is to the right (closer to 1)
                return 1.0
            else:
                # Decreasing at lo, so max is to the left (closer to 0)
                return x_lo
        
        x_star = brentq(h, x_lo, x_hi, xtol=1e-12, rtol=1e-12)
        return x_star
    except Exception:
        # Fallback to grid value
        return x_init

def _solve_regime(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]

    if time.time() > _DEADLINE:
        # Fallback
        sigma = params['sigma']
        Q = params['Q']
        n_q = 2 * Q + 1
        N = params['steps']
        opt_b = np.full((N, n_q), 0.05)
        opt_a = np.full((N, n_q), 0.05)
        opt_b[:, 2 * Q] = np.inf
        opt_a[:, 0] = np.inf
        _cache[key] = (opt_b, opt_a)
        return _cache[key]

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)

    # Pre-compute cost
    cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt

    # Grid for x = exp(-kd)
    # x in (0, 1]
    # Log space is good.
    n_grid = 2000
    x_grid = np.logspace(-12, 0, n_grid)
    log_x = np.log(x_grid)
    inv_k = 1.0 / k
    d_grid = -log_x * inv_k
    p_grid = 1.0 - np.exp(-A * x_grid * dt)
    
    # Pre-compute terms for J = p * (C + d)
    # J = p * (C - log(x)/k) = p*C - p*log(x)/k
    # We will compute this on the fly for each C.

    # Iterate over time steps backwards
    for t in range(N - 1, -1, -1):
        if time.time() > _DEADLINE:
             # Fill remaining with fallback
             for tt in range(t, -1, -1):
                 opt_b[tt, :] = 0.05
                 opt_a[tt, :] = 0.05
                 opt_b[tt, 2 * Q] = np.inf
                 opt_a[tt, 0] = np.inf
             _cache[key] = (opt_b, opt_a)
             return _cache[key]

        V_next = V[t + 1, :]
        
        # C_b[q] = V(q+1) - V(q)
        # C_a[q] = V(q-1) - V(q)
        C_b = np.zeros(n_q)
        C_a = np.zeros(n_q)
        
        # q < Q (idx < 2Q)
        C_b[:2*Q] = V_next[1:] - V_next[:-1]
        # q > -Q (idx > 0)
        C_a[1:] = V_next[:-1] - V_next[1:]

        # --- Solve for Bid Depths ---
        b_vals = np.full(n_q, np.inf)
        p_b_vals = np.zeros(n_q)
        
        valid_b = np.arange(n_q) < 2*Q
        
        # Vectorized Grid Search for Initial Estimate
        # C_b shape (n_q,), x_grid shape (n_grid,)
        # J_b shape (n_q, n_grid)
        # J_b[i, j] = p_grid[j] * (C_b[i] - log_x[j] * inv_k)
        
        # To save memory/time, we can do this in chunks or just broadcast if n_q*n_grid is small enough.
        # n_q = 61, n_grid = 2000. 122k floats. Fine.
        
        J_b_grid = p_grid[None, :] * (C_b[:, None] - log_x[None, :] * inv_k)
        
        max_idx_b = np.argmax(J_b_grid, axis=1)
        max_val_b = J_b_grid[np.arange(n_q), max_idx_b]
        
        # Determine if we should quote
        # If max_val <= 0, don't quote (inf depth)
        quote_b_mask = max_val_b > 1e-9
        
        # Get x_init for refinement
        x_init_b = x_grid[max_idx_b]
        
        # Refine for valid quotes
        for i in np.where(valid_b & quote_b_mask)[0]:
            C = C_b[i]
            x_init = x_init_b[i]
            x_star = _refine_depth(C, A, k, dt, x_init)
            
            # Calculate actual depth and prob
            d = -np.log(x_star) * inv_k
            p = 1.0 - np.exp(-A * x_star * dt)
            
            # Ensure depth is valid (positive)
            if d < 0: d = 0.0
            if d > 100: d = 100.0
            
            b_vals[i] = d
            p_b_vals[i] = p
            
        # --- Solve for Ask Depths ---
        a_vals = np.full(n_q, np.inf)
        p_a_vals = np.zeros(n_q)
        
        valid_a = np.arange(n_q) > 0
        
        J_a_grid = p_grid[None, :] * (C_a[:, None] - log_x[None, :] * inv_k)
        
        max_idx_a = np.argmax(J_a_grid, axis=1)
        max_val_a = J_a_grid[np.arange(n_q), max_idx_a]
        
        quote_a_mask = max_val_a > 1e-9
        
        x_init_a = x_grid[max_idx_a]
        
        for i in np.where(valid_a & quote_a_mask)[0]:
            C = C_a[i]
            x_init = x_init_a[i]
            x_star = _refine_depth(C, A, k, dt, x_init)
            
            d = -np.log(x_star) * inv_k
            p = 1.0 - np.exp(-A * x_star * dt)
            
            if d < 0: d = 0.0
            if d > 100: d = 100.0
            
            a_vals[i] = d
            p_a_vals[i] = p

        opt_b[t, :] = b_vals
        opt_a[t, :] = a_vals
        
        # --- Update V ---
        # V(t, q) = V(t+1, q) + cost(q) + E[Reward]
        # Reward = p_b * (b + C_b) + p_a * (a + C_a) - p_b * p_a * (C_b + C_a)
        # Note: C_b = V(q+1)-V(q), C_a = V(q-1)-V(q)
        # If both fill, inventory goes to q (no net change? No, bid fill -> q+1, ask fill -> q-1. 
        # If both fill: q -> q+1 -> q. Net change 0.
        # Value if both fill: V(t+1, q).
        # Value if bid only: V(t+1, q+1) = V(t+1, q) + C_b.
        # Value if ask only: V(t+1, q-1) = V(t+1, q) + C_a.
        # Value if none: V(t+1, q).
        
        # E[Next V] = (1-p_b)(1-p_a)V(q) + p_b(1-p_a)(V(q)+C_b) + (1-p_b)p_a(V(q)+C_a) + p_b p_a V(q)
        # = V(q) + p_b(1-p_a)C_b + (1-p_b)p_a C_a
        # = V(q) + p_b C_b - p_b p_a C_b + p_a C_a - p_b p_a C_a
        # = V(q) + p_b C_b + p_a C_a - p_b p_a (C_b + C_a)
        
        # Immediate Reward (Depth earned)
        # Bid fill earns b. Ask fill earns a.
        # E[Depth] = p_b * b + p_a * a
        
        # Total E[Step Value] = p_b * b + p_a * a + p_b C_b + p_a C_a - p_b p_a (C_b + C_a)
        # = p_b (b + C_b) + p_a (a + C_a) - p_b p_a (C_b + C_a)
        
        reward_b = p_b_vals * (b_vals + C_b)
        reward_b = np.where(np.isinf(b_vals), 0.0, reward_b)
        
        reward_a = p_a_vals * (a_vals + C_a)
        reward_a = np.where(np.isinf(a_vals), 0.0, reward_a)
        
        cross_term = p_b_vals * p_a_vals * (C_b + C_a)
        cross_term = np.where(np.isfinite(C_b + C_a), cross_term, 0.0)
        
        V[t, :] = V_next + cost + reward_b + reward_a - cross_term

    _cache[key] = (opt_b, opt_a)
    return _cache[key]

def quote(t, inventory, mid, params):
    opt_b, opt_a = _solve_regime(params)
    dt = params['T'] / params['steps']
    Q = params['Q']
    
    t_idx = int(t / dt)
    if t_idx >= len(opt_b):
        t_idx = len(opt_b) - 1
    if t_idx < 0:
        t_idx = 0
        
    idxs = inventory + Q
    
    bid_depths = opt_b[t_idx, idxs]
    ask_depths = opt_a[t_idx, idxs]
    
    return bid_depths, ask_depths

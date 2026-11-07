"""Optimal market making with inventory risk - backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # safety margin before 330s kill

_OPT_TABLES = {}


def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using backward induction on the discrete HJB equation.
    
    Returns a tuple (bid_table, ask_table) where each is an array of shape
    (steps, 2*Q+1) mapping (step, inventory+Q) to optimal depth.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _OPT_TABLES:
        return _OPT_TABLES[key]

    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps

    # Depth grid: we need to search over possible depths
    # The fill probability is p = 1 - exp(-A * exp(-k*d) * dt)
    # For small dt, p ≈ A * exp(-k*d) * dt
    # We want depths that give reasonable fill probabilities
    
    # Create a grid of depths to evaluate
    # Minimum valid depth: > -3/k
    # Maximum useful depth: very large (fill prob ~ 0)
    # We'll use a logarithmic spacing in the "intensity" space
    
    # Let's use intensity = A * exp(-k*d) * dt as the parameter
    # d = (1/k) * ln(A*dt / intensity)
    # For intensity from very small to A*dt (max possible)
    
    # Actually, let's just use a grid of depths directly
    # d ranges from about 0 to maybe 5/k or so
    # With k=0.75, 5/k ≈ 6.67
    # With k=1.5, 5/k ≈ 3.33
    
    n_depths = 200
    # Use a range that covers reasonable depths
    d_min = 0.001
    d_max = 5.0 / k
    
    # Linear grid in depth space
    depths_grid = np.linspace(d_min, d_max, n_depths)
    
    # Precompute fill probabilities for each depth
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_kd = np.exp(-k * depths_grid)
    fill_probs = 1.0 - np.exp(-A * exp_kd * dt)
    
    # Precompute the "cost" of each depth: the depth itself (earned if filled)
    # Objective contribution from a fill at depth d: +d (earn the depth)
    # But we also pay inventory risk: (gamma/2) * sigma^2 * q^2 * dt per step
    
    # Backward induction
    # V[n, q] = optimal expected value from step n with inventory q
    # At step n, we choose bid_depth b and ask_depth a
    # Then:
    #   - Bid fill prob: p_b = 1 - exp(-A*exp(-k*b)*dt) if q < Q, else 0
    #   - Ask fill prob: p_a = 1 - exp(-A*exp(-k*a)*dt) if q > -Q, else 0
    #   - Both can fill independently
    #   - Inventory changes: q -> q+1 (bid fill), q-1 (ask fill), q (no fill)
    #   - Mid moves by sigma*sqrt(dt)*N(0,1), but this doesn't affect the objective
    #     directly (the objective is depths earned minus risk cost)
    
    # Wait, let me re-read the objective:
    # "depths earned minus sum over steps of (gamma/2) sigma^2 q^2 dt"
    # "A fill earns its depth"
    
    # So if bid fills, we earn bid_depth (the distance below mid)
    # If ask fills, we earn ask_depth (the distance above mid)
    # And we pay (gamma/2) * sigma^2 * q^2 * dt each step
    
    # The mid movement doesn't directly enter the objective, so we can ignore it
    # for the optimization (it's independent of our choices)
    
    # V[n, q] = max over (b, a) of:
    #   - (gamma/2) * sigma^2 * q^2 * dt
    #   + p_b * (b + V[n+1, q+1]) if q < Q
    #   + p_a * (a + V[n+1, q-1]) if q > -Q
    #   + (1-p_b)*(1-p_a) * V[n+1, q]
    #   + p_b*(1-p_a) * V[n+1, q+1]  # wait, this is already counted above
    #   
    # Actually, let's be more careful:
    # Let B = bid fill event, A = ask fill event
    # P(B) = p_b (if q < Q), else 0
    # P(A) = p_a (if q > -Q), else 0
    # P(B and A) = p_b * p_a (independent)
    # 
    # Expected value:
    # E = - (gamma/2) * sigma^2 * q^2 * dt
    #   + p_b * b  + p_a * a
    #   + P(no fill) * V[n+1, q]
    #   + P(B only) * V[n+1, q+1]
    #   + P(A only) * V[n+1, q-1]
    #   + P(both) * V[n+1, q]  # q+1-1 = q
    #
    # Where:
    # P(no fill) = (1-p_b)*(1-p_a)
    # P(B only) = p_b*(1-p_a)
    # P(A only) = p_a*(1-p_b)
    # P(both) = p_b*p_a
    #
    # So: E = -risk + p_b*b + p_a*a + [(1-p_b)*(1-p_a) + p_b*p_a] * V[n+1,q]
    #          + p_b*(1-p_a) * V[n+1,q+1] + p_a*(1-p_b) * V[n+1,q-1]
    
    # Terminal condition: V[steps, q] = 0 (no terminal charge)
    
    # State space: q in [-Q, Q], so 2*Q+1 states
    n_states = 2 * Q + 1
    
    # Initialize V at terminal step
    V = np.zeros((steps + 1, n_states))
    
    # Bid and ask depth tables
    bid_table = np.zeros((steps, n_states))
    ask_table = np.zeros((steps, n_states))
    
    # Backward induction from n = steps-1 down to 0
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 5:
            # Time running out, break and use what we have
            break
        
        V_next = V[n + 1]
        risk_cost = 0.5 * gamma * sigma * sigma * dt
        
        for qi in range(n_states):
            q = qi - Q  # actual inventory value
            
            # Check boundaries
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            best_val = -np.inf
            best_bid = depths_grid[0]
            best_ask = depths_grid[0]
            
            if not can_bid and not can_ask:
                # No fills possible, just pay risk
                best_val = -risk_cost * q * q
                best_bid = np.inf
                best_ask = np.inf
            else:
                # Search over bid and ask depths
                # For speed, we can separate the optimization
                # But let's do a joint search on a reduced grid
                
                # Reduce grid size for speed
                n_search = 50
                search_indices = np.linspace(0, n_depths - 1, n_search, dtype=int)
                search_depths = depths_grid[search_indices]
                search_probs = fill_probs[search_indices]
                
                for bi in range(n_search):
                    if not can_bid:
                        continue
                    b = search_depths[bi]
                    pb = search_probs[bi]
                    
                    for ai in range(n_search):
                        if not can_ask:
                            continue
                        a = search_depths[ai]
                        pa = search_probs[ai]
                        
                        # Compute expected value
                        val = -risk_cost * q * q + pb * b + pa * a
                        val += (1 - pb) * (1 - pa) * V_next[qi]
                        if q + 1 <= Q:
                            val += pb * (1 - pa) * V_next[qi + 1]
                        if q - 1 >= -Q:
                            val += pa * (1 - pb) * V_next[qi - 1]
                        # Both fills: inventory returns to q
                        val += pb * pa * V_next[qi]
                        
                        if val > best_val:
                            best_val = val
                            best_bid = b
                            best_ask = a
                
                # Handle boundary cases
                if not can_bid:
                    best_bid = np.inf
                    # Optimize ask only
                    best_val = -np.inf
                    for ai in range(n_search):
                        a = search_depths[ai]
                        pa = search_probs[ai]
                        val = -risk_cost * q * q + pa * a
                        val += (1 - pa) * V_next[qi]
                        if q - 1 >= -Q:
                            val += pa * V_next[qi - 1]
                        if val > best_val:
                            best_val = val
                            best_ask = a
                
                if not can_ask:
                    best_ask = np.inf
                    # Optimize bid only
                    best_val = -np.inf
                    for bi in range(n_search):
                        b = search_depths[bi]
                        pb = search_probs[bi]
                        val = -risk_cost * q * q + pb * b
                        val += (1 - pb) * V_next[qi]
                        if q + 1 <= Q:
                            val += pb * V_next[qi + 1]
                        if val > best_val:
                            best_val = val
                            best_bid = b
            
            V[n, qi] = best_val
            bid_table[n, qi] = best_bid
            ask_table[n, qi] = best_ask
    
    _OPT_TABLES[key] = (bid_table, ask_table)
    return _OPT_TABLES[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths using pre-computed backward induction table.
    """
    if time.time() > _DEADLINE:
        # Fallback: simple heuristic
        sigma = params['sigma']
        A = params['A']
        k = params['k']
        gamma = params['gamma']
        Q = params['Q']
        steps = params['steps']
        T = params['T']
        dt = T / steps
        q = inventory.astype(np.float64)
        d_base = 1.0 / k
        time_factor = np.sqrt(np.maximum((T - t) / T, 0.0))
        alpha = 0.5 / (k * Q) if Q > 0 else 0.0
        bid_depth = d_base + alpha * q * time_factor
        ask_depth = d_base - alpha * q * time_factor
        min_depth = np.maximum(0.01, -3.0/k + 0.01)
        bid_depth = np.maximum(bid_depth, min_depth)
        ask_depth = np.maximum(ask_depth, min_depth)
        bid_depth[inventory >= Q] = np.inf
        ask_depth[inventory <= -Q] = np.inf
        return bid_depth, ask_depth

    bid_table, ask_table = _get_optimal_table(params)
    
    steps = params['steps']
    Q = params['Q']
    
    # Determine current step
    T = params['T']
    dt = T / steps
    step_idx = int(round(t / dt))
    step_idx = np.clip(step_idx, 0, steps - 1)
    
    # Map inventory to table index
    inv_idx = inventory + Q  # inventory in [-Q, Q] -> [0, 2Q]
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[step_idx, inv_idx]
    ask_depths = ask_table[step_idx, inv_idx]
    
    # Ensure valid range: > -3/k, no NaN
    k = params['k']
    min_depth = -3.0 / k + 0.01
    bid_depths = np.where(np.isfinite(bid_depths), np.maximum(bid_depths, min_depth), np.inf)
    ask_depths = np.where(np.isfinite(ask_depths), np.maximum(ask_depths, min_depth), np.inf)
    
    # Ensure correct shape
    bid_depths = np.asarray(bid_depths, dtype=np.float64)
    ask_depths = np.asarray(ask_depths, dtype=np.float64)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END

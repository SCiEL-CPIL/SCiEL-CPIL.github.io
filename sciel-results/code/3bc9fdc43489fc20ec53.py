import numpy as np

# Global cache for precomputed policies
_POLICY_CACHE = {}

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid and ask depths for all steps and inventory levels.
    """
    dt = T / steps
    n_steps = steps
    q_min = -Q
    q_max = Q
    n_q = 2 * Q + 1
    
    # Value function V[n, q_index]
    V = np.zeros((n_steps + 1, n_q), dtype=np.float64)
    
    # Optimal depths
    opt_bid = np.full((n_steps, n_q), np.inf, dtype=np.float64)
    opt_ask = np.full((n_steps, n_q), np.inf, dtype=np.float64)
    
    # Grid for depth search
    d_max = 5.0 / k
    n_grid = 500
    depths = np.linspace(0.0, d_max, n_grid)
    
    # Precompute probabilities for the grid
    exp_kd = np.exp(-k * depths)
    probs = 1.0 - np.exp(-A * exp_kd * dt)
    
    # Backward induction
    for n in range(n_steps - 1, -1, -1):
        V_next = V[n + 1]
        
        for i in range(n_q):
            q = q_min + i
            
            penalty = 0.5 * gamma * (sigma ** 2) * (q ** 2) * dt
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            if not can_bid and not can_ask:
                V[n, i] = -penalty
                continue
            
            W_plus = 0.0
            if q < Q:
                W_plus = V_next[i + 1] - V_next[i]
                
            W_minus = 0.0
            if q > -Q:
                W_minus = V_next[i] - V_next[i - 1]
            
            # Initialize
            b = 1.0 / k
            a = 1.0 / k
            
            # Iterate to find fixed point
            for _ in range(10):
                # Update b
                if can_bid:
                    p_a_curr = 1.0 - np.exp(-A * np.exp(-k * a) * dt) if np.isfinite(a) else 0.0
                    target_b = b + W_plus - p_a_curr * W_minus
                    vals = probs * (depths + target_b)
                    idx_b = np.argmax(vals)
                    if vals[idx_b] < 0:
                        b = np.inf
                    else:
                        b = depths[idx_b]
                else:
                    b = np.inf
                    p_b_curr = 0.0
                    
                # Update a
                if can_ask:
                    p_b_curr = 1.0 - np.exp(-A * np.exp(-k * b) * dt) if np.isfinite(b) else 0.0
                    target_a = a - W_minus + p_b_curr * W_plus
                    vals = probs * (depths + target_a)
                    idx_a = np.argmax(vals)
                    if vals[idx_a] < 0:
                        a = np.inf
                    else:
                        a = depths[idx_a]
                else:
                    a = np.inf
                    p_a_curr = 0.0

            # Final calculation
            p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt) if np.isfinite(b) else 0.0
            p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt) if np.isfinite(a) else 0.0
            
            reward = p_b * b + p_a * a
            
            exp_V = V_next[i] * (p_b * p_a + (1 - p_b) * (1 - p_a))
            if q < Q:
                exp_V += V_next[i + 1] * p_b * (1 - p_a)
            if q > -Q:
                exp_V += V_next[i - 1] * (1 - p_b) * p_a
                
            V[n, i] = reward + exp_V - penalty
            opt_bid[n, i] = b
            opt_ask[n, i] = a
            
    return opt_bid, opt_ask

def _get_policy(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _precompute_regime(*key)
    return _POLICY_CACHE[key]

def quote(t, inventory, mid, params):
    bid_table, ask_table = _get_policy(params)
    
    steps = params['steps']
    T = params['T']
    Q = params['Q']
    
    dt = T / steps
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    inv = np.asarray(inventory)
    indices = inv + Q
    indices = np.clip(indices, 0, 2 * Q)
    
    bid_depths = bid_table[n, indices]
    ask_depths = ask_table[n, indices]
    
    return bid_depths, ask_depths

import numpy as np
import time

_solution_cache = {}

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid and ask depths for all (step, inventory) states
    using backward induction with a separable 1D optimization.
    """
    dt = T / steps
    max_p = 1.0 - np.exp(-A * dt)  # max fill probability
    
    # Precompute the depth for a range of fill probabilities
    # p = 1 - exp(-exp(-k*d)*A*dt)  =>  d = -1/k * ln( -ln(1-p) / (A*dt) )
    # We'll precompute for a grid of p values and interpolate
    
    # Actually, for the optimization we need to solve:
    # max_p  p * (d(p) + C)  where d(p) = -1/k * ln(-ln(1-p)/(A*dt))
    # This is a 1D problem. We can use a fine grid or closed form.
    
    # Let's use a fine grid of p values
    N_p = 2048
    p_grid = np.linspace(0, max_p, N_p)
    p_grid[0] = 1e-15  # avoid log(0)
    
    # Compute d(p) for the grid
    with np.errstate(divide='ignore', invalid='ignore'):
        log_term = -np.log(1 - p_grid) / (A * dt)
        d_grid = -np.log(log_term) / k
    d_grid = np.clip(d_grid, 0, 100/k)  # reasonable bounds
    
    # For each C, find optimal p: max p*(d(p)+C)
    # C = V(q+1) - V(q) for bid, C = V(q-1) - V(q) for ask
    # Range of C: bounded by max possible value difference
    # Max cost per step ~ 0.5*gamma*sigma^2*Q^2*dt
    # Total ~ 0.5*gamma*sigma^2*Q^2
    max_cost = 0.5 * gamma * sigma**2 * Q**2
    C_min, C_max = -2*max_cost, 2*max_cost
    N_C = 4096
    C_grid = np.linspace(C_min, C_max, N_C)
    
    # For each C, compute p*(d+C) over p_grid and find argmax
    # vals[i,j] = p_grid[j] * (d_grid[j] + C_grid[i])
    # This is N_C x N_p = 4096 x 2048 ~ 8M floats = 64MB. OK.
    vals = p_grid[None, :] * (d_grid[None, :] + C_grid[:, None])
    idx_max = np.argmax(vals, axis=1)
    p_opt_grid = p_grid[idx_max]
    d_opt_grid = d_grid[idx_max]
    
    # Now backward induction
    n_states = 2 * Q + 1
    V = np.zeros(n_states)  # value function for next step
    bid_depths = np.full((steps, n_states), np.inf)
    ask_depths = np.full((steps, n_states), np.inf)
    
    for n in range(steps - 1, -1, -1):
        V_next = V.copy()
        V_curr = np.zeros(n_states)
        
        for i in range(n_states):
            q = i - Q
            
            # Compute C values
            # For bid: C_b = V(q+1) - V(q)
            # For ask: C_a = V(q-1) - V(q)
            if i < 2 * Q:
                C_b = V_next[i + 1] - V_next[i]
            else:
                C_b = -np.inf  # no bid at q=Q
            
            if i > 0:
                C_a = V_next[i - 1] - V_next[i]
            else:
                C_a = -np.inf  # no ask at q=-Q
            
            # Find optimal p and d
            if C_b > C_max:
                p_b = max_p
                d_b = 0.0
            elif C_b < C_min:
                p_b = 0.0
                d_b = np.inf
            else:
                idx = int((C_b - C_min) / (C_max - C_min) * (N_C - 1))
                idx = np.clip(idx, 0, N_C - 1)
                p_b = p_opt_grid[idx]
                d_b = d_opt_grid[idx]
            
            if C_a > C_max:
                p_a = max_p
                d_a = 0.0
            elif C_a < C_min:
                p_a = 0.0
                d_a = np.inf
            else:
                idx = int((C_a - C_min) / (C_max - C_min) * (N_C - 1))
                idx = np.clip(idx, 0, N_C - 1)
                p_a = p_opt_grid[idx]
                d_a = d_opt_grid[idx]
            
            bid_depths[n, i] = d_b
            ask_depths[n, i] = d_a
            
            # Compute expected value
            # Outcomes:
            # bid fill, no ask: prob p_b*(1-p_a), new q = q+1
            # no bid, ask fill: prob (1-p_b)*p_a, new q = q-1
            # both: prob p_b*p_a, new q = q
            # neither: prob (1-p_b)*(1-p_a), new q = q
            
            if i < 2 * Q:
                V_q1 = V_next[i + 1]
            else:
                V_q1 = V_next[i]  # shouldn't happen if p_b=0 at boundary
            
            if i > 0:
                V_qm1 = V_next[i - 1]
            else:
                V_qm1 = V_next[i]  # shouldn't happen if p_a=0 at boundary
            
            V_q = V_next[i]
            
            p10 = p_b * (1 - p_a)
            p01 = (1 - p_b) * p_a
            p11 = p_b * p_a
            p00 = (1 - p_b) * (1 - p_a)
            
            EV = p10 * V_q1 + p01 * V_qm1 + (p11 + p00) * V_q
            
            # Immediate reward
            reward_b = p_b * d_b if np.isfinite(d_b) else 0.0
            reward_a = p_a * d_a if np.isfinite(d_a) else 0.0
            
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            V_curr[i] = reward_b + reward_a - cost + EV
        
        V = V_curr
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _solution_cache:
        try:
            bid_depths, ask_depths = _precompute_regime(sigma, A, k, gamma, T, steps, Q)
            _solution_cache[key] = (bid_depths, ask_depths)
        except Exception:
            # Fallback: fixed depth
            depth = 1.0 / k
            n_paths = len(inventory)
            return np.full(n_paths, depth), np.full(n_paths, depth)
    
    bid_depths, ask_depths = _solution_cache[key]
    
    dt = T / steps
    n = int(t / dt + 0.5)
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
    
    # inventory is array of ints in [-Q, Q]
    idx = inventory + Q
    idx = np.clip(idx, 0, 2 * Q)
    
    b = bid_depths[n, idx]
    a = ask_depths[n, idx]
    
    # Ensure valid values: no negative depths, no NaN
    b = np.where(np.isnan(b) | (b < 0), np.inf, b)
    a = np.where(np.isnan(a) | (a < 0), np.inf, a)
    
    return b, a

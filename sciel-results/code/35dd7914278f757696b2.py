"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def _solve_depths_vectorized(A, k, dt, deltas, d_max=15.0/k, max_iter=50, tol=1e-12):
    """
    Solve for optimal depths for an array of deltas using vectorized Newton steps.
    Objective: f(d) = p(d) * (d + delta)
    p(d) = 1 - exp(-A exp(-k d) dt)
    FOC: p'(d)(d+delta) + p(d) = 0
    """
    n = len(deltas)
    d = np.full(n, 1.0 / k) # Initial guess
    
    # Clip initial guess
    d = np.clip(d, 0.0, d_max)
    
    for _ in range(max_iter):
        exp_kd = np.exp(-k * d)
        lam = A * exp_kd * dt
        
        # Calculate p, p', p''
        # Use stable expm1 for small lam
        # p = 1 - exp(-lam)
        # p' = k * lam * exp(-lam)
        # p'' = -k^2 * lam * exp(-lam) * (1 - lam)
        
        exp_neg_lam = np.exp(-lam)
        p = -np.expm1(-lam)
        p_prime = k * lam * exp_neg_lam
        p_double_prime = -k * k * lam * exp_neg_lam * (1.0 - lam)
        
        # g(d) = p'(d)(d+delta) + p(d)
        g = p_prime * (d + deltas) + p
        
        # g'(d) = p''(d)(d+delta) + p'(d)
        g_prime = p_double_prime * (d + deltas) + p_prime
        
        # Newton step
        # Handle division by zero
        mask = np.abs(g_prime) > 1e-15
        step = np.zeros(n)
        step[mask] = g[mask] / g_prime[mask]
        
        d_new = d - step
        
        # Clip to valid range
        d_new = np.clip(d_new, 0.0, d_max)
        
        # Check convergence
        if np.max(np.abs(d_new - d)) < tol:
            d = d_new
            break
        d = d_new
        
    return d

def _get_dp_tables(params):
    """Compute optimal bid/ask depth tables via dynamic programming."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    V = np.zeros((steps + 1, 2 * Q + 1))
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)
    
    q_indices = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma**2 * q_indices**2 * dt
    
    d_max = 15.0 / k
    
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            # If time is up, fill remaining with conservative quotes (large depth)
            # or just break and let quote() handle it with inf
            break
        
        V_next = V[n + 1]
        
        # Calculate deltas for bid and ask
        # delta_b[q] = V(q+1) - V(q) for q < Q
        # delta_a[q] = V(q-1) - V(q) for q > -Q
        
        # We need to solve for d for each q.
        # We can do this in two vectorized calls: one for bids, one for asks.
        
        bid_depths = np.full(2 * Q + 1, np.inf)
        ask_depths = np.full(2 * Q + 1, np.inf)
        
        # Bid side: q from -Q to Q-1
        if Q > 0:
            # indices 0 to 2Q-1 correspond to q = -Q to Q-1
            # V_next indices: 0..2Q
            # delta_b[i] = V_next[i+1] - V_next[i]
            deltas_b = V_next[1:] - V_next[:-1]
            
            # Solve for depths
            d_b = _solve_depths_vectorized(A, k, dt, deltas_b, d_max)
            
            # Calculate rewards
            exp_kd = np.exp(-k * d_b)
            lam = A * exp_kd * dt
            p = -np.expm1(-lam)
            rewards_b = p * (d_b + deltas_b)
            
            # Handle boundary: if optimal depth leads to negative reward and 0 is better?
            # The function f(d) at d=0 is p(0)*delta.
            # p(0) = 1 - exp(-A dt).
            # If delta is very negative, f(d) is negative for all d?
            # If f(d) < 0, we should quote np.inf (no quote).
            # But wait, the problem says "np.inf means no quote".
            # If quoting yields negative expected value, we should not quote.
            # So if max(f(d), 0) == 0, we set depth to inf.
            # Actually, we can always choose d -> inf, which gives p -> 0, reward -> 0.
            # So the optimal reward is max(0, max_d f(d)).
            # If max_d f(d) <= 0, we set depth to inf.
            
            valid_b = rewards_b > 1e-12
            bid_depths[:-1] = np.where(valid_b, d_b, np.inf)
            bid_rewards = np.where(valid_b, rewards_b, 0.0)
        else:
            bid_rewards = np.zeros(2 * Q + 1)
            
        # Ask side: q from -Q+1 to Q
        if Q > 0:
            # indices 1 to 2Q correspond to q = -Q+1 to Q
            # delta_a[i] = V_next[i-1] - V_next[i]
            deltas_a = V_next[:-1] - V_next[1:]
            
            d_a = _solve_depths_vectorized(A, k, dt, deltas_a, d_max)
            
            exp_kd = np.exp(-k * d_a)
            lam = A * exp_kd * dt
            p = -np.expm1(-lam)
            rewards_a = p * (d_a + deltas_a)
            
            valid_a = rewards_a > 1e-12
            ask_depths[1:] = np.where(valid_a, d_a, np.inf)
            ask_rewards = np.zeros(2 * Q + 1)
            ask_rewards[1:] = np.where(valid_a, rewards_a, 0.0)
        else:
            ask_rewards = np.zeros(2 * Q + 1)
        
        # Combine
        V[n] = V_next - risk + bid_rewards + ask_rewards
        opt_bid[n] = bid_depths
        opt_ask[n] = ask_depths
    
    _cache[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """Quote optimal depths based on dynamic programming solution."""
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        n_paths = len(inventory)
        return np.full(n_paths, np.inf), np.full(n_paths, np.inf)
    
    opt_bid, opt_ask = _get_dp_tables(params)
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    n = int(round(t / dt))
    n = min(max(n, 0), steps - 1)
    
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= 0), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= 0), ask_depths, np.inf)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)

# EVOLVE-BLOCK-END

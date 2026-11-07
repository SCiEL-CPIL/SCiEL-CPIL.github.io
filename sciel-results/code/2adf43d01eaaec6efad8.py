import numpy as np
import time

# Global cache for precomputed policies
_POLICY_CACHE = {}

def _solve_optimal_depth(beta, k, delta_v):
    """
    Finds optimal depth d >= 0 maximizing (1 - exp(-beta * exp(-k*d))) * (d + delta_v)
    beta = A * dt
    """
    if delta_v < 0:
        # If the value of increasing inventory is negative (we want to reduce it),
        # we might want to quote tighter (smaller d) to increase fill probability.
        # If delta_v is very negative, we might even want d=0.
        pass
        
    # We want to maximize f(d) = p(d) * (d + delta_v)
    # p(d) = 1 - exp(-beta * exp(-k*d))
    # f'(d) = p'(d)*(d+delta_v) + p(d)
    # p'(d) = k * beta * exp(-k*d) * exp(-beta * exp(-k*d))
    
    def f_prime(d):
        term = beta * np.exp(-k * d)
        p_prime = k * term * np.exp(-term)
        p = 1.0 - np.exp(-term)
        return p_prime * (d + delta_v) + p

    def f_second(d):
        # Numerical derivative of f_prime
        h = 1e-6
        return (f_prime(d + h) - f_prime(d - h)) / (2 * h)

    # Initial guess
    # If delta_v is large positive, we prefer high d (low prob, high reward)
    # If delta_v is large negative, we prefer low d (high prob, low reward) to offload inventory
    # A reasonable start is 0.1
    d = 0.1
    
    # Newton's method
    for _ in range(10):
        fp = f_prime(d)
        fpp = f_second(d)
        if abs(fpp) < 1e-12:
            break
        step = fp / fpp
        d_new = d - step
        if d_new < 0:
            d_new = 0.0
        if abs(d_new - d) < 1e-8:
            d = d_new
            break
        d = d_new
        
    # Ensure non-negative
    if d < 0:
        d = 0.0
        
    return d

def _compute_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    beta = A * dt
    
    # V[n, q] where q in [-Q, Q]
    # Index q by q + Q
    num_q = 2 * Q + 1
    V = np.zeros((steps + 1, num_q))
    d_b_table = np.full((steps, num_q), np.inf)
    d_a_table = np.full((steps, num_q), np.inf)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        for q in range(-Q, Q + 1):
            idx = q + Q
            
            # Penalty
            penalty = -0.5 * gamma * sigma**2 * q**2 * dt
            
            # Next state values
            v_next_q = V[n + 1, idx]
            
            # Bid side (q+1)
            if q < Q:
                delta_v_b = V[n + 1, idx + 1] - v_next_q
                d_b = _solve_optimal_depth(beta, k, delta_v_b)
                p_b = 1.0 - np.exp(-beta * np.exp(-k * d_b))
                d_b_table[n, idx] = d_b
            else:
                d_b = np.inf
                p_b = 0.0
                d_b_table[n, idx] = np.inf
                
            # Ask side (q-1)
            if q > -Q:
                delta_v_a = V[n + 1, idx - 1] - v_next_q
                d_a = _solve_optimal_depth(beta, k, delta_v_a)
                p_a = 1.0 - np.exp(-beta * np.exp(-k * d_a))
                d_a_table[n, idx] = d_a
            else:
                d_a = np.inf
                p_a = 0.0
                d_a_table[n, idx] = np.inf
            
            # Calculate V[n, q]
            # E[R] = p_b * (d_b + V[n+1, q+1]) + p_a * (d_a + V[n+1, q-1]) + (1-p_b)(1-p_a) * V[n+1, q]
            
            term_b = 0.0
            if q < Q:
                term_b = p_b * (d_b + V[n + 1, idx + 1])
            
            term_a = 0.0
            if q > -Q:
                term_a = p_a * (d_a + V[n + 1, idx - 1])
                
            term_none = (1.0 - p_b) * (1.0 - p_a) * v_next_q
            
            V[n, idx] = penalty + term_b + term_a + term_none
            
    return V, d_b_table, d_a_table

def quote(t, inventory, mid, params):
    # Check cache
    cache_key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if cache_key not in _POLICY_CACHE:
        # Compute policy
        # This might take some time, but it's done once per regime
        V, d_b_table, d_a_table = _compute_policy(params)
        _POLICY_CACHE[cache_key] = (V, d_b_table, d_a_table)
        
    else:
        V, d_b_table, d_a_table = _POLICY_CACHE[cache_key]
        
    steps = params['steps']
    T = params['T']
    Q = params['Q']
    
    dt = T / steps
    n = int(round(t / dt))
    
    # Clamp n to valid range [0, steps-1]
    if n < 0:
        n = 0
    if n >= steps:
        n = steps - 1
        
    # Map inventory to index
    # inventory is an array of ints
    # idx = inventory + Q
    idxs = inventory + Q
    
    # Get depths
    bid_depths = d_b_table[n, idxs]
    ask_depths = d_a_table[n, idxs]
    
    return bid_depths, ask_depths

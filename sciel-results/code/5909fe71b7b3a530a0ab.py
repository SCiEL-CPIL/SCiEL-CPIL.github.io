import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0
_POLICY_CACHE = {}

def _solve_optimal_depth_vec(C_vals, alpha, k):
    """
    Vectorized solver for optimal depth x >= 0 maximizing (1 - exp(-alpha * exp(-k*x))) * (x + C)
    for each C in C_vals.
    Uses Newton's method with safeguards.
    """
    C_vals = np.asarray(C_vals, dtype=float)
    x = np.zeros_like(C_vals)
    
    # Initial guess
    x[:] = 1.0 / k
    
    for _ in range(100):
        exp_kx = np.exp(-k * x)
        p = 1.0 - np.exp(-alpha * exp_kx)
        
        # f(x) = p * (x + C)
        # f'(x) = p' * (x + C) + p
        # p' = -alpha * k * exp(-kx) * exp(-alpha * exp(-kx))
        
        p_prime = -alpha * k * exp_kx * np.exp(-alpha * exp_kx)
        
        f_val = p * (x + C_vals)
        f_prime = p_prime * (x + C_vals) + p
        
        # f''(x) = p''(x)*(x+C) + 2*p'(x)
        # p''(x) = k^2 * exp(-v) * (v - v^2) where v = alpha * exp(-kx)
        
        v = alpha * exp_kx
        p_double_prime = k**2 * np.exp(-v) * (v - v**2)
        
        f_double_prime = p_double_prime * (x + C_vals) + 2 * p_prime
        
        # Newton step for f'(x) = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            dx = -f_prime / f_double_prime
        
        # Safeguard: limit step size to prevent divergence
        dx = np.clip(dx, -10.0/k, 10.0/k)
        
        x_new = x + dx
        x_new = np.maximum(x_new, 0.0)
        
        # Check convergence
        if np.all(np.abs(x_new - x) < 1e-8):
            x = x_new
            break
        x = x_new
    
    return x

def _compute_policy(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
        
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    alpha = A * dt
    
    n_states = 2 * Q + 1
    # V[n, q_idx]
    V = np.zeros((steps + 1, n_states))
    
    bid_depths = np.full((steps, n_states), np.inf)
    ask_depths = np.full((steps, n_states), np.inf)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 2.0:
            # Emergency fallback
            # Fill remaining with a reasonable heuristic
            # e.g. depth = 1/k
            for i in range(n, steps):
                bid_depths[i] = 1.0/k
                ask_depths[i] = 1.0/k
            _POLICY_CACHE[key] = (bid_depths, ask_depths)
            return _POLICY_CACHE[key]

        V_next = V[n+1]
        V_curr = V[n]
        
        # Vectorized calculation for all q at this step n
        # q indices: 0 to 2Q
        # q values: -Q to Q
        
        # Calculate C_b and C_a for all valid q
        # C_b[q] = V_next[q+1] - V_next[q]
        # Valid for q < Q (indices 0 to 2Q-1)
        # C_b is shape (2Q,)
        C_b = V_next[1:] - V_next[:-1]
        
        # C_a[q] = V_next[q-1] - V_next[q]
        # Valid for q > -Q (indices 1 to 2Q)
        # C_a is shape (2Q,)
        # Note: C_a for q_idx i corresponds to V_next[i-1] - V_next[i]
        C_a = V_next[:-1] - V_next[1:]
        
        # Solve for optimal depths directly
        b_opt = _solve_optimal_depth_vec(C_b, alpha, k)
        a_opt = _solve_optimal_depth_vec(C_a, alpha, k)
        
        # Store optimal depths
        # bid_depths[n, 0:2Q] = b_opt
        # bid_depths[n, 2Q] = inf
        bid_depths[n, :2*Q] = b_opt
        bid_depths[n, 2*Q] = np.inf
        
        # ask_depths[n, 1:2Q+1] = a_opt
        # ask_depths[n, 0] = inf
        ask_depths[n, 1:] = a_opt
        ask_depths[n, 0] = np.inf
        
        # Calculate probabilities and gains
        # p_b = 1 - exp(-alpha * exp(-k * b_opt))
        # For q=Q, p_b = 0
        
        p_b = np.zeros(n_states)
        p_a = np.zeros(n_states)
        
        # Only compute for valid ones
        p_b[:2*Q] = 1.0 - np.exp(-alpha * np.exp(-k * b_opt))
        p_a[1:] = 1.0 - np.exp(-alpha * np.exp(-k * a_opt))
        
        # Gains
        # Gain_b = p_b * (b_opt + C_b)
        # Note: C_b is the diff V_next[q+1] - V_next[q].
        # The term in value eq is p_b * (b_opt + V_next[q+1]) + ...
        # Baseline V_next[q].
        # Incremental = p_b * (b_opt + V_next[q+1] - V_next[q]) = p_b * (b_opt + C_b)
        
        gain_b = np.zeros(n_states)
        gain_a = np.zeros(n_states)
        
        gain_b[:2*Q] = p_b[:2*Q] * (b_opt + C_b)
        gain_a[1:] = p_a[1:] * (a_opt + C_a)
        
        # Holding Cost
        # q values
        q_vals = np.arange(-Q, Q+1)
        cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
        
        # V_curr = -cost + V_next + gain_b + gain_a
        V_curr[:] = -cost + V_next + gain_b + gain_a

    _POLICY_CACHE[key] = (bid_depths, ask_depths)
    return _POLICY_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Returns (bid_depths, ask_depths) arrays of shape (4000,)
    """
    bid_table, ask_table = _compute_policy(params)
    
    steps = params["steps"]
    T = params["T"]
    dt = T / steps
    
    n = int(t / dt)
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    Q = params["Q"]
    
    inv_idx = np.clip(inventory + Q, 0, 2 * Q)
    
    bid_depths = bid_table[n, inv_idx]
    ask_depths = ask_table[n, inv_idx]
    
    return bid_depths.astype(float), ask_depths.astype(float)

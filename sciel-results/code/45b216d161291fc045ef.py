import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0
_POLICY_CACHE = {}

def _solve_depth_newton(alpha, k, C):
    """
    Solve for optimal depth x given C using Newton's method on the first-order condition.
    f(x) = (1 - exp(-alpha*exp(-k*x))) * (x + C)
    f'(x) = alpha*k*exp(-k*x)*(1 - exp(-alpha*exp(-k*x))) + (1 - exp(-alpha*exp(-k*x)))
         = (1 - exp(-alpha*exp(-k*x))) * (1 + alpha*k*exp(-k*x))
    We want f'(x) = 0, which means 1 + alpha*k*exp(-k*x) = 0 or exp(-alpha*exp(-k*x)) = 1.
    The latter only happens when alpha*exp(-k*x) = 0, i.e., x -> infinity.
    But we also have the boundary at x=0.
    
    Actually, let me reconsider. The objective is:
    g(x) = p(x) * (x + C) where p(x) = 1 - exp(-alpha * exp(-k*x))
    
    g'(x) = p'(x) * (x + C) + p(x)
    p'(x) = alpha * k * exp(-k*x) * exp(-alpha*exp(-k*x))
    
    So g'(x) = alpha * k * exp(-k*x) * exp(-alpha*exp(-k*x)) * (x + C) + (1 - exp(-alpha*exp(-k*x)))
    
    Set g'(x) = 0:
    alpha * k * exp(-k*x) * exp(-alpha*exp(-k*x)) * (x + C) = exp(-alpha*exp(-k*x)) - 1
    
    This is transcendental. Let's use Newton's method directly on g'(x) = 0.
    """
    C = np.atleast_1d(np.asarray(C, dtype=np.float64))
    n = len(C)
    x = np.zeros(n)
    
    # Initial guess: for large |C|, x is small if C>0 (want high prob, low cost) 
    # or x is large if C<0 (want low prob to avoid negative gain).
    # Actually, if C is very negative, we want x large to make p small.
    # If C is very positive, we want x small to make p large.
    
    # Heuristic initial guess
    x = np.where(C > 0, 1.0/k, 5.0/k)
    
    for _ in range(50):
        # Compute p and p'
        exp_kx = np.exp(-k * x)
        alpha_exp_kx = alpha * exp_kx
        p = 1.0 - np.exp(-alpha_exp_kx)
        p_prime = alpha * k * exp_kx * np.exp(-alpha_exp_kx)
        
        # g'(x) = p' * (x + C) + p
        grad = p_prime * (x + C) + p
        
        # g''(x) = p'' * (x + C) + p' + p'
        # p'' = alpha * k * exp(-k*x) * (-k * alpha * exp(-k*x) * exp(-alpha*exp(-k*x)) - k * exp(-alpha*exp(-k*x)))
        # p'' = alpha * k * exp(-k*x) * exp(-alpha*exp(-k*x)) * (-k*alpha*exp(-k*x) - k)
        # p'' = -alpha * k^2 * exp(-k*x) * exp(-alpha*exp(-k*x)) * (alpha*exp(-k*x) + 1)
        
        p_double_prime = -alpha * k**2 * exp_kx * np.exp(-alpha_exp_kx) * (alpha * exp_kx + 1.0)
        
        hessian = p_double_prime * (x + C) + 2.0 * p_prime
        
        # Newton step
        # Avoid division by zero
        hessian_safe = np.where(np.abs(hessian) < 1e-12, 1e-12, hessian)
        dx = grad / hessian_safe
        
        # Damping to ensure we don't go too far
        # If x + dx < 0, use bisection-like approach
        x_new = x - dx
        
        # Clip to valid range [0, 10/k]
        x_new = np.clip(x_new, 0.0, 10.0/k)
        
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
        
        # Solve for optimal depths directly using Newton's method
        b_opt = _solve_depth_newton(alpha, k, C_b)
        a_opt = _solve_depth_newton(alpha, k, C_a)
        
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

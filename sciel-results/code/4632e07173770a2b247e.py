import numpy as np
import time
from scipy.optimize import brentq

_cache = {}
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0

def _solve_depth(C, A, k, dt):
    """
    Solve for optimal depth d given constant C.
    Maximize (1 - exp(-A exp(-kd) dt)) * (d + C)
    Let x = exp(-kd). d = -log(x)/k.
    Maximize f(x) = (1 - exp(-A x dt)) * (C - log(x)/k)
    """
    L = A * dt
    inv_k = 1.0 / k
    
    # Objective derivative w.r.t x
    # f'(x) = L exp(-Lx) (C - log(x)/k) - (1 - exp(-Lx)) / (k x)
    # Set to 0:
    # L exp(-Lx) (C - log(x)/k) = (1 - exp(-Lx)) / (k x)
    
    def h(x):
        exp_term = np.exp(-L * x)
        term1 = L * exp_term * (C - inv_k * np.log(x))
        term2 = (1.0 - exp_term) * inv_k / x
        return term1 - term2

    # Check boundary x=1 (d=0)
    # h(1) = L * 1 * (C - 0) - 0 = L * C
    if h(1.0) >= 0:
        # Derivative positive at x=1, so max is at x=1 (since x<=1)
        # Wait, if f'(1) > 0, f is increasing as x increases.
        # But x cannot exceed 1. So max is at x=1.
        return 0.0
    
    # If h(1) < 0, there is a root in (0, 1)
    # h(x) -> +inf as x -> 0
    # Use brentq
    try:
        x_star = brentq(h, 1e-12, 1.0, xtol=1e-12, rtol=1e-12)
    except ValueError:
        # Fallback to grid search if brentq fails
        x_grid = np.linspace(1e-6, 1.0, 1000)
        vals = np.array([h(x) for x in x_grid])
        # Find sign change
        idx = np.where(np.diff(np.sign(vals)) != 0)[0]
        if len(idx) > 0:
             x_star = brentq(h, x_grid[idx[0]], x_grid[idx[0]+1])
        else:
             # Should not happen if h(1)<0 and h(0)>0
             x_star = 1e-6

    d = -np.log(x_star) * inv_k
    return d

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
        
        # Compute C_b and C_a for all q
        # C_b[q] = V(q+1) - V(q)
        # C_a[q] = V(q-1) - V(q)
        
        C_b = np.full(n_q, -np.inf)
        C_a = np.full(n_q, -np.inf)
        
        # q < Q (idx < 2Q)
        C_b[:2*Q] = V_next[1:] - V_next[:-1]
        
        # q > -Q (idx > 0)
        C_a[1:] = V_next[:-1] - V_next[1:]

        # Solve for depths
        # We will iterate to handle the cross term p_b p_a
        # Initial guess for p_b, p_a
        p_b_curr = np.zeros(n_q)
        p_a_curr = np.zeros(n_q)
        
        b_vals = np.full(n_q, np.inf)
        a_vals = np.full(n_q, np.inf)
        
        # Only solve for interior q
        valid_b = np.arange(n_q) < 2*Q
        valid_a = np.arange(n_q) > 0
        
        # 2 iterations of fixed point
        for _ in range(2):
            # Solve b
            K_b = C_b - p_a_curr * (C_b + C_a)
            # For q where C_b is -inf, K_b is -inf, depth should be inf
            # _solve_depth handles C=-inf? 
            # If C is -inf, h(x) -> -inf? 
            # L exp(-Lx) (-inf) -> -inf.
            # So h(x) < 0 for all x. Max at x=0? 
            # If C is very negative, we don't want to quote.
            # _solve_depth returns d. If C is -inf, we want d=inf.
            
            # Vectorized approach is not possible with brentq directly.
            # Loop over valid q
            for q_idx in np.where(valid_b)[0]:
                C_val = C_b[q_idx]
                if not np.isfinite(C_val):
                    b_vals[q_idx] = np.inf
                    p_b_curr[q_idx] = 0.0
                    continue
                    
                d = _solve_depth(C_val, A, k, dt)
                b_vals[q_idx] = d
                p_b_curr[q_idx] = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                
            # Solve a
            K_a = C_a - p_b_curr * (C_b + C_a)
            for q_idx in np.where(valid_a)[0]:
                C_val = C_a[q_idx]
                if not np.isfinite(C_val):
                    a_vals[q_idx] = np.inf
                    p_a_curr[q_idx] = 0.0
                    continue
                    
                d = _solve_depth(C_val, A, k, dt)
                a_vals[q_idx] = d
                p_a_curr[q_idx] = 1.0 - np.exp(-A * np.exp(-k * d) * dt)

        opt_b[t, :] = b_vals
        opt_a[t, :] = a_vals
        
        # Update V
        # Value = cost + V_next + p_b(b + C_b) + p_a(a + C_a) - p_b p_a (C_b + C_a)
        # Note: p_b, p_a are for the solved depths.
        
        # Compute rewards
        reward_b = np.zeros(n_q)
        reward_a = np.zeros(n_q)
        
        for q_idx in np.where(valid_b)[0]:
             if np.isfinite(b_vals[q_idx]):
                 reward_b[q_idx] = p_b_curr[q_idx] * (b_vals[q_idx] + C_b[q_idx])
        
        for q_idx in np.where(valid_a)[0]:
             if np.isfinite(a_vals[q_idx]):
                 reward_a[q_idx] = p_a_curr[q_idx] * (a_vals[q_idx] + C_a[q_idx])
                 
        cross_term = p_b_curr * p_a_curr * (C_b + C_a)
        # Handle infs in C_b+C_a
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

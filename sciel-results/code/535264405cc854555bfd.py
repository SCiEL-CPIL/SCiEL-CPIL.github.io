import numpy as np
import time
from scipy.optimize import brentq

_cache = {}
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0

def _solve_regime(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]

    if time.time() > _DEADLINE:
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
    lambda_t = A * dt
    lambda_k = lambda_t * k
    n_q = 2 * Q + 1

    V = np.zeros((N + 1, n_q))
    opt_b = np.full((N, n_q), np.inf)
    opt_a = np.full((N, n_q), np.inf)

    q_vals = np.arange(-Q, Q + 1)
    cost = -0.5 * gamma * sigma**2 * q_vals**2 * dt

    def find_optimal_depth(C):
        if C <= 1e-9:
            return np.inf
        
        # f(b) = lambda_k * exp(-k*b) * (C + b) - (1 - exp(-k*b))
        # f(0) = lambda_k * C > 0
        # f(inf) = -1
        
        b_high = 1.0
        for _ in range(100):
            exp_term = np.exp(-k * b_high)
            f_val = lambda_k * exp_term * (C + b_high) - (1.0 - exp_term)
            if f_val < 0:
                break
            b_high *= 2.0
        else:
            return 0.0
            
        try:
            b_opt = brentq(lambda b: lambda_k * np.exp(-k * b) * (C + b) - (1.0 - np.exp(-k * b)), 1e-9, b_high, xtol=1e-8)
            return b_opt
        except ValueError:
            return 0.0

    for t in range(N - 1, -1, -1):
        if time.time() > _DEADLINE:
            for tt in range(t, -1, -1):
                opt_b[tt, :] = 0.05
                opt_a[tt, :] = 0.05
                opt_b[tt, 2 * Q] = np.inf
                opt_a[tt, 0] = np.inf
            _cache[key] = (opt_b, opt_a)
            return _cache[key]

        V_next = V[t + 1, :]

        C_b = np.zeros(n_q)
        C_b[:2*Q] = V_next[1:] - V_next[:-1]
        
        C_a = np.zeros(n_q)
        C_a[1:] = V_next[:-1] - V_next[1:]

        for i in range(2 * Q):
            opt_b[t, i] = find_optimal_depth(C_b[i])
        opt_b[t, 2*Q] = np.inf

        for i in range(1, n_q):
            opt_a[t, i] = find_optimal_depth(C_a[i])
        opt_a[t, 0] = np.inf

        V_curr = V_next.copy()
        
        b_arr = opt_b[t, :2*Q]
        valid_b = ~np.isinf(b_arr)
        if np.any(valid_b):
            b_v = b_arr[valid_b]
            C_v = C_b[:2*Q][valid_b]
            J_b_v = (1.0 - np.exp(-lambda_t * np.exp(-k * b_v))) * (C_v + b_v)
            idxs_b = np.where(valid_b)[0]
            V_curr[idxs_b] += J_b_v

        a_arr = opt_a[t, 1:]
        valid_a = ~np.isinf(a_arr)
        if np.any(valid_a):
            a_v = a_arr[valid_a]
            C_v = C_a[1:][valid_a]
            J_a_v = (1.0 - np.exp(-lambda_t * np.exp(-k * a_v))) * (C_v + a_v)
            idxs_a = np.where(valid_a)[0] + 1
            V_curr[idxs_a] += J_a_v

        V[t, :] = V_curr + cost

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

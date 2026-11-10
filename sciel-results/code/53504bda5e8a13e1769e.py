import numpy as np
import time

_start_time = time.time()
_DEADLINE = 310.0

_dp_cache = {}

def _solve_depth_1d(A, k, dt, delta_v, is_ask=False):
    """Find optimal depth via 1D search."""
    # Search range: [0, 15/k] should be sufficient
    n_grid = 300
    d_grid = np.linspace(0.0, 15.0 / k, n_grid)
    
    ex = np.exp(-k * d_grid)
    p = 1.0 - np.exp(-A * ex * dt)
    
    if is_ask:
        # Maximize p * (-d + delta_v)
        vals = p * (-d_grid + delta_v)
    else:
        # Maximize p * (d + delta_v)
        vals = p * (d_grid + delta_v)
    
    idx = np.argmax(vals)
    d_opt = d_grid[idx]
    
    # Refine
    if 0 < idx < n_grid - 1:
        d_lo = d_grid[idx - 1]
        d_hi = d_grid[idx + 1]
        d_fine = np.linspace(d_lo, d_hi, 200)
        ex_f = np.exp(-k * d_fine)
        p_f = 1.0 - np.exp(-A * ex_f * dt)
        if is_ask:
            vals_f = p_f * (-d_fine + delta_v)
        else:
            vals_f = p_f * (d_fine + delta_v)
        d_opt = d_fine[np.argmax(vals_f)]
    
    return max(0.0, float(d_opt))


def _precompute_dp(params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    V_next = np.zeros(n_q, dtype=np.float64)
    opt_d_bid = np.full((steps, n_q), np.inf, dtype=np.float64)
    opt_d_ask = np.full((steps, n_q), np.inf, dtype=np.float64)
    
    for step in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_q, dtype=np.float64)
        
        for qi in range(n_q):
            q = q_vals[qi]
            
            if q >= Q:
                # No bid
                opt_d_bid[step, qi] = np.inf
                if qi + 1 < n_q:
                    dv_a = V_next[qi + 1] - V_next[qi]
                    d_a = _solve_depth_1d(A, k, dt, dv_a, is_ask=True)
                else:
                    d_a = 1.0 / k
                opt_d_ask[step, qi] = d_a
                p_a = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
                if qi + 1 < n_q:
                    V_curr[qi] = V_next[qi] + p_a * (-d_a + (V_next[qi+1] - V_next[qi]))
                else:
                    V_curr[qi] = V_next[qi]
            elif q <= -Q:
                # No ask
                opt_d_ask[step, qi] = np.inf
                if qi - 1 >= 0:
                    dv_b = V_next[qi - 1] - V_next[qi]
                    d_b = _solve_depth_1d(A, k, dt, dv_b, is_ask=False)
                else:
                    d_b = 1.0 / k
                opt_d_bid[step, qi] = d_b
                p_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
                if qi - 1 >= 0:
                    V_curr[qi] = V_next[qi] + p_b * (d_b + (V_next[qi-1] - V_next[qi]))
                else:
                    V_curr[qi] = V_next[qi]
            else:
                dv_b = V_next[qi - 1] - V_next[qi]
                dv_a = V_next[qi + 1] - V_next[qi]
                
                d_b = _solve_depth_1d(A, k, dt, dv_b, is_ask=False)
                d_a = _solve_depth_1d(A, k, dt, dv_a, is_ask=True)
                
                opt_d_bid[step, qi] = d_b
                opt_d_ask[step, qi] = d_a
                
                p_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
                p_a = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
                
                imm = p_b * d_b - p_a * d_a - 0.5 * gamma * sigma**2 * q**2 * dt
                cont = (1 - p_b - p_a) * V_next[qi] + p_b * V_next[qi-1] + p_a * V_next[qi+1]
                V_curr[qi] = imm + cont
        
        V_next = V_curr
    
    return opt_d_bid, opt_d_ask


def quote(t, inventory, mid, params):
    if time.time() - _start_time > _DEADLINE:
        k = params["k"]
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    step_idx = min(int(round(t / dt)), steps - 1)
    
    # Check cache
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key not in _dp_cache:
        # Precompute
        opt_d_bid, opt_d_ask = _precompute_dp(params)
        _dp_cache[cache_key] = (opt_d_bid, opt_d_ask)
    
    opt_d_bid, opt_d_ask = _dp_cache[cache_key]
    
    q = np.asarray(inventory, dtype=np.int64)
    n_q = 2 * Q + 1
    
    # Map inventory to index
    q_idx = q + Q  # 0 to 2Q
    
    d_bid = opt_d_bid[step_idx, q_idx].astype(np.float64)
    d_ask = opt_d_ask[step_idx, q_idx].astype(np.float64)
    
    # Ensure valid range
    min_depth = -3.0 / k + 1e-6
    d_bid = np.where(d_bid < min_depth, min_depth, d_bid)
    d_ask = np.where(d_ask < min_depth, min_depth, d_ask)
    
    return (d_bid, d_ask)

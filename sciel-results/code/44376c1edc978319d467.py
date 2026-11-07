"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _solve_depth_newton(A, k, dt, delta, d0, max_iter=60, tol=1e-14):
    """
    Solve the FOC for optimal depth d given inventory change delta.
    f(d) = p(d) * (d + delta) where p(d) = 1 - exp(-A*exp(-k*d)*dt).
    FOC: p'(d)*(d+delta) + p(d) = 0
    """
    d = max(0.0, min(d0, 15.0 / k))
    
    for _ in range(max_iter):
        exp_kd = np.exp(-k * d)
        lam = A * exp_kd * dt
        
        if lam < 1e-10:
            p = lam
            exp_neg_lam = 1.0 - lam + 0.5 * lam * lam
        elif lam < 1e-4:
            p = -np.expm1(-lam)
            exp_neg_lam = np.exp(-lam)
        else:
            exp_neg_lam = np.exp(-lam)
            p = 1.0 - exp_neg_lam
        
        p_prime = k * lam * exp_neg_lam
        g = p_prime * (d + delta) + p
        
        if abs(g) < tol:
            break
        
        p_double_prime = -k * k * lam * exp_neg_lam * (1.0 - lam)
        g_prime = p_double_prime * (d + delta) + p_prime
        
        if abs(g_prime) < 1e-30:
            break
        
        newton_step = g / g_prime
        d_new = d - newton_step
        
        if d_new < 0:
            d_new = 0.0
        elif d_new > 15.0 / k:
            d_new = 15.0 / k
        
        if abs(d_new - d) < tol:
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
    
    # Denser grid for better initial guesses
    d_max = 15.0 / k
    num_points = 500
    D_grid = np.linspace(0.0, d_max, num_points)
    
    exp_term = np.exp(-k * D_grid)
    lam_grid = A * exp_term * dt
    p_grid = -np.expm1(-lam_grid)  # 1 - exp(-lam) more accurately
    
    def _optimal_depth(delta_val):
        """Find optimal depth for a given delta."""
        vals = p_grid * (D_grid + delta_val)
        max_val = np.max(vals)
        
        if max_val <= 0:
            return np.inf, 0.0
        
        idx = np.argmax(vals)
        d0 = D_grid[idx]
        
        # Better initial guess: adjust based on delta
        if delta_val > 0:
            # Want more aggressive (smaller depth)
            d0 = min(d0, 2.0 / k)
        elif delta_val < 0:
            # Want more conservative (larger depth)
            d0 = max(d0, 1.0 / k)
        
        d_opt = _solve_depth_newton(A, k, dt, delta_val, d0)
        
        # Evaluate at Newton solution
        exp_kd = np.exp(-k * d_opt)
        lam = A * exp_kd * dt
        if lam < 1e-10:
            p_opt = lam
        else:
            p_opt = -np.expm1(-lam)
        
        f_opt = p_opt * (d_opt + delta_val)
        
        # Safety: if Newton went wrong, fall back to grid best
        if f_opt < max_val - 1e-10:
            d_opt = d0
            f_opt = max_val
        
        # Verify it's a maximum by checking neighbors
        if f_opt > 0:
            # Check if this is really a max
            d_check1 = max(0, d_opt - 0.01 / k)
            d_check2 = min(d_max, d_opt + 0.01 / k)
            
            for d_chk in [d_check1, d_check2]:
                exp_kd_chk = np.exp(-k * d_chk)
                lam_chk = A * exp_kd_chk * dt
                if lam_chk < 1e-10:
                    p_chk = lam_chk
                else:
                    p_chk = -np.expm1(-lam_chk)
                f_chk = p_chk * (d_chk + delta_val)
                if f_chk > f_opt + 1e-12:
                    f_opt = f_chk
                    d_opt = d_chk
        
        if f_opt <= 0:
            return np.inf, 0.0
        
        return d_opt, f_opt
    
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        V_next = V[n + 1]
        
        delta_b = np.full(2 * Q + 1, 0.0)
        if Q > 0:
            delta_b[:-1] = V_next[1:] - V_next[:-1]
            delta_b[-1] = 0.0
        
        delta_a = np.full(2 * Q + 1, 0.0)
        if Q > 0:
            delta_a[1:] = V_next[:-1] - V_next[1:]
            delta_a[0] = 0.0
        
        bid_depths = np.full(2 * Q + 1, np.inf)
        ask_depths = np.full(2 * Q + 1, np.inf)
        bid_rewards = np.zeros(2 * Q + 1)
        ask_rewards = np.zeros(2 * Q + 1)
        
        for q_idx in range(2 * Q + 1):
            if q_idx < 2 * Q:
                d_b, r_b = _optimal_depth(delta_b[q_idx])
                bid_depths[q_idx] = d_b
                bid_rewards[q_idx] = r_b
            
            if q_idx > 0:
                d_a, r_a = _optimal_depth(delta_a[q_idx])
                ask_depths[q_idx] = d_a
                ask_rewards[q_idx] = r_a
        
        if Q > 0:
            bid_depths[-1] = np.inf
            bid_rewards[-1] = 0.0
            ask_depths[0] = np.inf
            ask_rewards[0] = 0.0
        
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

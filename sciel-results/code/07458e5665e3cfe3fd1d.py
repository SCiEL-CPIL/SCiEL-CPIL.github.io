"""Market making with inventory risk - improved DP with Newton polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0


def _solve_depth_newton(A, k, dt, delta, d0, max_iter=50, tol=1e-14):
    """
    Solve FOC: f'(d) = 0 where f(d) = p(d)*(d+delta)
    p(d) = 1 - exp(-A*exp(-k*d)*dt)
    
    g(d) = p'(d)*(d+delta) + p(d) = 0
    p'(d) = k*lambda*exp(-lambda) where lambda = A*exp(-k*d)*dt
    """
    d = d0
    for _ in range(max_iter):
        lam = A * np.exp(-k * d) * dt
        if lam < 1e-15:
            # p(d) ≈ 0, p'(d) ≈ 0, g(d) ≈ 0
            break
        exp_neg_lam = np.exp(-lam)
        p = 1.0 - exp_neg_lam
        p_prime = k * lam * exp_neg_lam
        
        g = p_prime * (d + delta) + p
        
        # g'(d) = p''(d)*(d+delta) + p'(d)
        # p''(d) = derivative of k*lambda*exp(-lambda) w.r.t. d
        # lambda' = -k*lambda
        # d/dlambda [k*lambda*exp(-lambda)] = k*exp(-lambda) - k*lambda*exp(-lambda) = k*exp(-lambda)*(1-lambda)
        # p'' = k*exp(-lambda)*(1-lambda) * (-k*lambda) = -k^2*lambda*exp(-lambda)*(1-lambda)
        p_double_prime = -k * k * lam * exp_neg_lam * (1.0 - lam)
        
        g_prime = p_double_prime * (d + delta) + p_prime
        
        if abs(g_prime) < 1e-30:
            break
        
        step = g / g_prime
        d_new = d - step
        
        # Keep d >= 0
        if d_new < 0:
            d_new = 0.0
        # Keep d reasonable
        if d_new > 100.0 / k:
            d_new = 100.0 / k
        
        if abs(d_new - d) < tol:
            d = d_new
            break
        d = d_new
    
    return max(d, 0.0)


def _get_dp_tables(params):
    """Compute optimal bid/ask depth tables via dynamic programming with Newton polish."""
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
    
    # Value table
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal depth tables
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)
    
    # Precompute risk penalty
    q_indices = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma**2 * q_indices**2 * dt
    
    # Grid for initial depth search
    d_max = 12.0 / k
    num_points = 800
    D_grid = np.linspace(0.0, d_max, num_points)
    
    # Precompute fill probabilities for the grid
    exp_term = np.exp(-k * D_grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            # Return what we have; remaining steps stay as inf
            _cache[key] = (opt_bid, opt_ask)
            return opt_bid, opt_ask
        
        V_next = V[n + 1]
        
        # Delta V for bid: q -> q+1
        delta_b = np.full(2 * Q + 1, 0.0)
        if Q > 0:
            delta_b[:-1] = V_next[1:] - V_next[:-1]
        
        # Delta V for ask: q -> q-1
        delta_a = np.full(2 * Q + 1, 0.0)
        if Q > 0:
            delta_a[1:] = V_next[:-1] - V_next[1:]
        
        # Vectorized optimization for Bid
        vals_b = p_grid[None, :] * (D_grid[None, :] + delta_b[:, None])
        max_vals_b = np.max(vals_b, axis=1)
        argmax_idx_b = np.argmax(vals_b, axis=1)
        
        bid_depths = np.full(2 * Q + 1, np.inf)
        valid_b = max_vals_b > 1e-12
        if np.any(valid_b):
            grid_depths_b = D_grid[argmax_idx_b[valid_b]]
            # Newton polish for each valid state
            for idx in np.where(valid_b)[0]:
                d0 = grid_depths_b[idx - np.sum(valid_b[:idx])]
                delta = delta_b[idx]
                d_new = _solve_depth_newton(A, k, dt, delta, d0)
                # Verify improvement
                lam = A * np.exp(-k * d_new) * dt
                p_new = 1.0 - np.exp(-lam)
                val_new = p_new * (d_new + delta)
                if val_new > max_vals_b[idx]:
                    bid_depths[idx] = d_new
                    max_vals_b[idx] = val_new
                else:
                    bid_depths[idx] = d0
        
        # Vectorized optimization for Ask
        vals_a = p_grid[None, :] * (D_grid[None, :] + delta_a[:, None])
        max_vals_a = np.max(vals_a, axis=1)
        argmax_idx_a = np.argmax(vals_a, axis=1)
        
        ask_depths = np.full(2 * Q + 1, np.inf)
        valid_a = max_vals_a > 1e-12
        if np.any(valid_a):
            grid_depths_a = D_grid[argmax_idx_a[valid_a]]
            for idx in np.where(valid_a)[0]:
                d0 = grid_depths_a[idx - np.sum(valid_a[:idx])]
                delta = delta_a[idx]
                d_new = _solve_depth_newton(A, k, dt, delta, d0)
                lam = A * np.exp(-k * d_new) * dt
                p_new = 1.0 - np.exp(-lam)
                val_new = p_new * (d_new + delta)
                if val_new > max_vals_a[idx]:
                    ask_depths[idx] = d_new
                    max_vals_a[idx] = val_new
                else:
                    ask_depths[idx] = d0
        
        # Handle boundary conditions
        if Q > 0:
            bid_depths[-1] = np.inf
            ask_depths[0] = np.inf
        
        # Rewards
        bid_rewards = np.maximum(max_vals_b, 0.0)
        ask_rewards = np.maximum(max_vals_a, 0.0)
        
        if Q > 0:
            bid_rewards[-1] = 0.0
            ask_rewards[0] = 0.0
        
        V[n] = V_next - risk + bid_rewards + ask_rewards
        opt_bid[n] = bid_depths
        opt_ask[n] = ask_depths
    
    _cache[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """Quote optimal depths based on dynamic programming solution."""
    # Deadline guard
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        n_paths = len(inventory)
        return np.full(n_paths, np.inf), np.full(n_paths, np.inf)
    
    opt_bid, opt_ask = _get_dp_tables(params)
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    # Current step index
    n = int(t / dt)
    n = min(n, steps - 1)
    
    # Map inventory to indices
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    # Safety: ensure valid values
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 0), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 0), ask_depths, np.inf)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)

# EVOLVE-BLOCK-END

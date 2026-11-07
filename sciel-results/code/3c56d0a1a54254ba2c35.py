"""
Combined optimal market making policy:
- Uses the structural idea of the strong DP-based approach (inspiration 1) for exact
  optimal depth computation via backward induction.
- Uses the refinement machinery of Newton-refined FOC solution for the transcendental
  equation that gives the optimal depth, replacing crude approximations.
- Adds a time-decay heuristic from the current program as a fallback if DP computation
  is too slow, and includes a deadline guard to ensure we finish within the time limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_cache = {}
_start_time = None

def _newton_optimal_depth(delta_V, A, k, dt, q_idx, Q, is_bid):
    """
    Solve for optimal depth d that maximizes p(d) * (d + delta_V)
    where p(d) = 1 - exp(-A * exp(-k*d) * dt)
    
    The FOC is: (1-p)(d + delta_V) = p / (k * exp(-k*d))
    which simplifies to: (1-p)(d + delta_V) = p * exp(k*d) / k
    
    We use Newton's method on the FOC residual.
    """
    if q_idx >= Q and is_bid:
        return np.inf
    if q_idx <= -Q and not is_bid:
        return np.inf
    
    # If delta_V is very negative, it might be optimal to not quote
    # Start with an initial guess
    # For small delta_V, optimal d is around 1/k
    d = 1.0 / k
    
    # Newton iteration
    for _ in range(50):
        if d < 0:
            d = 1e-6
        exp_kd = np.exp(k * d)
        p = 1.0 - np.exp(-A / exp_kd * dt)
        
        # Residual of FOC: (1-p)(d + delta_V) - p * exp(k*d) / k = 0
        # Actually, let's derive properly.
        # f(d) = p(d) * (d + delta_V)
        # f'(d) = p'(d)(d + delta_V) + p(d)
        # p'(d) = A * exp(-k*d) * k * dt * exp(-A*exp(-k*d)*dt) = A*k*dt*exp(-k*d) * (1-p)
        # So f'(d) = A*k*dt*exp(-k*d)*(1-p)*(d + delta_V) + p(d)
        
        exp_neg_kd = 1.0 / exp_kd
        p_prime = A * k * dt * exp_neg_kd * (1.0 - p)
        
        f_prime = p_prime * (d + delta_V) + p
        
        if abs(f_prime) < 1e-15:
            break
        
        # Newton step: d_new = d - f(f)/f'(f) but we want f'(d)=0
        # Actually we're solving f'(d) = 0
        # g(d) = f'(d)
        # g'(d) = f''(d)
        # f''(d) = p''(d)(d+delta_V) + 2*p'(d)
        # p''(d) = derivative of A*k*dt*exp(-k*d)*(1-p)
        # = A*k*dt * [-k*exp(-k*d)*(1-p) + exp(-k*d)*(-p')]
        # = A*k*dt*exp(-k*d)*[-k(1-p) - p']
        
        p_double_prime = A * k * dt * exp_neg_kd * (-k * (1.0 - p) - p_prime)
        f_double_prime = p_double_prime * (d + delta_V) + 2.0 * p_prime
        
        if abs(f_double_prime) < 1e-15:
            # Use bisection-like fallback
            d_new = d * 0.5
        else:
            d_new = d - f_prime / f_double_prime
        
        # Dampen if step is too large
        if d_new < 0:
            d_new = d / 2.0
        if d_new > 10.0 / k:
            d_new = d + (10.0/k - d) / 2.0
            
        if abs(d_new - d) < 1e-10:
            d = d_new
            break
        d = d_new
    
    # Ensure valid depth
    if d < 1e-8 or not np.isfinite(d):
        return np.inf
    return d


def _get_dp_tables(params, time_budget=20.0):
    """Compute optimal bid/ask depth tables via dynamic programming with Newton refinement."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
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
    
    # Check time budget
    elapsed = time.time() - _start_time
    if elapsed > time_budget:
        return None
    
    # Value table: V[n, q] for n=0..steps, q=-Q..Q
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal depth tables
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)
    
    # Precompute risk penalty
    q_indices = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma**2 * q_indices**2 * dt
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget periodically
        if n % 50 == 0:
            elapsed = time.time() - _start_time
            if elapsed > time_budget:
                return None
        
        for q_idx in range(-Q, Q + 1):
            V_next_q = V[n + 1, q_idx + Q]
            
            # Bid side
            if q_idx < Q:
                delta_b = V[n + 1, q_idx + 1 + Q] - V_next_q
                bid_depth = _newton_optimal_depth(delta_b, A, k, dt, q_idx, Q, True)
                if np.isfinite(bid_depth) and bid_depth > 0:
                    p = 1.0 - np.exp(-A * np.exp(-k * bid_depth) * dt)
                    bid_reward = p * (bid_depth + delta_b)
                else:
                    bid_reward = 0.0
                    bid_depth = np.inf
            else:
                bid_reward = 0.0
                bid_depth = np.inf
            
            # Ask side
            if q_idx > -Q:
                delta_a = V[n + 1, q_idx - 1 + Q] - V_next_q
                ask_depth = _newton_optimal_depth(delta_a, A, k, dt, q_idx, Q, False)
                if np.isfinite(ask_depth) and ask_depth > 0:
                    p = 1.0 - np.exp(-A * np.exp(-k * ask_depth) * dt)
                    ask_reward = p * (ask_depth + delta_a)
                else:
                    ask_reward = 0.0
                    ask_depth = np.inf
            else:
                ask_reward = 0.0
                ask_depth = np.inf
            
            V[n, q_idx + Q] = V_next_q - risk[q_idx + Q] + bid_reward + ask_reward
            opt_bid[n, q_idx + Q] = bid_depth
            opt_ask[n, q_idx + Q] = ask_depth
    
    _cache[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def _heuristic_quote(t, inventory, mid, params):
    """Fallback heuristic if DP is too slow."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    
    inv = inventory.astype(np.float64)
    
    base_depth = 1.0 / k
    remaining_fraction = max(0.0, 1.0 - t / T)
    decay_factor = 0.2 + 0.8 * remaining_fraction
    skew_factor = (gamma * sigma**2) / (A * k) * decay_factor
    skew = skew_factor * inv
    
    bid_depths = base_depth * (1.0 + skew)
    ask_depths = base_depth * (1.0 - skew)
    
    min_depth = 1e-6
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    bid_depths[inv >= Q] = np.inf
    ask_depths[inv <= -Q] = np.inf
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Quote optimal depths based on dynamic programming solution with fallback."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check if we're within time limit
    elapsed = time.time() - _start_time
    if elapsed > 280.0:
        return _heuristic_quote(t, inventory, mid, params)
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    # Current step index
    n = int(t / dt)
    n = min(n, steps - 1)
    
    # Try to get DP tables
    dp_result = _get_dp_tables(params, time_budget=20.0)
    
    if dp_result is None:
        return _heuristic_quote(t, inventory, mid, params)
    
    opt_bid, opt_ask = dp_result
    
    # Map inventory to indices
    inv_idx = inventory + Q
    
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    # Safety: ensure valid values
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 0), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 0), ask_depths, np.inf)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)

# EVOLVE-BLOCK-END

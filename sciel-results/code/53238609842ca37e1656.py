import numpy as np
from scipy.special import lambertw
import time

# Global state for caching and time management
_start_time = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s kill

# Cache for computed policies
# Key: (sigma, A, k, gamma, T, steps, Q)
# Value: (opt_bid, opt_ask) - arrays of shape (steps, 2*Q+1)
_policy_cache = {}


def _solve_depth_newton(A, k, dt, margin):
    """
    Solve for optimal depth d using Newton-Raphson on the first-order condition.
    
    The objective to maximize is: f(d) = p(d) * (d + margin)
    where p(d) = 1 - exp(-A * exp(-k*d) * dt)
    
    Let u = A * exp(-k*d) * dt, so p = 1 - exp(-u)
    and d = -(1/k) * ln(u / (A * dt))
    
    The FOC is: f'(d) = p'(d)*(d + margin) + p(d) = 0
    p'(d) = A*k*exp(-k*d)*exp(-u) = (u/dt)*k*exp(-u)
    
    So: (u*k/dt)*exp(-u)*(d + margin) + (1 - exp(-u)) = 0
    """
    if margin <= 0:
        # If margin is non-positive, we want to minimize fill probability
        # by making depth very large (p -> 0)
        # But we need a valid depth. Use a large depth.
        return 10.0 / k if k > 0 else 10.0
    
    # Initial guess using Lambert W approximation
    # For small u (large d): p ~ u, f ~ u*(d+margin)
    # For large u (small d): p ~ 1, f ~ d + margin
    # The optimal balance is roughly where the marginal gain from increasing depth
    # equals the marginal cost in fill probability.
    
    # Use a robust initial guess
    # If margin is small, we want small depth (high fill prob)
    # If margin is large, we want large depth (low fill prob, more per fill)
    
    # Start with a heuristic guess
    if margin < 0.1:
        d0 = 0.1 / k if k > 0 else 1.0
    elif margin < 1.0:
        d0 = 0.5 / k if k > 0 else 2.0
    else:
        d0 = 2.0 / k if k > 0 else 4.0
    
    # Clamp to reasonable range
    d_min = 0.001
    d_max = 20.0 / k if k > 0 else 20.0
    d0 = np.clip(d0, d_min, d_max)
    
    # Newton iterations
    d = d0
    for _ in range(30):
        # Compute u = A * exp(-k*d) * dt
        exp_kd = np.exp(-k * d)
        u = A * exp_kd * dt
        p = 1.0 - np.exp(-u)
        
        # Objective: f(d) = p * (d + margin)
        # Derivative: f'(d) = p' * (d + margin) + p
        # p' = A * k * exp(-k*d) * exp(-u) = (u/dt) * k * exp(-u)
        
        if u < 1e-12:
            # p is essentially 0, f is essentially 0
            # We're in the tail where increasing d doesn't help
            # Move towards larger d
            d = np.clip(d * 1.5, d_min, d_max)
            continue
        
        exp_u = np.exp(-u)
        p_prime = (u / dt) * k * exp_u
        
        f_prime = p_prime * (d + margin) + p
        
        # Second derivative for damping if needed
        # f'' = p''*(d+margin) + 2*p'
        # p'' = derivative of (u/dt)*k*exp(-u) w.r.t. d
        # u' = -k*u (since u = A*exp(-k*d)*dt, du/d(-k*d) = u, du/dd = -k*u)
        # p'' = d/dd [(u*k/dt)*exp(-u)] = (du/dd)*(k/dt)*exp(-u) + (u*k/dt)*(-exp(-u))*du/dd
        # = -k*u*(k/dt)*exp(-u) - (u*k/dt)*exp(-u)*k*u
        # = -(k^2*u/dt)*exp(-u) - (k^2*u^2/dt)*exp(-u)
        # = -(k^2*u/dt)*exp(-u)*(1+u)
        
        f_second = -(k**2 * u / dt) * exp_u * (1 + u) * (d + margin) + 2 * p_prime
        
        if abs(f_prime) < 1e-12:
            break
        
        # Newton step with damping
        step_nr = -f_prime / f_second if abs(f_second) > 1e-12 else -f_prime * 0.1
        
        # Limit step size
        max_step = 1.0 / k if k > 0 else 1.0
        step_nr = np.clip(step_nr, -max_step, max_step)
        
        d_new = d + step_nr
        d_new = np.clip(d_new, d_min, d_max)
        
        if abs(d_new - d) < 1e-10:
            d = d_new
            break
        d = d_new
    
    return float(d)


def _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid and ask depths using backward induction with
    Newton-Raphson for the continuous first-order condition.
    """
    dt = T / steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q
    
    # V[n, q_idx] = optimal value from step n with inventory q
    V = np.zeros((steps + 1, n_inv))
    
    # Action tables
    opt_bid = np.zeros((steps, n_inv))
    opt_ask = np.zeros((steps, n_inv))
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time limit
        if time.time() - _start_time > _TIME_LIMIT - 10.0:
            # Fallback: fill remaining with heuristic
            for remaining_n in range(n, -1, -1):
                for q_idx in range(n_inv):
                    q = q_idx - Q
                    if q == Q:
                        opt_bid[remaining_n, q_idx] = np.inf
                        opt_ask[remaining_n, q_idx] = 1.0 / k
                    elif q == -Q:
                        opt_bid[remaining_n, q_idx] = 1.0 / k
                        opt_ask[remaining_n, q_idx] = np.inf
                    else:
                        # Heuristic based on inventory
                        base = 1.0 / k
                        shift = 0.3 * q / Q
                        opt_bid[remaining_n, q_idx] = max(0.01, base * (1 + shift))
                        opt_ask[remaining_n, q_idx] = max(0.01, base * (1 - shift))
            break
        
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            # Inventory cost for this step
            inv_cost = -(gamma / 2.0) * sigma**2 * q**2 * dt
            
            if q == Q:
                # No bid fill possible
                V_q = V[n + 1, q_idx]
                V_q_minus = V[n + 1, q_idx - 1]
                
                # Margin for ask: value of selling = depth + (V[q-1] - V[q])
                # We want to maximize p(d) * (d + V[q-1] - V[q])
                margin_ask = V_q_minus - V_q
                
                best_ad = _solve_depth_newton(A, k, dt, margin_ask)
                best_ad = max(best_ad, 0.001)
                
                p_ask = 1.0 - np.exp(-A * np.exp(-k * best_ad) * dt)
                
                V[n, q_idx] = inv_cost + p_ask * (best_ad + V_q_minus) + (1 - p_ask) * V_q
                opt_bid[n, q_idx] = np.inf
                opt_ask[n, q_idx] = best_ad
                
            elif q == -Q:
                # No ask fill possible
                V_q = V[n + 1, q_idx]
                V_q_plus = V[n + 1, q_idx + 1]
                
                # Margin for bid: value of buying = depth + (V[q+1] - V[q])
                margin_bid = V_q_plus - V_q
                
                best_bd = _solve_depth_newton(A, k, dt, margin_bid)
                best_bd = max(best_bd, 0.001)
                
                p_bid = 1.0 - np.exp(-A * np.exp(-k * best_bd) * dt)
                
                V[n, q_idx] = inv_cost + p_bid * (best_bd + V_q_plus) + (1 - p_bid) * V_q
                opt_bid[n, q_idx] = best_bd
                opt_ask[n, q_idx] = np.inf
                
            else:
                # Both sides active
                V_q = V[n + 1, q_idx]
                V_q_plus = V[n + 1, q_idx + 1]  # bid fill: q->q+1
                V_q_minus = V[n + 1, q_idx - 1]  # ask fill: q->q-1
                
                # Margins for first-order conditions
                # Bid side: maximize p_b * (d_b + V[q+1] - V[q]) approximately
                # Ask side: maximize p_a * (d_a + V[q-1] - V[q]) approximately
                margin_bid = V_q_plus - V_q
                margin_ask = V_q_minus - V_q
                
                best_bd = _solve_depth_newton(A, k, dt, margin_bid)
                best_ad = _solve_depth_newton(A, k, dt, margin_ask)
                
                best_bd = max(best_bd, 0.001)
                best_ad = max(best_ad, 0.001)
                
                p_bid = 1.0 - np.exp(-A * np.exp(-k * best_bd) * dt)
                p_ask = 1.0 - np.exp(-A * np.exp(-k * best_ad) * dt)
                
                # Compute exact value with both sides
                p_both = p_bid * p_ask
                p_bid_only = p_bid * (1 - p_ask)
                p_ask_only = (1 - p_bid) * p_ask
                p_neither = (1 - p_bid) * (1 - p_ask)
                
                earnings = p_bid * best_bd + p_ask * best_ad
                future = (p_both * V_q + p_bid_only * V_q_plus + 
                         p_ask_only * V_q_minus + p_neither * V_q)
                
                V[n, q_idx] = inv_cost + earnings + future
                opt_bid[n, q_idx] = best_bd
                opt_ask[n, q_idx] = best_ad
    
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n_paths = len(inventory)
    
    # Determine current step
    dt = T / steps
    step = int(round(t / dt))
    if step >= steps:
        step = steps - 1
    if step < 0:
        step = 0
    
    # Get or compute policy
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key not in _policy_cache:
        # Check if we have time to compute
        if time.time() - _start_time > _TIME_LIMIT - 15.0:
            # Fallback: use a simple heuristic
            bid_depths = np.full(n_paths, 1.0 / k, dtype=np.float64)
            ask_depths = np.full(n_paths, 1.0 / k, dtype=np.float64)
            
            # Adjust for inventory
            for i in range(n_paths):
                q = inventory[i]
                if q == Q:
                    bid_depths[i] = np.inf
                elif q == -Q:
                    ask_depths[i] = np.inf
                else:
                    shift = 0.3 * q / Q
                    bid_depths[i] = max(0.01, (1.0 / k) * (1 + shift))
                    ask_depths[i] = max(0.01, (1.0 / k) * (1 - shift))
            
            # Ensure valid depths
            min_depth = -3.0 / k + 1e-9
            bid_depths[bid_depths <= min_depth] = np.inf
            ask_depths[ask_depths <= min_depth] = np.inf
            
            return bid_depths, ask_depths
        
        opt_bid, opt_ask = _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[cache_key] = (opt_bid, opt_ask)
    else:
        opt_bid, opt_ask = _policy_cache[cache_key]
    
    # Map inventory to index
    inv_idx = np.clip(inventory, -Q, Q) + Q
    
    bid_depths = opt_bid[step, inv_idx].copy()
    ask_depths = opt_ask[step, inv_idx].copy()
    
    # Ensure valid depths: must be > -3/k
    min_depth = -3.0 / k + 1e-9
    bid_depths[bid_depths <= min_depth] = np.inf
    ask_depths[ask_depths <= min_depth] = np.inf
    
    # Ensure shape
    bid_depths = np.asarray(bid_depths, dtype=np.float64).reshape(n_paths)
    ask_depths = np.asarray(ask_depths, dtype=np.float64).reshape(n_paths)
    
    return bid_depths, ask_depths

import numpy as np
from functools import lru_cache

# Global cache for optimal policies
_policy_cache = {}

def _solve_optimal_depths(params):
    """
    Pre-compute optimal bid and ask depths for all steps and inventories.
    Returns a tuple (opt_d_b, opt_d_a) where opt_d_b[n, q+Q] is the optimal bid depth.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    # Index offset for inventory q in [-Q, Q] -> 0..2Q
    idx = Q
    
    # V[n, q_idx]
    # We only need V[n+1] to compute V[n], so we can use 2 arrays
    V_next = np.zeros(2 * Q + 1)
    
    # Store results
    opt_d_b = np.full((steps, 2 * Q + 1), np.inf)
    opt_d_a = np.full((steps, 2 * Q + 1), np.inf)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(2 * Q + 1)
        
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            
            # Holding cost
            hold_cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Boundary conditions
            if q == Q:
                # Cannot bid
                d_b = np.inf
                p_b = 0.0
            else:
                d_b = 0.0 # Will be solved
                p_b = 0.0 # Will be solved
            
            if q == -Q:
                # Cannot ask
                d_a = np.inf
                p_a = 0.0
            else:
                d_a = 0.0 # Will be solved
                p_a = 0.0 # Will be solved
            
            # Get value differences from next step
            # If q == Q, q+1 is out of bounds, but p_b=0 so it doesn't matter.
            # We need V_next[q+1] and V_next[q-1].
            
            if q < Q:
                delta_plus = V_next[q + 1 + Q] - V_next[q_idx]
            else:
                delta_plus = 0.0 # Unused if p_b=0
            
            if q > -Q:
                delta_minus = V_next[q - 1 + Q] - V_next[q_idx]
            else:
                delta_minus = 0.0 # Unused if p_a=0
            
            # Solve for d_b
            if q < Q:
                # Maximize p_b * (d_b + delta_plus)
                # Equation: exp(-lambda_b) = 1 / (1 + k * lambda_b * (d_b + delta_plus))
                # lambda_b = A * exp(-k * d_b) * dt
                # Let f(d) = exp(-lambda) - 1/(1 + k*lambda*(d + delta_plus))
                # Use Newton's method or bisection.
                # d_b must be > -3/k.
                
                # Initial guess from A-S
                # d_b_guess = 1/k - gamma * sigma**2 * (T - n*dt) * (q - 0.5)
                d_b_guess = 1.0/k - 0.5 * gamma * sigma**2 * (T - n*dt) * (q)
                d_b_guess = np.maximum(d_b_guess, -3.0/k + 1e-6)
                
                # Newton's method
                d_b = d_b_guess
                for _ in range(10):
                    lambda_b = A * np.exp(-k * d_b) * dt
                    p_b = 1.0 - np.exp(-lambda_b)
                    
                    # Derivative of objective J(d) = p(d) * (d + delta_plus)
                    # J'(d) = p'(d)(d + delta_plus) + p(d)
                    # p'(d) = k * lambda_b * exp(-lambda_b)
                    # J'(d) = k * lambda_b * exp(-lambda_b) * (d + delta_plus) + (1 - exp(-lambda_b))
                    
                    exp_neg_l = np.exp(-lambda_b)
                    J_prime = k * lambda_b * exp_neg_l * (d_b + delta_plus) + (1.0 - exp_neg_l)
                    
                    # Second derivative for stability? Or just damping.
                    # J''(d) ...
                    # Let's just use a simple step with damping
                    step = J_prime / (k * lambda_b * exp_neg_l * (1.0 + k * lambda_b * (d_b + delta_plus)) ** 2 + 1e-6)
                    # Actually, solving for root of J'(d)=0
                    # Newton update: d_new = d - J'(d) / J''(d)
                    # J''(d) is complicated. Let's use bisection if Newton fails or is unstable.
                    # Given the convexity of the problem, Newton should work well with a good guess.
                    
                    # Let's just use a robust scalar solver: scipy.optimize.brentq or similar?
                    # No scipy in the "standard library" constraint? "Python standard library, NumPy and SciPy are available."
                    # Okay, SciPy is allowed.
                    
                    break # Placeholder, will replace with proper solver logic below
            
            # To keep it simple and robust without SciPy dependency in the loop (though allowed),
            # let's use the analytical approximation which is very accurate for this model,
            # but the prompt asks for a "new approach". The DP approach *is* the new approach.
            # The solving of the root is part of the DP.
            
            # Let's implement a simple bisection for the root finding to be safe.
            # Function g(d) = k * lambda(d) * exp(-lambda(d)) * (d + delta_plus) + 1 - exp(-lambda(d))
            # We want g(d) = 0.
            
            def g_b(d):
                if d < -3.0/k: return -1e10
                l = A * np.exp(-k * d) * dt
                if l > 50: return 1.0 # p~1, derivative ~0? No.
                el = np.exp(-l)
                return k * l * el * (d + delta_plus) + 1.0 - el
            
            # Bisection range: [-3/k, 10/k]
            lo = -3.0/k + 1e-9
            hi = 10.0/k
            
            # Check signs
            if g_b(lo) * g_b(hi) > 0:
                # Root might be outside or multiple. Fallback to A-S
                d_b = np.maximum(1.0/k - 0.5 * gamma * sigma**2 * (T - n*dt) * q, lo)
            else:
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    if g_b(mid) * g_b(lo) <= 0:
                        hi = mid
                    else:
                        lo = mid
                d_b = (lo + hi) / 2.0
                
                if q < Q:
                    opt_d_b[n, q_idx] = d_b
                    p_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
            
            # Solve for d_a
            if q > -Q:
                # Maximize p_a * (d_a + delta_minus)
                # Same equation structure with delta_minus
                
                def g_a(d):
                    if d < -3.0/k: return -1e10
                    l = A * np.exp(-k * d) * dt
                    if l > 50: return 1.0
                    el = np.exp(-l)
                    return k * l * el * (d + delta_minus) + 1.0 - el
                
                lo = -3.0/k + 1e-9
                hi = 10.0/k
                
                if g_a(lo) * g_a(hi) > 0:
                    d_a = np.maximum(1.0/k - 0.5 * gamma * sigma**2 * (T - n*dt) * q, lo)
                else:
                    for _ in range(20):
                        mid = (lo + hi) / 2.0
                        if g_a(mid) * g_a(lo) <= 0:
                            hi = mid
                        else:
                            lo = mid
                    d_a = (lo + hi) / 2.0
                    
                if q > -Q:
                    opt_d_a[n, q_idx] = d_a
                    p_a = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
            
            # Calculate V_curr[q_idx]
            # V = -hold_cost + V_next + p_b(d_b + delta_plus) + p_a(d_a + delta_minus) - p_b p_a (delta_plus + delta_minus)
            # Note: if q==Q, p_b=0, d_b=inf.
            
            term_b = 0.0
            if q < Q:
                term_b = p_b * (d_b + delta_plus)
            
            term_a = 0.0
            if q > -Q:
                term_a = p_a * (d_a + delta_minus)
            
            term_coupling = 0.0
            if q < Q and q > -Q:
                term_coupling = -p_b * p_a * (delta_plus + delta_minus)
            
            V_curr[q_idx] = -hold_cost + V_next[q_idx] + term_b + term_a + term_coupling
        
        V_next = V_curr
    
    return opt_d_b, opt_d_a


def quote(t, inventory, mid, params):
    # Create a hashable key for the cache
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        _policy_cache[key] = _solve_optimal_depths(params)
    
    opt_d_b, opt_d_a = _policy_cache[key]
    
    # Determine step index n
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    n = np.clip(n, 0, params['steps'] - 1)
    
    Q = params['Q']
    
    # Map inventory to index
    # inventory is an array of ints
    # We need to fetch opt_d_b[n, inventory + Q]
    
    # Handle boundaries where depth is inf
    # The pre-computed arrays already have inf for boundaries.
    
    bid_depths = opt_d_b[n, inventory + Q]
    ask_depths = opt_d_a[n, inventory + Q]
    
    # Ensure types are float
    return bid_depths.astype(float), ask_depths.astype(float)

import numpy as np
import time

# Module-level cache for policy tables
_policy_cache = {}
_start_time = time.time()
_TIME_LIMIT = 320  # safety margin before 330s kill

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via backward induction on the discrete-time model.
    Uses vectorized Newton-Raphson for 1D maximization of bid and ask depths.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1)
        ask_table: array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_range = np.arange(-Q, Q + 1)
    
    # Value function
    V_next = np.zeros(n_inv)
    
    # Policy tables
    bid_table = np.full((steps, n_inv), np.inf)
    ask_table = np.full((steps, n_inv), np.inf)
    
    # Pre-compute constants
    # We will solve max_d p(d)(d + delta)
    # p(d) = 1 - exp(-A exp(-kd) dt)
    
    # Initial guess for depths: 
    # Asymptotically, if delta is large positive, we want large depth? 
    # Actually, p(d) -> 1 as d -> inf. J(d) ~ d + delta. 
    # But we are bounded by the fact that we might not want to quote if delta is negative enough?
    # If delta < 0, J(0) = p(0) * delta. If delta is very negative, J(0) is negative.
    # J(inf) = inf. So we always want to quote? 
    # Wait, if we quote at d=inf, p=0, J=0.
    # If we quote at d=0, p=max, J=p*delta.
    # If delta < 0, J(0) < 0. J(inf) = 0. So max is 0 (no quote).
    # So if delta <= 0, optimal depth is inf (no quote)? 
    # Let's check J(d) for delta < 0.
    # J(d) = (1 - e^{-u})(d+\Delta). 
    # If d is large, u is small, 1-e^{-u} ~ u. J(d) ~ u(d+\Delta). 
    # u = A dt e^{-kd}. 
    # J(d) ~ A dt e^{-kd} (d+\Delta). 
    # As d -> inf, e^{-kd} dominates d, so J(d) -> 0 from negative side? 
    # If d+\Delta < 0, and 1-e^{-u} > 0, then J(d) < 0.
    # So if delta < 0, the max is indeed 0 (by choosing d=inf).
    # If delta > 0, there is an interior maximum.
    
    # We'll use a heuristic initial guess.
    # For delta > 0, the optimal d is roughly where the marginal gain equals marginal cost.
    # A simple guess: d_0 = 1/k * log(A*dt / max(delta, 1e-6)) ?
    # Let's just start with a small positive value and let Newton converge, or use a grid fallback if Newton fails.
    
    # For vectorization, we process all q at once for a given n.
    
    for n in range(steps - 1, -1, -1):
        if time.time() - _start_time > _TIME_LIMIT:
            # Fill remaining with inf (no quote) to be safe
            bid_table[n+1:, :] = np.inf
            ask_table[n+1:, :] = np.inf
            break
            
        # Compute deltas for all q
        # delta_b[q] = V_next[q+1] - V_next[q]
        # delta_a[q] = V_next[q-1] - V_next[q]
        
        delta_b = np.zeros(n_inv)
        delta_a = np.zeros(n_inv)
        
        # q from -Q to Q-1 can have bid (q+1 exists)
        # q from -Q+1 to Q can have ask (q-1 exists)
        
        # Indices:
        # q = -Q -> idx 0. q+1 -> idx 1.
        # q = Q -> idx 2Q. q-1 -> idx 2Q-1.
        
        # delta_b for q in [-Q, Q-1] (idx 0 to 2Q-1)
        delta_b[:-1] = V_next[1:] - V_next[:-1]
        # delta_b for q=Q (idx 2Q) is not used (bid disabled)
        
        # delta_a for q in [-Q+1, Q] (idx 1 to 2Q)
        delta_a[1:] = V_next[:-1] - V_next[1:]
        # delta_a for q=-Q (idx 0) is not used (ask disabled)
        
        # Initialize depths
        # We want to find d that maximizes J(d) = p(d)(d+delta)
        # If delta <= 0, optimal d is inf (no quote).
        # If delta > 0, optimal d is finite.
        
        # Let's assume we start with a small depth for active sides.
        d_b = np.full(n_inv, 0.1)
        d_a = np.full(n_inv, 0.1)
        
        # Mark inactive sides
        d_b[-1] = np.inf  # q=Q, no bid
        d_a[0] = np.inf   # q=-Q, no ask
        
        # Newton Iterations
        # We only update finite depths
        active_b = np.isfinite(d_b)
        active_a = np.isfinite(d_a)
        
        max_iter = 20
        tol = 1e-8
        
        for _ in range(max_iter):
            # Update d_b
            if np.any(active_b):
                db = d_b[active_b]
                delt_b = delta_b[active_b]
                
                # Calculate u, P, J', J''
                u = A * np.exp(-k * db) * dt
                exp_u = np.exp(-u)
                P = 1.0 - exp_u
                
                # J'(d) = -k u exp(-u) (d+delta) + (1 - exp(-u))
                J_prime = -k * u * exp_u * (db + delt_b) + P
                
                # J''(d) = k u exp(-u) [ -(d+delta) + k(1-u) - 1 ]
                J_double_prime = k * u * exp_u * (-(db + delt_b) + k * (1 - u) - 1)
                
                # Newton step: d_new = d - J' / J''
                # Ensure J'' is not zero and step is safe
                # If J'' is positive (convex), we are at a minimum or edge. 
                # For this function, it should be concave (J'' < 0) near the max.
                # If J'' > 0, we might be far from the max or in a weird region.
                # Let's clamp the step size to prevent divergence.
                
                valid_step = np.abs(J_double_prime) > 1e-12
                step = np.zeros_like(db)
                step[valid_step] = J_prime[valid_step] / J_double_prime[valid_step]
                
                # Limit step size to 1.0 to ensure stability
                step = np.clip(step, -1.0, 1.0)
                
                new_db = db - step
                # Ensure d >= 0
                new_db = np.maximum(new_db, 0.0)
                
                # Check convergence
                converged = np.abs(step) < tol
                d_b[active_b] = np.where(converged, db, new_db)
                
            # Update d_a
            if np.any(active_a):
                da = d_a[active_a]
                delt_a = delta_a[active_a]
                
                u = A * np.exp(-k * da) * dt
                exp_u = np.exp(-u)
                P = 1.0 - exp_u
                
                J_prime = -k * u * exp_u * (da + delt_a) + P
                J_double_prime = k * u * exp_u * (-(da + delt_a) + k * (1 - u) - 1)
                
                valid_step = np.abs(J_double_prime) > 1e-12
                step = np.zeros_like(da)
                step[valid_step] = J_prime[valid_step] / J_double_prime[valid_step]
                
                step = np.clip(step, -1.0, 1.0)
                
                new_da = da - step
                new_da = np.maximum(new_da, 0.0)
                
                converged = np.abs(step) < tol
                d_a[active_a] = np.where(converged, da, new_da)
                
            # Early exit if all converged
            # Simple check: if max step was small
            # We can just run all iterations, it's vectorized and fast.
            
        # Calculate final values for V_curr
        # V_curr[q] = -cost + V_next[q] + P_b(d_b)(d_b + delta_b) + P_a(d_a)(d_a + delta_a)
        # Note: For inactive sides, contribution is 0.
        
        cost = 0.5 * gamma * sigma**2 * inv_range**2 * dt
        
        # Calculate contributions
        # P(d) = 1 - exp(-A exp(-kd) dt)
        
        # For bid
        contrib_b = np.zeros(n_inv)
        if np.any(active_b):
            db = d_b[active_b]
            u_b = A * np.exp(-k * db) * dt
            P_b = 1.0 - np.exp(-u_b)
            contrib_b[active_b] = P_b * (db + delta_b[active_b])
            
        # For ask
        contrib_a = np.zeros(n_inv)
        if np.any(active_a):
            da = d_a[active_a]
            u_a = A * np.exp(-k * da) * dt
            P_a = 1.0 - np.exp(-u_a)
            contrib_a[active_a] = P_a * (da + delta_a[active_a])
            
        V_curr = -cost + V_next + contrib_b + contrib_a
        
        bid_table[n, :] = d_b
        ask_table[n, :] = d_a
        
        V_next = V_curr
        
    return bid_table, ask_table

def _get_policy_table(params):
    """Get or compute the policy table for given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        sigma, A, k, gamma = params['sigma'], params['A'], params['k'], params['gamma']
        T, steps, Q = params['T'], params['steps'], params['Q']
        
        bid_table, ask_table = _compute_policy_table(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[key] = (bid_table, ask_table)
    
    return _policy_cache[key]

def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    
    Args:
        t: current time (scalar)
        inventory: array of shape (4000,) with inventory levels
        mid: array of shape (4000,) with mid prices
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths): tuple of two arrays of shape (4000,)
    """
    # Determine current step
    dt = params['T'] / params['steps']
    step = int(round(t / dt))
    step = min(max(step, 0), params['steps'] - 1)
    
    Q = params['Q']
    
    # Get policy table
    bid_table, ask_table = _get_policy_table(params)
    
    # Map inventory to index
    inv_indices = inventory + Q
    inv_indices = np.clip(inv_indices, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[step, inv_indices].astype(float)
    ask_depths = ask_table[step, inv_indices].astype(float)
    
    # Ensure valid values: must be > -3/k, finite, non-negative
    min_valid = -3.0 / params['k']
    bid_depths = np.where((bid_depths > min_valid) & np.isfinite(bid_depths) & (bid_depths >= 0), bid_depths, np.inf)
    ask_depths = np.where((ask_depths > min_valid) & np.isfinite(ask_depths) & (ask_depths >= 0), ask_depths, np.inf)
    
    return bid_depths, ask_depths

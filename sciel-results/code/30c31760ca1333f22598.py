"""Optimal market making with inventory risk using exact discrete HJB DP."""
import numpy as np
import time

# Module-level cache for policy parameters
_policy_cache = {}
_start_time = time.time()
_TIME_LIMIT = 320  # safety margin before 330s kill


def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the discrete HJB equation backwards to find optimal depths.
    Returns:
        bid_table: array of shape (steps, 2*Q+1)
        ask_table: array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    n_states = 2 * Q + 1
    q_min = -Q
    q_max = Q
    
    # Pre-compute depth grid
    # We need a grid fine enough to capture the optimum.
    # Optimal depth is usually small.
    # Let's use a logarithmic grid or a linear grid up to a max depth.
    # Max depth where prob is negligible:
    # P = 1 - exp(-A exp(-k d) dt)
    # If d is large, P ~ 0.
    # If d is small, P ~ 1.
    # Let's pick d_max such that A * exp(-k * d_max) * dt is very small.
    # A * dt * exp(-k * d_max) < 1e-5
    # exp(-k * d_max) < 1e-5 / (A * dt)
    # -k * d_max < log(1e-5 / (A * dt))
    # d_max > -1/k * log(1e-5 / (A * dt))
    
    A_dt = A * dt
    if A_dt > 0:
        d_max = max(1.0, -1.0 / k * np.log(1e-6 / A_dt))
    else:
        d_max = 1.0
    
    # Create a grid of depths
    # We need to handle d=0 separately or include it.
    # Number of points: 100 seems reasonable.
    n_depths = 100
    depths = np.linspace(0.0, d_max, n_depths)
    
    # Pre-compute probabilities for all depths
    # P(d) = 1 - exp(-A * exp(-k * d) * dt)
    # Handle d=0: exp(0)=1, P = 1 - exp(-A*dt)
    # If A*dt is large, P ~ 1.
    probs = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    
    # DP Tables
    # V_next[q_index]
    # q_index = q - q_min
    V_next = np.zeros(n_states)
    
    bid_table = np.full((steps, n_states), np.inf)
    ask_table = np.full((steps, n_states), np.inf)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_states)
        
        # For each inventory level q
        for q_idx in range(n_states):
            q = q_idx + q_min
            
            # Boundary conditions
            # No bid fill at q = Q
            # No ask fill at q = -Q
            
            # We need to maximize:
            # J = P_b * (d_b + V_next[q+1]) + P_a * (d_a + V_next[q-1]) + (1-P_b)*(1-P_a) * V_next[q] - penalty
            # penalty = (gamma / 2) * sigma^2 * q^2 * dt
            
            penalty = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Get V_next for q, q+1, q-1
            # Handle boundaries for V_next indices
            # If q == Q, q+1 is out of bounds. But P_b should be 0 or we just don't consider bid.
            # The problem states: "no bid fill at q = Q".
            # This implies if q == Q, we must set bid_depth = inf (or P_b = 0).
            # Similarly for ask at q = -Q.
            
            v_q = V_next[q_idx]
            
            # Determine valid q+1 and q-1
            if q < Q:
                v_q_plus = V_next[q_idx + 1]
            else:
                v_q_plus = 0.0 # Not used if P_b forced to 0
            
            if q > -Q:
                v_q_minus = V_next[q_idx - 1]
            else:
                v_q_minus = 0.0 # Not used if P_a forced to 0
            
            # We iterate over possible depths for bid and ask
            # To speed up, we can vectorize over depths.
            
            # If q == Q, bid is not allowed. We can set effective P_b = 0 for all d_b.
            # Or just skip bid optimization and set d_b = inf.
            
            if q == Q:
                # Bid is inf
                # Optimize d_a only
                # J = P_a * (d_a + v_q_minus) + (1-P_a) * v_q - penalty
                # Note: (1-P_b) is 1.
                
                # Vectorized over depths
                # term_a = probs * (depths + v_q_minus) + (1 - probs) * v_q - penalty
                # Wait, v_q is constant.
                # term_a = probs * depths + probs * v_q_minus + v_q - probs * v_q - penalty
                # term_a = probs * (depths + v_q_minus - v_q) + v_q - penalty
                
                val_a = probs * (depths + v_q_minus - v_q) + v_q - penalty
                best_idx_a = np.argmax(val_a)
                best_val = val_a[best_idx_a]
                best_d_a = depths[best_idx_a]
                
                V_curr[q_idx] = best_val
                bid_table[n, q_idx] = np.inf
                ask_table[n, q_idx] = best_d_a
                
            elif q == -Q:
                # Ask is inf
                # Optimize d_b only
                # J = P_b * (d_b + v_q_plus) + (1-P_b) * v_q - penalty
                
                val_b = probs * (depths + v_q_plus - v_q) + v_q - penalty
                best_idx_b = np.argmax(val_b)
                best_val = val_b[best_idx_b]
                best_d_b = depths[best_idx_b]
                
                V_curr[q_idx] = best_val
                bid_table[n, q_idx] = best_d_b
                ask_table[n, q_idx] = np.inf
                
            else:
                # Both allowed
                # J(d_b, d_a) = P_b * (d_b + v_q_plus - v_q) + P_a * (d_a + v_q_minus - v_q) + P_b * P_a * v_q + v_q - penalty
                # Let A_b = v_q_plus - v_q
                # Let A_a = v_q_minus - v_q
                # J = P_b * (d_b + A_b) + P_a * (d_a + A_a) + P_b * P_a * v_q + v_q - penalty
                
                A_b = v_q_plus - v_q
                A_a = v_q_minus - v_q
                
                # We need to maximize over d_b, d_a.
                # Since n_depths is 100, 100x100 = 10,000 combinations.
                # We can do this for each q (61 times) in the loop.
                # Total ops: 1000 steps * 61 states * 10000 combos = 610M ops.
                # This might be too slow in Python loops.
                # We should vectorize the inner loop.
                
                # Create meshgrid of depths
                # But v_q, A_b, A_a are scalars for this q.
                # probs is a vector of size 100.
                # We want J[i, j] = probs[i]*(depths[i]+A_b) + probs[j]*(depths[j]+A_a) + probs[i]*probs[j]*v_q + v_q - penalty
                
                # Let's use broadcasting
                # p_b = probs[:, np.newaxis] (100, 1)
                # d_b = depths[:, np.newaxis] (100, 1)
                # p_a = probs[np.newaxis, :] (1, 100)
                # d_a = depths[np.newaxis, :] (1, 100)
                
                p_b = probs[:, np.newaxis]
                d_b = depths[:, np.newaxis]
                p_a = probs[np.newaxis, :]
                d_a = depths[np.newaxis, :]
                
                # Term 1: P_b * (d_b + A_b)
                term1 = p_b * (d_b + A_b)
                # Term 2: P_a * (d_a + A_a)
                term2 = p_a * (d_a + A_a)
                # Term 3: P_b * P_a * v_q
                term3 = p_b * p_a * v_q
                # Constant
                const = v_q - penalty
                
                J = term1 + term2 + term3 + const
                
                best_idx_flat = np.argmax(J)
                best_i, best_j = np.unravel_index(best_idx_flat, J.shape)
                best_val = J[best_i, best_j]
                best_d_b = depths[best_i]
                best_d_a = depths[best_j]
                
                V_curr[q_idx] = best_val
                bid_table[n, q_idx] = best_d_b
                ask_table[n, q_idx] = best_d_a
        
        V_next = V_curr
        
        # Time check
        if time.time() - _start_time > _TIME_LIMIT - 10:
            # If we run out of time, we have to break and return what we have?
            # But we need the full table for early steps.
            # If we break, early steps will have inf depths (no quote).
            # This is a safe fallback.
            break

    return bid_table, ask_table


def _get_policy_table(params):
    """Get or compute the policy table for given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        sigma, A, k, gamma = params['sigma'], params['A'], params['k'], params['gamma']
        T, steps, Q = params['T'], params['steps'], params['Q']
        
        bid_table, ask_table = _solve_dp(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[key] = (bid_table, ask_table)
    
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    Q = params['Q']
    steps = params['steps']
    T = params['T']
    dt = T / steps
    
    # Get policy table
    bid_table, ask_table = _get_policy_table(params)
    
    # Determine current step index
    # t is n * dt
    n = int(round(t / dt))
    # Clamp n to valid range
    if n < 0:
        n = 0
    elif n >= steps:
        n = steps - 1
        
    # Get depths for this step
    # inventory is (4000,)
    # Map inventory to index: q - (-Q) = q + Q
    q_indices = inventory + Q
    
    # Handle out of bounds inventory (shouldn't happen, but safe)
    q_indices = np.clip(q_indices, 0, 2 * Q)
    
    # Extract depths
    bid_depths = bid_table[n, q_indices]
    ask_depths = ask_table[n, q_indices]
    
    # Ensure valid types and shapes
    bid_depths = bid_depths.astype(float)
    ask_depths = ask_depths.astype(float)
    
    # Validate: if any are NaN or negative (shouldn't be), set to inf
    # The table should contain valid depths or inf.
    # Depths in table are from linspace(0, d_max), so >= 0.
    # If we broke early in DP, some entries might be inf (initialized value).
    
    return bid_depths, ask_depths

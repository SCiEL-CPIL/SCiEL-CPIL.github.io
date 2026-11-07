# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for optimal policies
_policy_cache = {}
_global_start_time = None

def _get_deadline():
    """Return the global deadline (330s wall-clock limit)."""
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    return _global_start_time + 330

def _solve_1d_optimum(lam, k, coeff, deadline):
    """
    Solve max P(d) * (coeff + d) where P(d) = 1 - exp(-lam * exp(-k*d)).
    Returns optimal d in [0, 3/k].
    """
    if coeff <= 0:
        # If coeff <= 0, we want to minimize fill probability -> d -> inf, but capped
        # Actually, if coeff is negative, we don't want to fill. But d is depth, larger d = smaller prob.
        # Wait, P(d) is increasing in d? No.
        # P(d) = 1 - exp(-lam * exp(-k*d)).
        # As d increases, exp(-k*d) decreases, so lam*exp(-k*d) decreases, so exp(-...) increases, so 1-... decreases.
        # So P(d) is DECREASING in d.
        # We want to maximize P(d) * (coeff + d).
        # If coeff is very negative, we want P(d) to be small, so d should be large.
        # If coeff is positive, we want a balance.
        
        # Let's just use a grid search to be safe.
        pass

    max_depth = 3.0 / k
    # Use a fine grid
    grid = np.linspace(0, max_depth, 300)
    
    P_grid = 1.0 - np.exp(-lam * np.exp(-k * grid))
    vals = P_grid * (coeff + grid)
    
    # Check deadline
    if time.time() > deadline:
        return grid[np.argmax(vals)]
        
    idx = np.argmax(vals)
    best_d = grid[idx]
    best_val = vals[idx]
    
    # Refine with golden section search around the best grid point
    if idx > 0 and idx < len(grid) - 1:
        lo = grid[idx - 1]
        hi = grid[idx + 1]
        gr = (np.sqrt(5) - 1) / 2  # Golden ratio conjugate
        
        for _ in range(30):
            if time.time() > deadline:
                break
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            f1 = (1.0 - np.exp(-lam * np.exp(-k * x1))) * (coeff + x1)
            f2 = (1.0 - np.exp(-lam * np.exp(-k * x2))) * (coeff + x2)
            if f1 < f2:
                lo = x1
            else:
                hi = x2
        
        best_d = (lo + hi) / 2
        best_val = (1.0 - np.exp(-lam * np.exp(-k * best_d))) * (coeff + best_d)
        
        # Ensure it's better than grid result
        if best_val < best_val:  # This is always false, just logic check
             pass
             
    return best_d

def _compute_optimal_policy(params, deadline):
    """
    Compute the optimal bid and ask depths for all (t, q) states using DP.
    Returns:
        bid_depths: array of shape (steps, 2*Q+1)
        ask_depths: array of shape (steps, 2*Q+1)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    lam = A * dt
    num_q = 2 * Q + 1

    V = np.zeros((steps + 1, num_q), dtype=np.float64)
    bid_depths = np.full((steps, num_q), np.inf, dtype=np.float64)
    ask_depths = np.full((steps, num_q), np.inf, dtype=np.float64)

    cost = (gamma / 2.0) * sigma**2 * dt

    for t in range(steps - 1, -1, -1):
        if time.time() > deadline - 10:
            # Fallback for remaining steps: use simple heuristic
            for q in range(-Q, Q + 1):
                bid_depths[t, q + Q] = 1.0 / k
                ask_depths[t, q + Q] = 1.0 / k
                V[t, q + Q] = V[t+1, q+Q] - cost
            continue

        V_next = V[t + 1]
        
        # Vectorized optimization for all q at step t
        # We need to optimize b and a for each q.
        # Due to the coupling (P_b affects a's optimization and vice versa), 
        # and the fact that the coupling is weak for small lam, 
        # we can approximate by decoupling:
        # Optimize b assuming a is optimal given b? 
        # Or just iterate a few times.
        
        # Let's try a vectorized alternating approach.
        # Initialize b, a with 1/k
        b_vals = np.full(num_q, 1.0 / k)
        a_vals = np.full(num_q, 1.0 / k)
        
        # Boundaries
        # q = Q -> b = inf
        # q = -Q -> a = inf
        # q_idx = q + Q
        # q = Q -> q_idx = 2Q
        # q = -Q -> q_idx = 0
        
        for _ in range(5):
            if time.time() > deadline - 10:
                break
                
            # Optimize b given a
            # For q < Q (q_idx < 2Q):
            # coeff_b(q) = (1-P_a)(V_next[q+1] - V_next[q]) + P_a*a + V_next[q] - (P_a*(a + V_next[q-1]) + (1-P_a)*V_next[q])
            # Wait, let's re-derive carefully.
            # Obj = P_b * (X + b) + (1-P_b) * Y
            # Maximize over b: P_b(b) * (X - Y + b)
            # Let C_b = X - Y
            # X = (1-P_a)V_next[q+1] + P_a(a + V_next[q])
            # Y = P_a(a + V_next[q-1]) + (1-P_a)V_next[q]
            
            # We need P_a for each q.
            P_a = 1.0 - np.exp(-lam * np.exp(-k * a_vals))
            
            # V_next is shape (num_q,)
            # q_idx ranges 0..2Q
            # q = q_idx - Q
            
            # For q < Q (q_idx < 2Q):
            # V_q_plus = V_next[q_idx+1]
            # V_q_minus = V_next[q_idx-1]
            
            # Handle boundaries in vectorization:
            # For q_idx = 0 (q=-Q), ask not allowed, so a should be inf, P_a=0.
            # For q_idx = 2Q (q=Q), bid not allowed, so b should be inf.
            
            # Let's compute C_b for all q_idx where bid is allowed (q_idx < 2Q)
            # q_idx < 2Q means q < Q.
            
            # V_q_plus: for q_idx < 2Q, V_next[q_idx+1] is valid.
            # V_q_minus: for q_idx > 0, V_next[q_idx-1] is valid.
            
            # Let's create arrays for X and Y.
            # X = (1-P_a)*V_next[q_idx+1] + P_a*(a_vals + V_next[q_idx])
            # Y = P_a*(a_vals + V_next[q_idx-1]) + (1-P_a)*V_next[q_idx]
            
            # But we need to be careful with boundaries.
            # If q_idx == 0 (q=-Q), a_vals should be inf, so P_a = 0.
            # If q_idx == 2Q (q=Q), b_vals should be inf.
            
            # Let's enforce boundaries.
            a_vals[0] = np.inf
            b_vals[2*Q] = np.inf
            
            # Recompute P_a with boundaries
            P_a = np.zeros(num_q)
            valid_a = ~np.isinf(a_vals)
            P_a[valid_a] = 1.0 - np.exp(-lam * np.exp(-k * a_vals[valid_a]))
            
            # Compute X and Y for all q_idx
            # V_q_plus = V_next[q_idx+1]
            # For q_idx = 2Q, this would be out of bounds, but we don't need X for q_idx=2Q (b=inf)
            # Let's compute for q_idx < 2Q
            
            V_q = V_next
            V_q_plus = np.roll(V_next, -1)
            V_q_minus = np.roll(V_next, 1)
            
            # Fix boundaries for roll
            V_q_plus[-1] = 0 # Will be masked
            V_q_minus[0] = 0 # Will be masked
            
            X = (1.0 - P_a) * V_q_plus + P_a * (a_vals + V_q)
            Y = P_a * (a_vals + V_q_minus) + (1.0 - P_a) * V_q
            
            # For q_idx = 0, a=inf, P_a=0.
            # X = V_q_plus[0] + 0 = V_next[1]
            # Y = 0 + V_q[0] = V_next[0]
            # C_b = X - Y = V_next[1] - V_next[0]
            # This is correct. If q=-Q, bid fill moves to -Q+1.
            
            C_b = X - Y
            
            # Optimize b for q_idx < 2Q
            # We need to solve max P(d) * (C_b + d) for each q_idx < 2Q
            # We can do this in a loop for now, or vectorize the grid search.
            
            # Vectorized grid search for b
            max_depth = 3.0 / k
            grid = np.linspace(0, max_depth, 300)
            
            # P_grid is same for all q_idx
            P_grid = 1.0 - np.exp(-lam * np.exp(-k * grid))
            
            # We want argmax over grid of P_grid * (C_b[q_idx] + grid)
            # This is equivalent to argmax of P_grid * grid + P_grid * C_b[q_idx]
            # It's not separable easily because C_b is added inside the product.
            # P_grid * (C_b + grid) = P_grid * grid + P_grid * C_b
            
            # Let's just do a loop over q_idx for now, it's only 61 iterations.
            for qi in range(2*Q):
                if time.time() > deadline - 10:
                    break
                coeff = C_b[qi]
                # Use _solve_1d_optimum
                b_opt = _solve_1d_optimum(lam, k, coeff, deadline)
                b_vals[qi] = b_opt
            
            # Optimize a given b
            P_b = np.zeros(num_q)
            valid_b = ~np.isinf(b_vals)
            P_b[valid_b] = 1.0 - np.exp(-lam * np.exp(-k * b_vals[valid_b]))
            
            # For a, we maximize P_a(a) * (C_a + a)
            # C_a = U - V_term
            # U = a + C1 -> No, U depends on a.
            # Let's re-derive C_a.
            # Obj in a: P_a * (a + C1 - V_term) + V_term
            # C1 = P_b(b + V_q) + (1-P_b)V_q_minus
            # V_term = P_b(b + V_q_plus) + (1-P_b)V_q
            # So coeff_a = C1 - V_term
            # C1 - V_term = P_b(b + V_q) + (1-P_b)V_q_minus - P_b(b + V_q_plus) - (1-P_b)V_q
            #             = P_b(V_q - V_q_plus) + (1-P_b)(V_q_minus - V_q)
            
            C_a = P_b * (V_q - V_q_plus) + (1.0 - P_b) * (V_q_minus - V_q)
            
            for qi in range(1, num_q): # q_idx > 0 (q > -Q)
                if time.time() > deadline - 10:
                    break
                coeff = C_a[qi]
                a_opt = _solve_1d_optimum(lam, k, coeff, deadline)
                a_vals[qi] = a_opt
            
            # Enforce boundaries again
            a_vals[0] = np.inf
            b_vals[2*Q] = np.inf

        # Store results
        bid_depths[t, :] = b_vals
        ask_depths[t, :] = a_vals
        
        # Compute V[t, :]
        # V[t, q] = Expected Value - cost
        # We need to compute expected value for each q.
        
        P_b = np.zeros(num_q)
        valid_b = ~np.isinf(b_vals)
        P_b[valid_b] = 1.0 - np.exp(-lam * np.exp(-k * b_vals[valid_b]))
        
        P_a = np.zeros(num_q)
        valid_a = ~np.isinf(a_vals)
        P_a[valid_a] = 1.0 - np.exp(-lam * np.exp(-k * a_vals[valid_a]))
        
        # EV = P_b P_a (b+a+V_q) + P_b(1-P_a)(b+V_q_plus) + (1-P_b)P_a(a+V_q_minus) + (1-P_b)(1-P_a)V_q
        # Let's compute term by term.
        
        # b_vals and a_vals may be inf.
        # If b=inf, P_b=0. So terms with P_b are 0.
        # If a=inf, P_a=0. So terms with P_a are 0.
        
        # Term 1: P_b P_a (b+a+V_q)
        # If b or a is inf, P_b or P_a is 0, so product is 0.
        # We can compute safely if we replace inf with 0 in the sum part?
        # No, if P_b=0, term is 0.
        
        # Let's use np.where to handle infs.
        b_safe = np.where(np.isinf(b_vals), 0, b_vals)
        a_safe = np.where(np.isinf(a_vals), 0, a_vals)
        
        term1 = P_b * P_a * (b_safe + a_safe + V_q)
        term2 = P_b * (1.0 - P_a) * (b_safe + V_q_plus)
        term3 = (1.0 - P_b) * P_a * (a_safe + V_q_minus)
        term4 = (1.0 - P_b) * (1.0 - P_a) * V_q
        
        # Fix boundaries for V_q_plus and V_q_minus
        # For q_idx=0, V_q_minus is invalid (rolled). But P_a[0]=0, so term3=0.
        # For q_idx=2Q, V_q_plus is invalid. But P_b[2Q]=0, so term2=0.
        
        V[t, :] = term1 + term2 + term3 + term4 - cost

    return bid_depths, ask_depths

def quote(t, inventory, mid, params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    deadline = _get_deadline()
    
    if key not in _policy_cache:
        if time.time() < deadline - 20:
            try:
                bid_table, ask_table = _compute_optimal_policy(params, deadline)
                _policy_cache[key] = (bid_table, ask_table)
            except Exception:
                # Fallback
                depth = np.full(len(inventory), 1.0 / params['k'])
                return depth, depth.copy()
        else:
            # Fallback: risk-neutral depth
            depth = np.full(len(inventory), 1.0 / params['k'])
            return depth, depth.copy()
    
    bid_table, ask_table = _policy_cache[key]
    
    dt = params['T'] / params['steps']
    step = int(round(t / dt))
    step = np.clip(step, 0, params['steps'] - 1)
    
    Q = params['Q']
    q_indices = inventory + Q
    q_indices = np.clip(q_indices, 0, 2 * Q)
    
    bid_depths = bid_table[step, q_indices]
    ask_depths = ask_table[step, q_indices]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END

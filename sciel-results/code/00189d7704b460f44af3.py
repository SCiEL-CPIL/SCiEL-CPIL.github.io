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
    return _global_start_time + 330  # 330s wall-clock limit

def _solve_1d_optimum(lam, k, C, V_term, max_depth, num_grid=200):
    """
    Solve max_a P(a)*(a + C - V_term) + V_term where P(a) = 1 - exp(-lam * exp(-k*a))
    Returns optimal a.
    """
    # Grid search + golden section refinement
    grid = np.linspace(0, max_depth, num_grid)
    best_val = -np.inf
    best_a = 0.0
    
    # Vectorized grid evaluation
    P_vals = 1.0 - np.exp(-lam * np.exp(-k * grid))
    vals = P_vals * (grid + C - V_term) + V_term
    idx = np.argmax(vals)
    best_a = grid[idx]
    best_val = vals[idx]
    
    # Golden section refinement in neighborhood
    if 0 < idx < num_grid - 1:
        a_left = grid[idx - 1]
        a_right = grid[idx + 1]
        if a_right > a_left:
            # Golden section search
            gr = (np.sqrt(5) - 1) / 2
            a1 = a_right - gr * (a_right - a_left)
            a2 = a_left + gr * (a_right - a_left)
            f1 = (1.0 - np.exp(-lam * np.exp(-k * a1))) * (a1 + C - V_term) + V_term
            f2 = (1.0 - np.exp(-lam * np.exp(-k * a2))) * (a2 + C - V_term) + V_term
            
            for _ in range(20):
                if f1 < f2:
                    a_left = a1
                    a1 = a2
                    f1 = f2
                    a2 = a_left + gr * (a_right - a_left)
                    f2 = (1.0 - np.exp(-lam * np.exp(-k * a2))) * (a2 + C - V_term) + V_term
                else:
                    a_right = a2
                    a2 = a1
                    f2 = f1
                    a1 = a_right - gr * (a_right - a_left)
                    f1 = (1.0 - np.exp(-lam * np.exp(-k * a1))) * (a1 + C - V_term) + V_term
            
            a_opt = (a_left + a_right) / 2.0
            f_opt = (1.0 - np.exp(-lam * np.exp(-k * a_opt))) * (a_opt + C - V_term) + V_term
            if f_opt > best_val:
                best_a = a_opt
                best_val = f_opt
    
    return best_a

def _compute_optimal_policy(params):
    """
    Compute optimal bid and ask depths for all (t, q) states using DP with alternating optimization.
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
    
    # V[step, q_idx] - value function
    V = np.zeros((steps + 1, num_q), dtype=np.float64)
    
    # Depth tables
    bid_depths = np.full((steps, num_q), np.inf, dtype=np.float64)
    ask_depths = np.full((steps, num_q), np.inf, dtype=np.float64)
    
    # Max depth for search - use a reasonable upper bound
    max_depth = 6.0 / k
    
    deadline = _get_deadline()
    
    for t in range(steps - 1, -1, -1):
        # Check deadline periodically
        if t % 50 == 0 and time.time() > deadline - 5:
            # Fill remaining with heuristic
            for tt in range(t - 1, -1, -1):
                for q in range(-Q, Q + 1):
                    q_idx = q + Q
                    bid_depths[tt, q_idx] = 1.0 / k
                    ask_depths[tt, q_idx] = 1.0 / k
                    V[tt, q_idx] = 0.0
            return bid_depths, ask_depths
        
        V_next = V[t + 1]
        cost = (gamma / 2.0) * sigma**2 * dt
        
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            
            # Boundary conditions
            can_bid = q < Q
            can_ask = q > -Q
            
            if not can_bid and not can_ask:
                # At boundary, no fills possible
                bid_depths[t, q_idx] = np.inf
                ask_depths[t, q_idx] = np.inf
                V[t, q_idx] = V_next[q_idx] - cost
                continue
            
            if not can_bid:
                # Only ask possible
                ask_depths[t, q_idx] = np.inf
                # Optimize ask only
                Pa_grid = 1.0 - np.exp(-lam * np.exp(-k * np.linspace(0, max_depth, 200)))
                V_q_minus = V_next[q_idx - 1]
                V_q = V_next[q_idx]
                vals = Pa_grid * (np.linspace(0, max_depth, 200) + V_q_minus) + (1.0 - Pa_grid) * V_q
                idx = np.argmax(vals)
                a_opt = np.linspace(0, max_depth, 200)[idx]
                Pa_opt = 1.0 - np.exp(-lam * np.exp(-k * a_opt))
                ask_depths[t, q_idx] = a_opt
                V[t, q_idx] = Pa_opt * (a_opt + V_q_minus) + (1.0 - Pa_opt) * V_q - cost
                continue
            
            if not can_ask:
                # Only bid possible
                bid_depths[t, q_idx] = np.inf
                # Optimize bid only
                Pb_grid = 1.0 - np.exp(-lam * np.exp(-k * np.linspace(0, max_depth, 200)))
                V_q_plus = V_next[q_idx + 1]
                V_q = V_next[q_idx]
                vals = Pb_grid * (np.linspace(0, max_depth, 200) + V_q_plus) + (1.0 - Pb_grid) * V_q
                idx = np.argmax(vals)
                b_opt = np.linspace(0, max_depth, 200)[idx]
                Pb_opt = 1.0 - np.exp(-lam * np.exp(-k * b_opt))
                bid_depths[t, q_idx] = b_opt
                V[t, q_idx] = Pb_opt * (b_opt + V_q_plus) + (1.0 - Pb_opt) * V_q - cost
                continue
            
            # Both can bid and ask
            V_q = V_next[q_idx]
            V_q_plus = V_next[q_idx + 1]
            V_q_minus = V_next[q_idx - 1]
            
            # Initialize
            b = 1.0 / k
            a = 1.0 / k
            
            # Alternating optimization
            for _ in range(8):
                # Optimize b given a
                Pa = 1.0 - np.exp(-lam * np.exp(-k * a))
                # f(b) = P(b) * (b + C1 - V_term) + V_term
                # C1 = (1-Pa)*V_q_plus + Pa*(a + V_q)
                # V_term = Pa*(a + V_q_minus) + (1-Pa)*V_q
                C1 = (1.0 - Pa) * V_q_plus + Pa * (a + V_q)
                V_term_b = Pa * (a + V_q_minus) + (1.0 - Pa) * V_q
                b = _solve_1d_optimum(lam, k, C1, V_term_b, max_depth)
                
                # Optimize a given b
                Pb = 1.0 - np.exp(-lam * np.exp(-k * b))
                # g(a) = P(a) * (a + C2 - V_term2) + V_term2
                # C2 = Pb*(b + V_q) + (1-Pb)*V_q_minus
                # V_term2 = Pb*(b + V_q_plus) + (1-Pb)*V_q
                C2 = Pb * (b + V_q) + (1.0 - Pb) * V_q_minus
                V_term_a = Pb * (b + V_q_plus) + (1.0 - Pb) * V_q
                a = _solve_1d_optimum(lam, k, C2, V_term_a, max_depth)
            
            bid_depths[t, q_idx] = b
            ask_depths[t, q_idx] = a
            
            # Compute value
            Pb = 1.0 - np.exp(-lam * np.exp(-k * b))
            Pa = 1.0 - np.exp(-lam * np.exp(-k * a))
            V[t, q_idx] = (Pb * Pa * (b + a + V_q) + 
                           Pb * (1.0 - Pa) * (b + V_q_plus) + 
                           (1.0 - Pb) * Pa * (a + V_q_minus) + 
                           (1.0 - Pb) * (1.0 - Pa) * V_q) - cost
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for the current state.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        deadline = _get_deadline()
        if time.time() < deadline - 10:  # Safety margin
            bid_table, ask_table = _compute_optimal_policy(params)
            _policy_cache[key] = (bid_table, ask_table)
        else:
            # Fallback heuristic
            k = params['k']
            depth = np.full(len(inventory), 1.0 / k)
            return depth, depth.copy()
    
    bid_table, ask_table = _policy_cache[key]
    
    # Get current step
    dt = params['T'] / params['steps']
    step = int(round(t / dt))
    step = np.clip(step, 0, params['steps'] - 1)
    
    # Map inventory to index
    Q = params['Q']
    q_indices = inventory + Q
    q_indices = np.clip(q_indices, 0, 2 * Q)
    
    # Lookup depths
    bid_depths = bid_table[step, q_indices]
    ask_depths = ask_table[step, q_indices]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END

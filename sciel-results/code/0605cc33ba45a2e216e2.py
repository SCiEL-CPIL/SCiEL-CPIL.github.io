"""
Strategy: Pre-compute the exact optimal policy using Dynamic Programming (Bellman Equation).
The state space is small (1000 steps x 61 inventory levels), so we can solve for the
optimal bid/ask depths for each state. We cache the results per regime (sigma, A, k, gamma)
so that the first call for a regime pays the DP cost, and subsequent calls are O(1) lookups.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for pre-computed optimal policies
# Key: tuple of regime parameters
# Value: (bid_depths, ask_depths) arrays of shape (steps, 2*Q+1)
_policy_cache = {}

def _compute_optimal_policy(params):
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

    # Pre-compute lambda = A * dt (the rate parameter for fill probability)
    lam = A * dt

    # Number of inventory levels: -Q to Q inclusive -> 2*Q + 1
    num_q = 2 * Q + 1

    # DP tables: V[step, q_idx] where q_idx = q + Q
    # V[steps] is terminal value = 0 (no terminal charge)
    V = np.zeros((steps + 1, num_q), dtype=np.float64)

    # Depth arrays: shape (steps, num_q)
    bid_depths = np.full((steps, num_q), np.inf, dtype=np.float64)
    ask_depths = np.full((steps, num_q), np.inf, dtype=np.float64)

    # We'll use a grid search for depths. The optimal depth should be in a reasonable range.
    # For small lambda, optimal depth ~ 1/k. Let's search in [0, 5/k] with fine resolution.
    # But we need to be efficient. Let's use a grid of ~100 points for each dimension? 
    # That's 10,000 combos per state. 60,000 states * 10,000 = 600M. Too slow in pure Python.
    
    # Better approach: The objective for a state (t, q) is:
    # Maximize over b, a >= 0:
    #   - (gamma/2) * sigma^2 * q^2 * dt  (constant for this state)
    #   + P_b * (b + V[t+1, q+1])  (if q < Q)
    #   + P_a * (a + V[t+1, q-1])  (if q > -Q)
    #   - P_b * P_a * 0  (independent fills, but they both happen? No, independent.)
    #   Actually, the expected value is:
    #   P_b * (b) + P_b * V[t+1, q+1] * 1_{q<Q}  +  (1-P_b) * V[t+1, q]
    #   + P_a * (a) + P_a * V[t+1, q-1] * 1_{q>-Q}  +  (1-P_a) * V[t+1, q]
    #   But we can't just sum them because the inventory changes affect the future value.
    #   Let's be precise.
    #   Let V_next(q') = V[t+1, q']
    #   If no fill: q stays, value V_next(q)
    #   If bid fill: q -> q+1 (if q<Q), value b + V_next(q+1)
    #   If ask fill: q -> q-1 (if q>-Q), value a + V_next(q-1)
    #   Both fill: q -> q (bid +1, ask -1), value b + a + V_next(q)
    #   Probabilities:
    #   P_b = 1 - exp(-lam * exp(-k*b))
    #   P_a = 1 - exp(-lam * exp(-k*a))
    #   P_both = P_b * P_a
    #   P_bid_only = P_b * (1 - P_a)
    #   P_ask_only = P_a * (1 - P_b)
    #   P_none = (1 - P_b) * (1 - P_a)
    #   
    #   Expected Value = P_bid_only * (b + V_next(q+1)) + P_ask_only * (a + V_next(q-1)) 
    #                   + P_both * (b + a + V_next(q)) + P_none * V_next(q)
    #                   + constant cost
    #   
    #   This is complex to optimize jointly. Let's use alternating minimization.
    #   Fix a, optimize b. Then fix b, optimize a.
    #   Given a, P_a is fixed. 
    #   Objective in b:
    #   P_b * (1-P_a) * (b + V_next(q+1)) + P_b * P_a * (b + a + V_next(q)) 
    #   + (1-P_b) * [ P_a * (a + V_next(q-1)) + (1-P_a) * V_next(q) ]
    #   = P_b * [ (1-P_a)(b+V_next(q+1)) + P_a(b+a+V_next(q)) ]
    #   + (1-P_b) * [ P_a(a+V_next(q-1)) + (1-P_a)V_next(q) ]
    #   Let W = (1-P_a)(b+V_next(q+1)) + P_a(b+a+V_next(q))  -- Wait, b is in here.
    #   Let's re-group terms with P_b and (1-P_b).
    #   Let X = (1-P_a)(V_next(q+1)) + P_a(a + V_next(q))
    #   Let Y = P_a(a + V_next(q-1)) + (1-P_a)V_next(q)
    #   Objective = P_b * (X + b) + (1-P_b) * Y
    #              = P_b * (X + b - Y) + Y
    #   So we maximize f(b) = P_b(b) * (X + b - Y) + Y
    #   where P_b(b) = 1 - exp(-lam * exp(-k*b))
    #   
    #   This is a 1D optimization problem for b.
    #   Similarly for a.
    
    # Let's implement a simple grid search for the 1D optimization, refined by golden section.
    # Or just a fine grid since the function is smooth.
    # Range for b: [0, max_depth]. max_depth = 5/k seems safe.
    # Number of grid points: 50.
    
    max_depth = 5.0 / k
    grid_b = np.linspace(0, max_depth, 50)
    
    for t in range(steps - 1, -1, -1):
        V_next = V[t + 1]
        cost = (gamma / 2.0) * sigma**2 * dt
        
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            V_q = V_next[q_idx]
            
            # Handle boundary conditions
            if q < Q:
                V_q_plus = V_next[q_idx + 1]
            else:
                V_q_plus = -np.inf  # No bid fill allowed
                
            if q > -Q:
                V_q_minus = V_next[q_idx - 1]
            else:
                V_q_minus = -np.inf  # No ask fill allowed
            
            # We need to optimize b and a.
            # If both are allowed, we alternate.
            # Initialize b, a
            b = 1.0 / k
            a = 1.0 / k
            
            # Alternate optimization
            for _ in range(5):  # A few iterations should converge
                # Optimize b given a
                if q < Q:
                    P_a = 1.0 - np.exp(-lam * np.exp(-k * a))
                    # X = (1-P_a)*V_q_plus + P_a*(a + V_q)
                    # Y = P_a*(a + V_q_minus) + (1-P_a)*V_q  (if q > -Q, else handle)
                    if q > -Q:
                        Y = P_a * (a + V_q_minus) + (1.0 - P_a) * V_q
                    else:
                        # If q == -Q, ask fill not allowed, so P_a effectively 0 for this state?
                        # No, we are choosing a. If q == -Q, we should set a = inf (no ask quote).
                        # But here we are optimizing b given a. If q == -Q, a should be inf.
                        # Let's handle boundaries more carefully.
                        pass
                    
                    X = (1.0 - P_a) * V_q_plus + P_a * (a + V_q)
                    if q > -Q:
                        Y = P_a * (a + V_q_minus) + (1.0 - P_a) * V_q
                    else:
                        Y = V_q  # Only none or bid fill possible. But if q=-Q, ask not allowed, so P_a=0 effectively?
                        # Actually, if q=-Q, we shouldn't quote ask. So a=inf, P_a=0.
                        # Let's just set a=inf if q==-Q in the main loop.
                    
                    # Maximize f(b) = P_b(b) * (X + b - Y) + Y
                    # P_b(b) = 1 - exp(-lam * exp(-k*b))
                    # Grid search
                    best_b = 0
                    best_val = -np.inf
                    for bi in grid_b:
                        Pb = 1.0 - np.exp(-lam * np.exp(-k * bi))
                        val = Pb * (X + bi - Y) + Y
                        if val > best_val:
                            best_val = val
                            best_b = bi
                    b = best_b
                else:
                    b = np.inf  # No bid quote at q=Q
                
                # Optimize a given b
                if q > -Q:
                    P_b = 1.0 - np.exp(-lam * np.exp(-k * b))
                    # Symmetric logic
                    # Objective in a: P_a * (Y' + a - X') + X'
                    # Let's derive:
                    # Obj = P_b * (X + b) + (1-P_b) * Y  -- this was for fixed a.
                    # Now fixed b.
                    # Obj = P_b * P_a * (b + a + V_q) + P_b * (1-P_a) * (b + V_q_plus)
                    #       + (1-P_b) * P_a * (a + V_q_minus) + (1-P_b)(1-P_a) V_q
                    # Group by P_a and (1-P_a):
                    # P_a * [ P_b(b+a+V_q) + (1-P_b)(a+V_q_minus) ]
                    # (1-P_a) * [ P_b(b+V_q_plus) + (1-P_b)V_q ]
                    # Let U = P_b(b+a+V_q) + (1-P_b)(a+V_q_minus) = P_b*b + P_b*V_q + a(1-P_b) - a*P_b? No.
                    # U = a * [ P_b + 1 - P_b ] + P_b(b+V_q) + (1-P_b)V_q_minus
                    #   = a + P_b(b+V_q) + (1-P_b)V_q_minus
                    # Let V_term = P_b(b+V_q_plus) + (1-P_b)V_q
                    # Maximize g(a) = P_a(a) * (U) + (1-P_a(a)) * V_term
                    #            = P_a(a) * (U - V_term) + V_term
                    U = a + P_b * (b + V_q) + (1.0 - P_b) * V_q_minus  # Wait, a is the variable here.
                    # U depends on a. U(a) = a + C1, where C1 = P_b(b+V_q) + (1-P_b)V_q_minus
                    # So g(a) = P_a(a) * (a + C1 - V_term) + V_term
                    C1 = P_b * (b + V_q) + (1.0 - P_b) * V_q_minus
                    V_term = P_b * (b + V_q_plus) + (1.0 - P_b) * V_q
                    best_a = 0
                    best_val = -np.inf
                    for ai in grid_b:
                        Pa = 1.0 - np.exp(-lam * np.exp(-k * ai))
                        val = Pa * (ai + C1 - V_term) + V_term
                        if val > best_val:
                            best_val = val
                            best_a = ai
                    a = best_a
                else:
                    a = np.inf  # No ask quote at q=-Q
            
            # Store optimal depths
            bid_depths[t, q_idx] = b
            ask_depths[t, q_idx] = a
            
            # Compute V[t, q_idx]
            if np.isinf(b) and np.isinf(a):
                V[t, q_idx] = V_q - cost
            elif np.isinf(b):
                # Only ask possible
                Pa = 1.0 - np.exp(-lam * np.exp(-k * a))
                V[t, q_idx] = Pa * (a + V_q_minus) + (1.0 - Pa) * V_q - cost
            elif np.isinf(a):
                # Only bid possible
                Pb = 1.0 - np.exp(-lam * np.exp(-k * b))
                V[t, q_idx] = Pb * (b + V_q_plus) + (1.0 - Pb) * V_q - cost
            else:
                Pb = 1.0 - np.exp(-lam * np.exp(-k * b))
                Pa = 1.0 - np.exp(-lam * np.exp(-k * a))
                V[t, q_idx] = Pb * Pa * (b + a + V_q) + \
                              Pb * (1.0 - Pa) * (b + V_q_plus) + \
                              (1.0 - Pb) * Pa * (a + V_q_minus) + \
                              (1.0 - Pb) * (1.0 - Pa) * V_q - cost
        
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    # Create cache key
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        # Compute optimal policy
        deadline = time.time() + 300  # Safety margin
        if time.time() < deadline:
            bid_table, ask_table = _compute_optimal_policy(params)
            _policy_cache[key] = (bid_table, ask_table)
        else:
            # Fallback: risk-neutral depth
            depth = np.full(len(inventory), 1.0 / params['k'])
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

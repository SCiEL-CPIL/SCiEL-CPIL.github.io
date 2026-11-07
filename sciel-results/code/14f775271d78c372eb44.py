import numpy as np
import time
from scipy.optimize import minimize_scalar

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0
_POLICY_CACHE = {}

def _solve_optimal_depth(C, alpha, k):
    """
    Finds optimal depth x >= 0 maximizing (1 - exp(-alpha * exp(-k*x))) * (x + C)
    C is the value difference V(q+1) - V(q) for bid, or V(q-1) - V(q) for ask.
    """
    if alpha <= 0:
        return 0.0
    
    # If C is very large positive, we want to maximize fill probability to capture C?
    # Actually, if C > 0, (x+C) is large.
    # If C < 0, we might want to avoid fills?
    
    def neg_obj(x):
        if x < 0: return -1e10
        p = 1.0 - np.exp(-alpha * np.exp(-k * x))
        return -p * (x + C)
        
    # Initial guess
    # If C is large, x should be small to get high p? Or large to get high x?
    # If C is negative, we want p to be small (x large) or x small?
    # Let's use a bounded scalar minimization.
    # Upper bound: if x is too large, p->0, obj->0.
    # If x is 0, obj = (1-exp(-alpha)) * C.
    # Max x: 10/k is usually enough for p to be negligible.
    x_max = 10.0 / k
    
    res = minimize_scalar(neg_obj, bounds=(0, x_max), method='bounded')
    return res.x

def _compute_policy(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
        
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    alpha = A * dt
    
    # V is shape (steps+1, 2*Q+1)
    # Index q_idx = q + Q
    n_states = 2 * Q + 1
    V = np.zeros((steps + 1, n_states))
    
    bid_depths = np.full((steps, n_states), np.inf)
    ask_depths = np.full((steps, n_states), np.inf)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        t = n * dt
        # Check time
        if time.time() > _DEADLINE - 5.0:
            # Fallback to simple heuristic if running out of time
            # Just return what we have so far or a default
            for i in range(n, steps):
                bid_depths[i] = 1.0/k
                ask_depths[i] = 1.0/k
            _POLICY_CACHE[key] = (bid_depths, ask_depths)
            return _POLICY_CACHE[key]

        V_next = V[n+1]
        V_curr = V[n]
        
        for q_idx in range(n_states):
            q = q_idx - Q
            
            # Boundary conditions
            # If q == Q, cannot buy (bid fill increases q to Q+1 which is invalid)
            # So bid_depth should be inf (no bid)
            # If q == -Q, cannot sell (ask fill decreases q to -Q-1 which is invalid)
            # So ask_depth should be inf (no ask)
            
            C_plus = 0.0
            C_minus = 0.0
            
            if q < Q:
                # Bid side
                # q+1 is valid
                C_plus = V_next[q_idx + 1] - V_next[q_idx]
                b_opt = _solve_optimal_depth(C_plus, alpha, k)
                bid_depths[n, q_idx] = b_opt
            else:
                bid_depths[n, q_idx] = np.inf
                
            if q > -Q:
                # Ask side
                # q-1 is valid
                C_minus = V_next[q_idx - 1] - V_next[q_idx]
                a_opt = _solve_optimal_depth(C_minus, alpha, k)
                ask_depths[n, q_idx] = a_opt
            else:
                ask_depths[n, q_idx] = np.inf
                
            # Calculate V[n, q]
            # V = -cost + V_next[q] + max_term
            # max_term is already captured in the optimization?
            # No, we need to plug the optimal b and a back into the value equation.
            
            b = bid_depths[n, q_idx]
            a = ask_depths[n, q_idx]
            
            p_b = 0.0
            p_a = 0.0
            gain_b = 0.0
            gain_a = 0.0
            
            if np.isfinite(b):
                p_b = 1.0 - np.exp(-alpha * np.exp(-k * b))
                # Expected contribution from bid
                # p_b * (b + V_next[q+1] - V_next[q])
                # Note: The term (b + V_next[q+1]) is the reward + next value.
                # The baseline is V_next[q].
                # So the incremental value is p_b * (b + V_next[q+1]) + (1-p_b) * V_next[q] ?
                # No.
                # E[R + V_next] = p_b (b + V_next[q+1]) + p_a (a + V_next[q-1]) + (1-p_b-p_a) V_next[q]
                # = V_next[q] + p_b (b + V_next[q+1] - V_next[q]) + p_a (a + V_next[q-1] - V_next[q])
                # = V_next[q] + p_b (b + C_plus) + p_a (a + C_minus)
                
                gain_b = p_b * (b + C_plus)
            else:
                C_plus = 0.0 # Not used if no bid
                
            if np.isfinite(a):
                p_a = 1.0 - np.exp(-alpha * np.exp(-k * a))
                gain_a = p_a * (a + C_minus)
            else:
                C_minus = 0.0

            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            V_curr[q_idx] = -cost + V_next[q_idx] + gain_b + gain_a

    _POLICY_CACHE[key] = (bid_depths, ask_depths)
    return _POLICY_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Returns (bid_depths, ask_depths) arrays of shape (4000,)
    """
    # Get or compute policy
    bid_table, ask_table = _compute_policy(params)
    
    # Determine step index
    # t = n * dt. n = int(t / dt)
    # Ensure n is within [0, steps-1]
    steps = params["steps"]
    T = params["T"]
    dt = T / steps
    
    n = int(t / dt)
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    Q = params["Q"]
    
    # Map inventory to index
    # inventory is array of ints
    # idx = inventory + Q
    # Clip to valid range just in case, though problem says [-Q, Q]
    inv_idx = np.clip(inventory + Q, 0, 2 * Q)
    
    # Lookup
    bid_depths = bid_table[n, inv_idx]
    ask_depths = ask_table[n, inv_idx]
    
    # Ensure floats
    return bid_depths.astype(float), ask_depths.astype(float)

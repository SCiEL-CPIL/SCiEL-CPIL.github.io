import numpy as np
import time

_V_TABLES = {}
_T0 = time.time()
_DEADLINE = 290.0

def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    M = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Depth grid
    # Optimal d is around 1/k. 
    # Max d: if d is very large, p is tiny.
    # Let's use 0 to 3/k.
    N_d = 100
    d_grid = np.linspace(0.0, 3.0 / k, N_d)
    
    # Pre-compute p_b, p_a for all d in grid
    # p = 1 - exp(-A * exp(-k * d) * dt)
    exp_kd = np.exp(-k * d_grid)
    lambda_vals = A * exp_kd * dt
    p_grid = 1.0 - np.exp(-lambda_vals)
    
    # V_next: value at n+1
    V_next = np.zeros(M)
    
    # Tables to store optimal actions
    # We only need to store the depths for the quote() function
    # quote() is called with specific t, inventory.
    # We need to return the depth for that specific (n, q).
    # So we store bid_depths[n, q] and ask_depths[n, q].
    bid_depths = np.zeros((steps, M))
    ask_depths = np.zeros((steps, M))
    
    # Penalty term
    penalty = (gamma / 2.0) * (sigma ** 2) * (q_vals ** 2) * dt
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() - _T0 > _DEADLINE - 10.0:
             # Fallback to simple heuristic if timeout
             return None 
        
        # V_next is shape (M,)
        # V_minus: V(q-1)
        V_minus = np.zeros(M)
        V_minus[1:] = V_next[:-1]
        # V_plus: V(q+1)
        V_plus = np.zeros(M)
        V_plus[:-1] = V_next[1:]
        
        # Shapes:
        # d_b: (N_d, 1, 1)
        # d_a: (1, N_d, 1)
        # p_b: (N_d, 1, 1)
        # p_a: (1, N_d, 1)
        # q: (1, 1, M)
        
        d_b = d_grid[:, None, None]
        d_a = d_grid[None, :, None]
        p_b = p_grid[:, None, None]
        p_a = p_grid[None, :, None]
        
        # Apply boundary constraints
        # If q == Q (last index), no bid fill allowed -> p_b = 0
        # If q == -Q (first index), no ask fill allowed -> p_a = 0
        p_b_eff = p_b.copy()
        p_b_eff[:, :, -1] = 0.0
        
        p_a_eff = p_a.copy()
        p_a_eff[:, :, 0] = 0.0
        
        # Calculate probabilities
        p_both = p_a_eff * p_b_eff
        p_only_a = p_a_eff * (1.0 - p_b_eff)
        p_only_b = (1.0 - p_a_eff) * p_b_eff
        p_none = (1.0 - p_a_eff) * (1.0 - p_b_eff)
        
        # Expected Future Value
        # E[V] = p_none * V(q) + p_only_a * V(q-1) + p_only_b * V(q+1) + p_both * V(q)
        # Note: V(q) is V_next
        V_next_broad = V_next[None, None, :]
        V_minus_broad = V_minus[None, None, :]
        V_plus_broad = V_plus[None, None, :]
        
        E_V = p_none * V_next_broad + p_only_a * V_minus_broad + p_only_b * V_plus_broad + p_both * V_next_broad
        
        # Immediate Reward
        # R = d_a * p_a - d_b * p_b - penalty
        # penalty is (1, 1, M)
        penalty_broad = penalty[None, None, :]
        R = d_a * p_a_eff - d_b * p_b_eff - penalty_broad
        
        J = R + E_V
        
        # Max over depth indices
        # J is (N_d, N_d, M)
        # We want max over first two dims for each q
        max_idx = np.argmax(J, axis=(0, 1)) # shape (M,)
        V_next = J.reshape(-1, M)[max_idx, np.arange(M)]
        
        # Store optimal depths
        # max_idx is a flat index into (N_d * N_d)
        i_idx = max_idx // N_d
        j_idx = max_idx % N_d
        
        bid_depths[n, :] = d_grid[i_idx]
        ask_depths[n, :] = d_grid[j_idx]
        
    return {
        'bid_depths': bid_depths,
        'ask_depths': ask_depths
    }

def quote(t, inventory, mid, params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    
    if key not in _V_TABLES:
        if time.time() - _T0 > _DEADLINE - 20.0:
            # Fallback
            depth = np.full(len(inventory), 1.0 / params["k"])
            return depth, depth.copy()
            
        result = _solve_dp(params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
        if result is None:
            depth = np.full(len(inventory), 1.0 / params["k"])
            return depth, depth.copy()
        _V_TABLES[key] = result
        
    data = _V_TABLES[key]
    dt = params["T"] / params["steps"]
    n = int(round(t / dt))
    n = max(0, min(n, params["steps"] - 1))
    
    Q = params["Q"]
    # inventory is int array in [-Q, Q]
    # index = inventory + Q
    q_idx = inventory + Q
    
    bid = data['bid_depths'][n, q_idx]
    ask = data['ask_depths'][n, q_idx]
    
    return bid, ask

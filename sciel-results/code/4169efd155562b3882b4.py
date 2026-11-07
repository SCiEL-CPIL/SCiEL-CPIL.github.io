"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

_policy_cache = {}

def _get_policy_table(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _policy_cache:
        return _policy_cache[key]
        
    dt = T / steps
    N_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Depth grid
    d_max = 5.0 / k
    N_d = 50
    D = np.linspace(0, d_max, N_d)
    
    # Probabilities for each depth
    P = 1.0 - np.exp(-A * np.exp(-k * D) * dt)
    
    # V_next is shape (N_q,)
    V_next = np.zeros(N_q)
    
    # policy_table[n, q_idx, 0] = d_bid, [n, q_idx, 1] = d_ask
    policy_table = np.zeros((steps, N_q, 2))
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(N_q)
        
        # Interior states: q from -Q+1 to Q-1
        if N_q > 2:
            q_interior = q_vals[1:-1]
            q_idx_interior = np.arange(1, N_q - 1)
            
            V_minus = V_next[q_idx_interior - 1]
            V_same = V_next[q_idx_interior]
            V_plus = V_next[q_idx_interior + 1]
            
            penalty = 0.5 * gamma * sigma**2 * q_interior**2 * dt
            
            # fill_reward shape (N_d, N_d)
            fill_reward = P[:, np.newaxis] * D[:, np.newaxis] + P[np.newaxis, :] * D[np.newaxis, :]
            
            Delta_plus = V_plus - V_same
            Delta_minus = V_minus - V_same
            
            base_val = (V_same - penalty)[:, np.newaxis, np.newaxis]
            term_i = P[:, np.newaxis, np.newaxis] * Delta_plus[:, np.newaxis, np.newaxis]
            term_j = P[np.newaxis, :, np.newaxis] * Delta_minus[:, np.newaxis, np.newaxis]
            
            full_val = base_val + fill_reward[np.newaxis, :, :] + term_i + term_j
            
            max_vals = np.max(full_val, axis=(1, 2))
            max_flat_idxs = np.argmax(full_val, axis=(1, 2))
            
            i_idxs, j_idxs = np.unravel_index(max_flat_idxs, (N_d, N_d))
            
            V_curr[q_idx_interior] = max_vals
            policy_table[n, q_idx_interior, 0] = D[i_idxs]
            policy_table[n, q_idx_interior, 1] = D[j_idxs]
            
        # Boundary states
        # q = Q
        q_Q = Q
        penalty_Q = 0.5 * gamma * sigma**2 * q_Q**2 * dt
        V_same_Q = V_next[N_q - 1]
        V_minus_Q = V_next[N_q - 2]
        
        Delta_minus_Q = V_minus_Q - V_same_Q
        vals_Q = V_same_Q - penalty_Q + P * (D + Delta_minus_Q)
        best_j_Q = np.argmax(vals_Q)
        
        V_curr[N_q - 1] = vals_Q[best_j_Q]
        policy_table[n, N_q - 1, 0] = np.inf
        policy_table[n, N_q - 1, 1] = D[best_j_Q]
        
        # q = -Q
        q_mQ = -Q
        penalty_mQ = 0.5 * gamma * sigma**2 * q_mQ**2 * dt
        V_same_mQ = V_next[0]
        V_plus_mQ = V_next[1]
        
        Delta_plus_mQ = V_plus_mQ - V_same_mQ
        vals_mQ = V_same_mQ - penalty_mQ + P * (D + Delta_plus_mQ)
        best_i_mQ = np.argmax(vals_mQ)
        
        V_curr[0] = vals_mQ[best_i_mQ]
        policy_table[n, 0, 0] = D[best_i_mQ]
        policy_table[n, 0, 1] = np.inf
        
        V_next = V_curr
        
    _policy_cache[key] = policy_table
    return policy_table

def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    table = _get_policy_table(sigma, A, k, gamma, T, steps, Q)
    
    n = int(t / (T / steps))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    q = np.asarray(inventory, dtype=int)
    idx = q + Q
    
    bid_depths = table[n, idx, 0].astype(float)
    ask_depths = table[n, idx, 1].astype(float)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END

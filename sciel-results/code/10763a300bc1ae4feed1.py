import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_BUDGET = 320.0

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1
    
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    max_depth = 5.0
    grid_size = 500
    depth_grid = np.linspace(0.0, max_depth, grid_size)
    
    with np.errstate(over='ignore', under='ignore'):
        exp_grid = np.exp(-k * depth_grid)
        p_grid = 1.0 - np.exp(-A * exp_grid * dt)
    p_grid = np.clip(p_grid, 0.0, 1.0)
    
    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))
    
    start_time = time.time()
    deadline = start_time + (_TIME_BUDGET - (time.time() - _GLOBAL_START))
    
    P_b = p_grid[:, np.newaxis]
    P_a = p_grid[np.newaxis, :]
    
    Depths_b = depth_grid[:, np.newaxis]
    Depths_a = depth_grid[np.newaxis, :]
    
    Rewards = Depths_b * P_b + Depths_a * P_a
    
    P_plus = P_b * (1.0 - P_a)
    P_minus = (1.0 - P_b) * P_a
    P_same = (1.0 - P_b) * (1.0 - P_a)
    
    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            if n < steps - 1:
                V[n:n+1, :] = V[n+1, :]
                opt_bid[n:n+1, :] = opt_bid[n+1, :]
                opt_ask[n:n+1, :] = opt_ask[n+1, :]
            break
        
        V_next = V[n + 1]
        
        V_q_plus = np.zeros(n_q)
        V_q_minus = np.zeros(n_q)
        V_q = V_next.copy()
        
        V_q_plus[:n_q - 1] = V_next[1:]
        V_q_minus[1:] = V_next[:-1]
        
        # Values shape: (n_q, G, G)
        # We need to handle boundaries q=Q and q=-Q separately or via masking.
        
        # Standard case
        Values = Rewards[np.newaxis, :, :] - C_inv[:, np.newaxis, np.newaxis]
        Values += P_plus[np.newaxis, :, :] * V_q_plus[:, np.newaxis, np.newaxis]
        Values += P_minus[np.newaxis, :, :] * V_q_minus[:, np.newaxis, np.newaxis]
        Values += P_same[np.newaxis, :, :] * V_q[:, np.newaxis, np.newaxis]
        
        # Boundary q = Q (index n_q - 1)
        # No bid fill.
        # P_plus = 0
        # P_minus = P_a
        # P_same = 1 - P_a
        # Reward = Depths_a * P_a
        # Note: P_a is (1, G), so we need to broadcast to (G, G)
        P_a_full = np.tile(P_a, (grid_size, 1)) # (G, G)
        Rewards_q = Depths_a * P_a_full
        
        Val_Q = Rewards_q - C_inv[n_q-1]
        Val_Q += 0.0 * V_q_plus[n_q-1]
        Val_Q += P_a_full * V_q_minus[n_q-1]
        Val_Q += (1.0 - P_a_full) * V_q[n_q-1]
        
        Values[n_q-1, :, :] = Val_Q
        
        # Boundary q = -Q (index 0)
        # No ask fill.
        # P_plus = P_b
        # P_minus = 0
        # P_same = 1 - P_b
        # Reward = Depths_b * P_b
        P_b_full = np.tile(P_b, (1, grid_size)) # (G, G)
        Rewards_minusQ = Depths_b * P_b_full
        
        Val_minusQ = Rewards_minusQ - C_inv[0]
        Val_minusQ += P_b_full * V_q_plus[0]
        Val_minusQ += 0.0 * V_q_minus[0]
        Val_minusQ += (1.0 - P_b_full) * V_q[0]
        
        Values[0, :, :] = Val_minusQ
        
        flat_values = Values.reshape(n_q, -1)
        max_indices = np.argmax(flat_values, axis=1)
        
        i_idx, j_idx = np.unravel_index(max_indices, (grid_size, grid_size))
        
        opt_bid[n, :] = depth_grid[i_idx]
        opt_ask[n, :] = depth_grid[j_idx]
        V[n, :] = flat_values[np.arange(n_q), max_indices]
        
    return opt_bid, opt_ask

def _get_policy(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'],
        params['T'], params['steps'], params['Q']
    )
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    opt_bid, opt_ask = _get_policy(params)
    
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    
    if n < 0:
        n = 0
    if n >= params['steps']:
        n = params['steps'] - 1
    
    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    bid_depths = np.where(np.isnan(bid_depths), 0.01, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), 0.01, ask_depths)
    bid_depths = np.where(bid_depths < 0, 0.01, bid_depths)
    ask_depths = np.where(ask_depths < 0, 0.01, ask_depths)
    
    return bid_depths.astype(float), ask_depths.astype(float)

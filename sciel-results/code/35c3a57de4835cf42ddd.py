"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes the optimal bid and ask depths for each (time, inventory) state.
    Uses vectorized DP with a refined depth grid and boundary handling.
    """
    dt = T / steps
    # Finer grid, including 0.0
    n_depths = 150
    depths = np.linspace(0.0, 3.0, n_depths)
    
    # State space: q ranges from -Q to Q
    # V[n, q_idx] where q_idx = q + Q
    V = np.zeros((steps + 1, 2 * Q + 1))
    opt_bid = np.zeros((steps, 2 * Q + 1))
    opt_ask = np.zeros((steps, 2 * Q + 1))
    
    # Pre-compute fill probabilities for each depth
    # p(d) = 1 - exp(-A * exp(-k * d) * dt)
    with np.errstate(over='ignore', under='ignore'):
        exp_term = np.exp(-k * depths)
        p_depths = 1.0 - np.exp(-A * exp_term * dt)
    p_depths = np.clip(p_depths, 0.0, 1.0)
    
    # Pre-compute meshgrids for rewards and probabilities
    P_b = np.tile(p_depths[:, np.newaxis], (1, n_depths))
    P_a = np.tile(p_depths[np.newaxis, :], (n_depths, 1))
    
    # Rewards: b * p_b + a * p_a
    Depths_b = np.tile(depths[:, np.newaxis], (1, n_depths))
    Depths_a = np.tile(depths[np.newaxis, :], (n_depths, 1))
    Rewards = Depths_b * P_b + Depths_a * P_a
    
    # Probability matrices for state transitions
    P_plus = P_b * (1.0 - P_a)
    P_minus = P_a * (1.0 - P_b)
    P_same = 1.0 - P_plus - P_minus
    
    # Pre-compute inventory costs
    q_vals = np.arange(-Q, Q + 1)
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin for 330s limit
    
    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            break
            
        V_next = V[n+1]
        
        # Vectorize over all q indices
        V_q_plus = np.zeros((2*Q+1, 1, 1))
        V_q_minus = np.zeros((2*Q+1, 1, 1))
        V_q = np.zeros((2*Q+1, 1, 1))
        
        V_q[:, 0, 0] = V_next
        V_q_plus[:2*Q, 0, 0] = V_next[1:]
        V_q_minus[1:, 0, 0] = V_next[:-1]
        
        # Boundary masks
        mask_plus = np.ones((2*Q+1, 1, 1))
        mask_plus[-1, 0, 0] = 0.0
        
        mask_minus = np.ones((2*Q+1, 1, 1))
        mask_minus[0, 0, 0] = 0.0
        
        P_plus_masked = P_plus[np.newaxis, :, :] * mask_plus
        P_minus_masked = P_minus[np.newaxis, :, :] * mask_minus
        P_same_masked = 1.0 - P_plus_masked - P_minus_masked
        
        # Compute total value
        Values = Rewards[np.newaxis, :, :] - C_inv[:, np.newaxis, np.newaxis] + \
                 P_plus_masked * V_q_plus + \
                 P_minus_masked * V_q_minus + \
                 P_same_masked * V_q
        
        # Find max over last two axes
        flat_values = Values.reshape(2*Q+1, -1)
        max_indices = np.argmax(flat_values, axis=1)
        
        i_idx, j_idx = np.unravel_index(max_indices, (n_depths, n_depths))
        
        V[n, :] = flat_values[np.arange(2*Q+1), max_indices]
        opt_bid[n, :] = depths[i_idx]
        opt_ask[n, :] = depths[j_idx]
        
    return opt_bid, opt_ask

def _get_policy(params):
    """
    Retrieves or computes the optimal policy for the given parameters.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'], 
        params['T'], params['steps'], params['Q']
    )
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed DP policy.
    """
    # Get the policy
    opt_bid, opt_ask = _get_policy(params)
    
    # t is current time. n is step index.
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    
    # Clamp n to valid range
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    # inventory is an array of ints
    Q = params['Q']
    q_idx = inventory + Q
    
    # Ensure indices are valid
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Retrieve depths
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    # Return as float arrays
    return bid_depths.astype(float), ask_depths.astype(float)

# EVOLVE-BLOCK-END

import numpy as np
import time

# Cache for precomputed tables
_TABLE_CACHE = {}

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid and ask depths for all steps and inventories.
    Uses vectorized grid search for speed and robustness.
    """
    dt = T / steps
    N = steps
    max_q = Q
    min_q = -Q
    n_states = 2 * Q + 1
    
    # V_next is the value function for step n+1
    V_next = np.zeros(n_states)
    opt_bid = np.zeros((N, n_states))
    opt_ask = np.zeros((N, n_states))
    
    # Define a grid for depths.
    # The probability of fill is 1 - exp(-A exp(-k d) dt).
    # For large d, exp(-k d) -> 0, p -> 0.
    # A reasonable upper bound for optimal depth is where the marginal benefit
    # of increasing depth (more fill prob) is outweighed by the cost (depth earned).
    # Typically d is small. Let's use a grid from 0 to 10/k.
    # We need fine resolution near 0.
    d_max = 10.0 / k
    num_grid = 100
    # Use log-spaced or linear? Linear is safer for the very small values if k is large.
    # But exp(-k d) drops fast.
    # Let's use a linear grid with high resolution.
    b_grid = np.linspace(0.0, d_max, num_grid)
    a_grid = np.linspace(0.0, d_max, num_grid)
    
    # Precompute constants
    risk_coeff = (gamma / 2.0) * sigma**2 * dt

    # Loop backwards in time
    for n in range(N - 1, -1, -1):
        # Extract V values for q-1, q, q+1
        # q ranges from -Q to Q. Index i corresponds to q = i - Q.
        V_q = V_next
        
        # V(q+1)
        V_q_plus = np.zeros(n_states)
        V_q_plus[:-1] = V_next[1:]
        # V(q+1) for q=Q is 0 (or irrelevant as prob is 0)
        
        # V(q-1)
        V_q_minus = np.zeros(n_states)
        V_q_minus[1:] = V_next[:-1]
        # V(q-1) for q=-Q is 0
        
        # Inventory values
        q_vals = np.arange(min_q, max_q + 1)
        risk_term = risk_coeff * q_vals**2
        
        # Create meshgrids for b and a
        # Shape: (num_grid, num_grid, n_states) is too big? 
        # 100 * 100 * 61 = 610,000 floats. ~5MB. This is fine.
        # But we can do it more efficiently by iterating over one dimension or using broadcasting.
        
        # Let's broadcast b and a to shape (num_grid, 1, n_states) and (1, num_grid, n_states)?
        # No, we need p_b to depend on q? No, p_b depends only on b.
        # p_a depends only on a.
        # The coupling is in the expectation term.
        
        # p_b shape: (num_grid, 1, 1) -> broadcast to (num_grid, num_grid, n_states)
        # p_a shape: (1, num_grid, 1) -> broadcast to (num_grid, num_grid, n_states)
        
        # Calculate probabilities
        # p_b = 1 - exp(-A exp(-k b) dt)
        exp_term_b = np.exp(-k * b_grid) # shape (num_grid,)
        p_b = 1.0 - np.exp(-A * exp_term_b * dt) # shape (num_grid,)
        
        exp_term_a = np.exp(-k * a_grid) # shape (num_grid,)
        p_a = 1.0 - np.exp(-A * exp_term_a * dt) # shape (num_grid,)
        
        # We need to compute J(b,a,q) for all b, a, q.
        # J = p_b * b + p_a * a - risk_term[q] + 
        #     p_b(1-p_a)V(q+1) + (1-p_b)p_a V(q-1) + (1 - p_b(1-p_a) - (1-p_b)p_a) V(q)
        
        # Let's reshape for broadcasting:
        # b: (num_grid, 1, 1)
        # a: (1, num_grid, 1)
        # V_q_plus: (1, 1, n_states)
        # V_q: (1, 1, n_states)
        # V_q_minus: (1, 1, n_states)
        # risk_term: (1, 1, n_states)
        
        b_3d = b_grid[:, np.newaxis, np.newaxis]
        a_3d = a_grid[np.newaxis, :, np.newaxis]
        p_b_3d = p_b[:, np.newaxis, np.newaxis]
        p_a_3d = p_a[np.newaxis, :, np.newaxis]
        
        V_q_plus_3d = V_q_plus[np.newaxis, np.newaxis, :]
        V_q_3d = V_q[np.newaxis, np.newaxis, :]
        V_q_minus_3d = V_q_minus[np.newaxis, np.newaxis, :]
        risk_3d = risk_term[np.newaxis, np.newaxis, :]
        
        # Probabilities of transitions
        # P(q->q+1) = p_b * (1 - p_a)
        p_b_not_a = p_b_3d * (1.0 - p_a_3d)
        # P(q->q-1) = (1 - p_b) * p_a
        not_p_b_p_a = (1.0 - p_b_3d) * p_a_3d
        # P(q->q) = 1 - p_b_not_a - not_p_b_p_a
        p_stay = 1.0 - p_b_not_a - not_p_b_p_a
        
        # Expected Value
        E_val = p_b_not_a * V_q_plus_3d + not_p_b_p_a * V_q_minus_3d + p_stay * V_q_3d
        
        # Reward
        reward = p_b_3d * b_3d + p_a_3d * a_3d - risk_3d
        
        # Total Objective
        J = reward + E_val
        
        # Handle boundaries explicitly to ensure no invalid transitions are rewarded
        # At q=Q (last index), P(q->q+1) should be 0.
        # In our formula, V_q_plus is 0 at the end, but p_b_not_a is not 0.
        # This means we are effectively allowing a transition to a state with value 0,
        # which is equivalent to "disappearing" or getting 0 future value.
        # However, the problem states "no bid fill at q=Q".
        # This means the event "bid fill" does not occur.
        # So P(q->q+1) = 0.
        # The probability mass goes to P(q->q) or P(q->q-1)?
        # If bid fill is impossible, the "bid hit" event is false.
        # The model says: "bid hit with prob... ask hit with prob... independently".
        # "no bid fill at q=Q" implies the physical constraint.
        # Usually, this means we don't quote, or the fill prob is 0.
        # If we set p_b = 0 for q=Q, then:
        # P(q->q+1) = 0.
        # P(q->q-1) = p_a.
        # P(q->q) = 1 - p_a.
        # This matches the logic in the original code.
        
        # So we must mask the probabilities for boundary states.
        
        # Create masks
        mask_q_max = np.zeros(n_states, dtype=bool)
        mask_q_max[-1] = True
        mask_q_min = np.zeros(n_states, dtype=bool)
        mask_q_min[0] = True
        
        # For q=Q, force p_b = 0
        p_b_masked = p_b_3d.copy()
        p_b_masked[:, :, -1] = 0.0
        
        # For q=-Q, force p_a = 0
        p_a_masked = p_a_3d.copy()
        p_a_masked[:, :, 0] = 0.0
        
        # Recalculate probabilities with masked p_b and p_a
        p_b_not_a_m = p_b_masked * (1.0 - p_a_masked)
        not_p_b_p_a_m = (1.0 - p_b_masked) * p_a_masked
        p_stay_m = 1.0 - p_b_not_a_m - not_p_b_p_a_m
        
        # Recalculate Expected Value
        E_val_m = p_b_not_a_m * V_q_plus_3d + not_p_b_p_a_m * V_q_minus_3d + p_stay_m * V_q_3d
        
        # Recalculate Reward (using masked probs? No, reward is depth earned IF fill occurs.
        # If p_b=0, no fill, so no reward b. Formula p_b * b handles this.)
        reward_m = p_b_masked * b_3d + p_a_masked * a_3d - risk_3d
        
        J_m = reward_m + E_val_m
        
        # Find optimal b and a for each q
        # J_m shape: (num_grid_b, num_grid_a, n_states)
        # We want argmax over axes 0 and 1 for each q (axis 2).
        
        # Flatten the first two dimensions
        J_flat = J_m.reshape(-1, n_states)
        max_indices = np.argmax(J_flat, axis=0)
        
        # Map indices back to b and a grid indices
        # idx = i_b * num_grid_a + i_a
        i_b = max_indices // num_grid
        i_a = max_indices % num_grid
        
        opt_bid[n, :] = b_grid[i_b]
        opt_ask[n, :] = a_grid[i_a]
        
        # Update V_curr
        # We need J_m at the optimal indices for each q.
        # J_flat[max_indices, q]
        V_next = J_flat[max_indices, np.arange(n_states)]

    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    # Determine step index
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    # Check cache
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    if key not in _TABLE_CACHE:
        # Precompute
        start_time = time.time()
        opt_bid, opt_ask = _precompute_regime(
            params['sigma'], params['A'], params['k'], params['gamma'], 
            params['T'], params['steps'], params['Q']
        )
        _TABLE_CACHE[key] = (opt_bid, opt_ask)
        
        # Time guard
        if time.time() - start_time > 300:
             depth = 1.0 / params["k"]
             return np.full(len(inventory), depth), np.full(len(inventory), depth)

    opt_bid, opt_ask = _TABLE_CACHE[key]
    
    # Map inventory to indices
    q_indices = np.array(inventory) + params['Q']
    
    # Look up depths
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    
    return bid_depths, ask_depths

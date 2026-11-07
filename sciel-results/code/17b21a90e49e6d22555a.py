import numpy as np
import time

# Cache for pre-computed policy tables
_POLICY_CACHE = {}

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    n_inv = 2 * Q + 1
    # Grid of depths
    # Max depth 5/k is safe. Min 0.
    # Use 100 points
    M = 100
    D = np.linspace(0, 5.0 / k, M)
    
    # Pre-compute exponentials for the grid
    # p = 1 - exp(-A * exp(-k * d) * dt)
    exp_term = np.exp(-k * D)
    P_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    # V_next is shape (n_inv,)
    V_next = np.zeros(n_inv)
    
    # Tables to store optimal depths
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # Pre-compute inventory indices and masks
    q_indices = np.arange(n_inv)
    q_values = q_indices - Q
    
    # Masks for boundaries
    mask_no_bid = (q_values == Q)
    mask_no_ask = (q_values == -Q)
    
    # Pre-compute penalty term for each q
    # penalty[q] = 0.5 * gamma * sigma^2 * q^2 * dt
    penalty = 0.5 * gamma * sigma**2 * q_values**2 * dt
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # We need to find optimal d_b, d_a for each q
        # P_grid is shape (M,)
        
        # Create meshgrid for P_b and P_a
        # P_b shape (M, 1), P_a shape (1, M)
        Pb = P_grid[:, np.newaxis]
        Pa = P_grid[np.newaxis, :]
        
        # Effective probabilities considering boundaries
        # For q=Q, Pb must be 0.
        # For q=-Q, Pa must be 0.
        # We can apply masks after computing J, or modify Pb/Pa.
        # Since Pb/Pa are broadcasted, we can't easily mask per q in the 2D grid without 3D ops.
        # We will compute J for all q and then mask the results.
        
        # J_fill shape (M, M)
        J_fill = Pb * D[:, np.newaxis] + Pa * D[np.newaxis, :]
        
        # J_value shape (M, M, n_inv)
        # V_next[q-1], V_next[q+1], V_next[q]
        # Handle boundaries for V_next indices
        # q-1: if q=0, index -1 is invalid. But q=0 is not a boundary for inventory (Q=30).
        # Only q=Q and q=-Q are boundaries for fills.
        # For q=Q, bid is forbidden.
        # For q=-Q, ask is forbidden.
        
        # Let's compute the value contribution
        # V_q = V_next
        # V_qm1 = np.roll(V_next, 1) # q-1
        # V_qp1 = np.roll(V_next, -1) # q+1
        # But roll wraps around. We need to handle boundaries carefully.
        
        V_q = V_next
        V_qm1 = np.empty(n_inv)
        V_qp1 = np.empty(n_inv)
        
        V_qm1[0] = V_next[0] # q=-Q, q-1 is -Q-1 (invalid)
        V_qm1[1:] = V_next[:-1]
        
        V_qp1[-1] = V_next[-1] # q=Q, q+1 is Q+1 (invalid)
        V_qp1[:-1] = V_next[1:]
        
        # Term 1: Pb * (1-Pa) * V_qm1
        # Term 2: (1-Pb) * Pa * V_qp1
        # Term 3: (1 - Pb - Pa + 2 Pb Pa) * V_q
        
        # Broadcast V_qm1, V_qp1, V_q to shape (1, 1, n_inv)
        Vqm1_b = V_qm1[np.newaxis, np.newaxis, :]
        Vqp1_b = V_qp1[np.newaxis, np.newaxis, :]
        Vq_b = V_q[np.newaxis, np.newaxis, :]
        
        # Pb, Pa broadcast to (M, M, 1)
        Pb_b = Pb[:, :, np.newaxis]
        Pa_b = Pa[:, :, np.newaxis]
        
        J_val = (Pb_b * (1 - Pa_b) * Vqm1_b + 
                 (1 - Pb_b) * Pa_b * Vqp1_b + 
                 (1 - Pb_b - Pa_b + 2 * Pb_b * Pa_b) * Vq_b)
        
        # Add fill profit
        J_total = J_fill[:, :, np.newaxis] + J_val - penalty[np.newaxis, np.newaxis, :]
        
        # Apply boundary constraints
        # For q=Q (last index), bid is forbidden. So Pb should be 0.
        # We can set J_total[:, :, -1] to -inf for all d_b > 0?
        # Or simply, we know the optimal d_b for q=Q is the one that minimizes Pb (max d_b).
        # So we can restrict the search for q=Q to d_b = D[-1].
        # Similarly for q=-Q, d_a = D[-1].
        
        # Let's mask out invalid combinations.
        # If q == Q, we only allow d_b = D[-1] (approx infinity).
        # If q == -Q, we only allow d_a = D[-1].
        
        # Create a mask of shape (M, M, n_inv)
        # This is memory heavy if done naively.
        
        # Alternative:
        # Compute the argmax for all.
        # Then for q=Q, force idx_b = M-1.
        # For q=-Q, force idx_a = M-1.
        
        # Find indices of max J_total
        # Reshape to (M*M, n_inv)
        J_flat = J_total.reshape(-1, n_inv)
        max_indices = np.argmax(J_flat, axis=0)
        
        # Convert back to 2D indices
        idx_b = max_indices // M
        idx_a = max_indices % M
        
        # Apply boundary corrections
        # q=Q is index Q+Q = 2Q
        # q=-Q is index 0
        idx_b[2*Q] = M - 1
        idx_a[0] = M - 1
        
        # Store optimal depths
        bid_table[n, :] = D[idx_b]
        ask_table[n, :] = D[idx_a]
        
        # Update V_next
        V_next = J_flat[max_indices, np.arange(n_inv)]
        
    return bid_table, ask_table

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _POLICY_CACHE:
        # Compute table
        # This might take a few seconds
        bid_table, ask_table = _compute_policy_table(sigma, A, k, gamma, T, steps, Q)
        _POLICY_CACHE[key] = (bid_table, ask_table)
    else:
        bid_table, ask_table = _POLICY_CACHE[key]
    
    # Determine current step
    # t = n * dt
    dt = T / steps
    n = int(round(t / dt))
    # Clamp n to valid range
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    # Get depths for current step and inventory
    # inventory is an array of ints
    # Map inventory to index: idx = inventory + Q
    idx = inventory + Q
    
    # Ensure idx is within bounds
    idx = np.clip(idx, 0, 2*Q)
    
    bid_depths = bid_table[n, idx]
    ask_depths = ask_table[n, idx]
    
    return bid_depths, ask_depths

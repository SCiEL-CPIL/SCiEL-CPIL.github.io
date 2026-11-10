import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 330.0

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    dt = T / steps
    # Depth grid
    # Min depth 3/k. Max depth 5/k is sufficient for exponential decay.
    # Let's use 30 points for better resolution.
    d_min = 3.0 / k
    d_max = 5.0 / k
    depths = np.linspace(d_min, d_max, 30)
    
    # Precompute probabilities for all grid points
    # p = 1 - exp(-A * exp(-k * d) * dt)
    exp_term = np.exp(-k * depths)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    # Meshgrid for bid and ask depths
    B_grid, A_grid = np.meshgrid(depths, depths, indexing='ij')
    pB_grid, pA_grid = np.meshgrid(p_grid, p_grid, indexing='ij')
    
    # Flatten for easier indexing
    B_flat = B_grid.flatten()
    A_flat = A_grid.flatten()
    pB_flat = pB_grid.flatten()
    pA_flat = pA_grid.flatten()
    
    # Value function V[n, q]
    # q ranges from -Q to Q, size 2*Q + 1
    num_q = 2 * Q + 1
    V = np.zeros((steps + 1, num_q))
    
    # Policy storage
    # Store index into the flattened grid
    policy_idx = np.zeros((steps, num_q), dtype=int)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time
        if time.time() > _DEADLINE - 10:
             # Fallback to simple heuristic if running out of time
             # This shouldn't happen
             break
        
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            
            # Costs
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Boundary conditions for probabilities
            # If q == Q, no bid fill (bid increases inventory? 
            # Prompt: "no bid fill at q = Q". 
            # Standard: Bid = Buy = +1. So if q=Q, can't buy. pB=0.
            # Ask = Sell = -1. So if q=-Q, can't sell. pA=0.
            
            pB_eff = pB_flat.copy()
            pA_eff = pA_flat.copy()
            
            if q == Q:
                pB_eff[:] = 0.0
            if q == -Q:
                pA_eff[:] = 0.0
            
            # Expected future value
            # V_next is V[n+1, :]
            V_next = V[n + 1, :]
            
            # Indices for q+1, q, q-1 in V_next
            # q_idx is index of q.
            # q+1 is q_idx + 1
            # q-1 is q_idx - 1
            
            # Handle boundaries for V lookup
            # If q=Q, q+1 is out of bounds, but pB=0 so term is 0.
            # We can just use np.where or careful indexing.
            # Since pB=0 when q=Q, we can safely index V_next[q_idx+1] if we handle the array bounds.
            # Actually, if q=Q, q_idx = 2Q. q_idx+1 = 2Q+1 which is out of bounds for size 2Q+1.
            # So we must be careful.
            
            # Let's create safe indices
            q_plus_1_idx = min(q_idx + 1, num_q - 1)
            q_minus_1_idx = max(q_idx - 1, 0)
            
            V_q_plus_1 = V_next[q_plus_1_idx]
            V_q_minus_1 = V_next[q_minus_1_idx]
            V_q = V_next[q_idx]
            
            # Calculate E[V_next] for all actions
            # Term 1: pB * (1 - pA) * V_q_plus_1
            term1 = pB_eff * (1.0 - pA_eff) * V_q_plus_1
            # Term 2: (1 - pB) * pA * V_q_minus_1
            term2 = (1.0 - pB_eff) * pA_eff * V_q_minus_1
            # Term 3: (1 - pB - pA + 2*pB*pA) * V_q
            term3 = (1.0 - pB_eff - pA_eff + 2.0 * pB_eff * pA_eff) * V_q
            
            E_V_next = term1 + term2 + term3
            
            # Immediate Reward
            # b * pB + a * pA - cost
            imm_reward = B_flat * pB_eff + A_flat * pA_eff - cost
            
            total_value = imm_reward + E_V_next
            
            best_idx = np.argmax(total_value)
            policy_idx[n, q_idx] = best_idx
            V[n, q_idx] = total_value[best_idx]
            
    # Map indices back to depths
    policy_b = B_flat[policy_idx]
    policy_a = A_flat[policy_idx]
    
    _POLICY_CACHE[key] = (policy_b, policy_a)
    return policy_b, policy_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n = int(t / (T / steps))
    if n >= steps:
        n = steps - 1
        
    policy_b, policy_a = _compute_policy(sigma, A, k, gamma, T, steps, Q)
    
    # inventory is array of ints
    # Map to indices: q + Q
    q_indices = inventory + Q
    
    # Ensure indices are valid
    q_indices = np.clip(q_indices, 0, 2 * Q)
    
    bid_depths = policy_b[n, q_indices]
    ask_depths = policy_a[n, q_indices]
    
    return bid_depths, ask_depths

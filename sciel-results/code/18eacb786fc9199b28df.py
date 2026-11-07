import numpy as np
import time

# Global cache for DP solutions
_dp_cache = {}

def _solve_dp(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    # Extend grid to handle boundary transitions
    Q_ext = Q + 2
    n_states = 2 * Q_ext + 1
    q_min = -Q_ext
    q_max = Q_ext
    
    # V[n, q_index]
    # q_index = q - q_min
    V = np.zeros((steps + 1, n_states), dtype=np.float64)
    Db = np.zeros((steps, n_states), dtype=np.float64)
    Da = np.zeros((steps, n_states), dtype=np.float64)
    
    # Precompute constants
    penalty_coeff = 0.5 * gamma * sigma**2 * dt
    lambda_base = A * dt
    
    for n in range(steps - 1, -1, -1):
        V_next = V[n + 1]
        for i, q in enumerate(range(q_min, q_max + 1)):
            # Get deltas
            # Handle boundaries of the extended grid
            # If q is at the edge of extended grid, assume V is 0 or constant?
            # For q = q_min, q-1 is out. Assume V(q-1)=V(q_min)?
            # Let's just clamp indices.
            
            idx_m1 = max(0, i - 1)
            idx_m0 = i
            idx_p1 = min(n_states - 1, i + 1)
            
            V_m1 = V_next[idx_m1]
            V_m0 = V_next[idx_m0]
            V_p1 = V_next[idx_p1]
            
            delta_minus = V_m1 - V_m0
            delta_plus = V_p1 - V_m0
            delta_cross = 2 * V_m0 - V_m1 - V_p1
            
            # Determine if we can quote
            # Original bounds: [-Q, Q]
            # If q > Q, no bid fill? Contract says "no bid fill at q=Q".
            # If q > Q, we are already out. Let's assume standard logic applies or clip.
            # The contract says inventory is in [-Q, Q].
            # So we only really care about q in [-Q, Q].
            # For q outside, we can just copy the boundary policy.
            
            can_bid = True
            can_ask = True
            if q >= Q:
                can_bid = False
            if q <= -Q:
                can_ask = False
            
            if can_bid and can_ask:
                # Fixed point iteration
                # Start with p_b, p_a from d=1/k
                d_b = 1.0 / k
                d_a = 1.0 / k
                p_b = 1.0 - np.exp(-lambda_base * np.exp(-k * d_b))
                p_a = 1.0 - np.exp(-lambda_base * np.exp(-k * d_a))
                
                for _ in range(10):
                    K_b = delta_minus + p_a * delta_cross
                    K_a = delta_plus + p_b * delta_cross
                    
                    d_b_new = 1.0 / k - K_b
                    d_a_new = 1.0 / k - K_a
                    
                    # Clip depths
                    min_depth = -3.0 / k + 1e-6
                    if d_b_new < min_depth: d_b_new = min_depth
                    if d_a_new < min_depth: d_a_new = min_depth
                    
                    p_b_new = 1.0 - np.exp(-lambda_base * np.exp(-k * d_b_new))
                    p_a_new = 1.0 - np.exp(-lambda_base * np.exp(-k * d_a_new))
                    
                    if abs(p_b_new - p_b) < 1e-8 and abs(p_a_new - p_a) < 1e-8:
                        p_b, p_a = p_b_new, p_a_new
                        d_b, d_a = d_b_new, d_a_new
                        break
                    
                    p_b, p_a = p_b_new, p_a_new
                    d_b, d_a = d_b_new, d_a_new
                
                Db[n, i] = d_b
                Da[n, i] = d_a
                
                # Calculate Value
                # E[V_next]
                # Probabilities:
                # both: p_b * p_a -> q
                # bid only: p_b * (1-p_a) -> q-1
                # ask only: (1-p_b) * p_a -> q+1
                # none: (1-p_b) * (1-p_a) -> q
                
                p_both = p_b * p_a
                p_bid_only = p_b * (1.0 - p_a)
                p_ask_only = (1.0 - p_b) * p_a
                p_none = (1.0 - p_b) * (1.0 - p_a)
                
                # Map q-1, q, q+1 to indices
                idx_qm1 = max(0, i - 1)
                idx_q = i
                idx_qp1 = min(n_states - 1, i + 1)
                
                EV_next = (p_both * V_next[idx_q] + 
                           p_bid_only * V_next[idx_qm1] + 
                           p_ask_only * V_next[idx_qp1] + 
                           p_none * V_next[idx_q])
                
                reward = p_b * d_b + p_a * d_a
                penalty = penalty_coeff * q * q
                
                V[n, i] = reward + EV_next - penalty

            elif can_bid:
                # No ask fill
                p_a = 0.0
                d_a = 0.0 # Irrelevant, but set to something valid
                Da[n, i] = 0.0
                
                K_b = delta_minus
                d_b = 1.0 / k - K_b
                min_depth = -3.0 / k + 1e-6
                if d_b < min_depth: d_b = min_depth
                
                p_b = 1.0 - np.exp(-lambda_base * np.exp(-k * d_b))
                Db[n, i] = d_b
                
                # E[V_next]
                # bid only: p_b -> q-1
                # none: 1-p_b -> q
                EV_next = p_b * V_next[max(0, i-1)] + (1.0 - p_b) * V_next[i]
                
                reward = p_b * d_b
                penalty = penalty_coeff * q * q
                V[n, i] = reward + EV_next - penalty

            elif can_ask:
                # No bid fill
                p_b = 0.0
                d_b = 0.0
                Db[n, i] = 0.0
                
                K_a = delta_plus
                d_a = 1.0 / k - K_a
                min_depth = -3.0 / k + 1e-6
                if d_a < min_depth: d_a = min_depth
                
                p_a = 1.0 - np.exp(-lambda_base * np.exp(-k * d_a))
                Da[n, i] = d_a
                
                # E[V_next]
                # ask only: p_a -> q+1
                # none: 1-p_a -> q
                EV_next = p_a * V_next[min(n_states-1, i+1)] + (1.0 - p_a) * V_next[i]
                
                reward = p_a * d_a
                penalty = penalty_coeff * q * q
                V[n, i] = reward + EV_next - penalty
            else:
                # Neither can fill? (Should not happen if Q>=1)
                V[n, i] = V_next[i] - penalty_coeff * q * q
                Db[n, i] = 0.0
                Da[n, i] = 0.0

    return V, Db, Da, q_min, Q

def quote(t, inventory, mid, params):
    # Identify regime
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _dp_cache:
        _dp_cache[key] = _solve_dp(params)
    
    V, Db, Da, q_min, Q = _dp_cache[key]
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    
    # Determine time step index
    # t = n * dt
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    # Inventory to index
    # inventory is array of ints
    # Map to q_min..q_max
    # q_min = -Q-2
    # We need to handle inventory outside [-Q, Q] by clipping to boundary policies
    
    inv = np.asarray(inventory)
    
    # Clip inventory to valid DP range for lookup
    # The DP was computed for -Q-2 to Q+2
    # But the "no fill" logic is applied based on original Q.
    # If inv > Q, we should use the policy for Q? Or Q+1?
    # The contract says inv is in [-Q, Q].
    # So we just map inv directly.
    
    # Indices in Db/Da
    # Db[n, idx]
    # idx = inv - q_min
    
    idx = inv - q_min
    
    # Clip indices to valid range
    idx = np.clip(idx, 0, len(Db[n]) - 1)
    
    bid_depths = Db[n, idx]
    ask_depths = Da[n, idx]
    
    # Ensure valid depths
    min_depth = -3.0 / params['k'] + 1e-6
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Replace NaN/inf
    bid_depths = np.nan_to_num(bid_depths, nan=1.0/params['k'], posinf=1.0/params['k'], neginf=min_depth)
    ask_depths = np.nan_to_num(ask_depths, nan=1.0/params['k'], posinf=1.0/params['k'], neginf=min_depth)
    
    return bid_depths, ask_depths

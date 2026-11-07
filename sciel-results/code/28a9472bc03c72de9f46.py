"""Market-making with inventory risk: Direct Full-Horizon Dynamic Programming."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_T0 = time.time()
_DEADLINE = 295.0  # Safety margin before 300s limit

# Cache for optimized parameters per regime
# Key: (sigma, A, k, gamma, T, steps, Q)
_DP_CACHE = {}

def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the optimal policy using backward induction (Dynamic Programming).
    
    The state is the inventory q in [-Q, Q].
    The action is the pair of depths (bid_depth, ask_depth).
    
    Since the mid-price does not affect the inventory dynamics or the reward (depths are relative),
    we can ignore the mid-price process entirely for the optimization.
    
    We discretize the depth space to make the DP tractable.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _DP_CACHE:
        return _DP_CACHE[key]

    dt = T / steps
    n_states = 2 * Q + 1
    
    # Value function V[q_index] where q_index = q + Q
    # Initialize V to 0 for the final step (no terminal charge)
    V = np.zeros(n_states)
    
    # We need to store the optimal policy for each step and state.
    # Policy[t][q_index] = (bid_depth, ask_depth)
    # Since steps=1000 and n_states=61, storing a full 1000x61x2 array is fine.
    # However, we only need the policy for the current step t in the evaluator.
    # But the evaluator calls quote(t, ...) sequentially. 
    # We can pre-compute the entire policy table if memory allows, 
    # or compute it on the fly if we store the value functions?
    # No, the policy depends on the future value functions.
    # We must compute the policy for all steps t=0..999 and store them.
    # 1000 steps * 61 states * 2 floats * 8 bytes = ~1 MB. Very cheap.
    
    policy_bid = np.zeros((steps, n_states))
    policy_ask = np.zeros((steps, n_states))
    
    # Depth discretization
    # The optimal depth is likely in [0, 3/k]. 
    # Let's use a fine grid.
    # For k=1.5, 3/k = 2. 
    # For k=0.75, 3/k = 4.
    # Let's use a grid of 100 points for depth.
    n_depths = 100
    max_depth = 4.0 / k  # Safe upper bound
    depths_grid = np.linspace(0.0, max_depth, n_depths)
    
    # Pre-compute fill probabilities for all depths and states?
    # p_bid(q, d) = 1 - exp(-A * exp(-k * d) * dt) if q < Q else 0
    # p_ask(q, d) = 1 - exp(-A * exp(-k * d) * dt) if q > -Q else 0
    
    # Note: The reward for a bid fill is the depth d.
    # The reward for an ask fill is the depth d.
    # The risk cost is (gamma / 2) * sigma^2 * q^2 * dt.
    
    # Backward induction
    # For t from steps-1 down to 0:
    #   For each state q:
    #     For each bid_depth in grid:
    #       For each ask_depth in grid:
    #         Compute expected value
    #         Track max
    
    # This is O(steps * n_states * n_depths^2).
    # 1000 * 61 * 100^2 = 610,000,000 operations.
    # This might be too slow in Python loops. We need to vectorize over depths.
    
    # Let's vectorize over the depth grid for a given state.
    # For a fixed q, we want to maximize over (db, da):
    #   (1 - p_b) * (1 - p_a) * V[q] 
    #   + p_b * (1 - p_a) * (db + V[q-1])
    #   + p_a * (1 - p_b) * (da + V[q+1])
    #   + p_b * p_a * (db + da + V[q])
    #   - risk_cost
    #
    # Where p_b = 1 - exp(-A * exp(-k * db) * dt) (0 if q==Q)
    #       p_a = 1 - exp(-A * exp(-k * da) * dt) (0 if q==-Q)
    #
    # This can be rearranged to:
    #   V[q] + p_b * db + p_a * da + p_b * p_a * (V[q] - V[q] - V[q])? No.
    #   Let's expand:
    #   E[V_next] = (1-p_b)(1-p_a)V[q] + p_b(1-p_a)V[q-1] + p_a(1-p_b)V[q+1] + p_b p_a V[q]
    #   = V[q] - p_b V[q] - p_a V[q] + p_b p_a V[q] + p_b V[q-1] - p_b p_a V[q-1] + p_a V[q+1] - p_a p_a V[q+1] + p_b p_a V[q]
    #   Wait, simpler:
    #   E[V_next] = V[q] + p_b(V[q-1] - V[q]) + p_a(V[q+1] - V[q]) + p_b p_a (V[q] - V[q-1] - V[q+1] + V[q])? No.
    #   
    #   Let's stick to the direct formula for clarity and vectorization.
    #   For a fixed q, we have arrays db (n_depths,) and da (n_depths,).
    #   We create a meshgrid of db and da? That's n_depths^2 = 10,000 points.
    #   For each of the 61 states, we do this.
    #   1000 steps * 61 states * (vectorized ops on 10,000 points)
    #   1000 * 61 = 61,000 iterations.
    #   Each iteration involves a few exp, multiplications, and argmax on 10,000 elements.
    #   This should be fast enough in NumPy.
    
    # Pre-compute exp(-k * depths_grid)
    exp_k_d = np.exp(-k * depths_grid)
    # p_d = 1 - exp(-A * exp_k_d * dt)
    p_d = 1.0 - np.exp(-A * exp_k_d * dt)
    
    # Loop backwards in time
    for t in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE:
            # Fallback to a simple heuristic if we run out of time
            # This shouldn't happen, but safety first.
            break
        
        for qi in range(n_states):
            q = qi - Q
            
            # Get current value function
            V_q = V[qi]
            
            # Handle boundaries for next states
            if q == Q:
                # No bid fill possible
                p_b_arr = np.zeros(n_depths)
                V_q_minus = V_q # q-1 is not reached, effectively stays at q if no fill? 
                # Actually, if q=Q, bid fill prob is 0. So q stays Q if no ask fill.
                # If ask fill, q becomes Q-1.
                # So V[q-1] is V[Q-1].
                V_qm1 = V[qi - 1] if qi > 0 else 0.0
            else:
                p_b_arr = p_d
                V_qm1 = V[qi - 1]
            
            if q == -Q:
                # No ask fill possible
                p_a_arr = np.zeros(n_depths)
                V_qp1 = V_q # q+1 not reached
                V_qp1 = V[qi + 1] if qi < n_states - 1 else 0.0
            else:
                p_a_arr = p_d
                V_qp1 = V[qi + 1]
            
            # We need to maximize over db, da.
            # Create 2D arrays for db, da, p_b, p_a
            # db_grid, da_grid = np.meshgrid(depths_grid, depths_grid)
            # This is memory intensive if done repeatedly. 
            # Better to use broadcasting.
            
            # db is shape (n_depths, 1)
            # da is shape (1, n_depths)
            db_col = depths_grid[:, np.newaxis]
            da_row = depths_grid[np.newaxis, :]
            
            p_b_2d = p_b_arr[:, np.newaxis]
            p_a_2d = p_a_arr[np.newaxis, :]
            
            # Calculate Expected Next Value
            # E[V_next] = (1-p_b)(1-p_a)V_q + p_b(1-p_a)V_qm1 + p_a(1-p_b)V_qp1 + p_b p_a V_q
            # Note: V_q is the value if inventory doesn't change.
            # V_qm1 is value if inventory decreases (bid fill).
            # V_qp1 is value if inventory increases (ask fill).
            
            term_00 = (1 - p_b_2d) * (1 - p_a_2d) * V_q
            term_10 = p_b_2d * (1 - p_a_2d) * V_qm1
            term_01 = p_a_2d * (1 - p_b_2d) * V_qp1
            term_11 = p_b_2d * p_a_2d * V_q
            
            E_V_next = term_00 + term_10 + term_01 + term_11
            
            # Immediate Reward
            # Reward = db * I(bid fill) + da * I(ask fill) - risk
            # E[Immediate] = p_b * db + p_a * da - 0.5 * gamma * sigma^2 * q^2 * dt
            
            risk = 0.5 * gamma * sigma**2 * q**2 * dt
            E_Immediate = p_b_2d * db_col + p_a_2d * da_row - risk
            
            Total_Value = E_V_next + E_Immediate
            
            # Find max
            max_val, max_idx = np.unravel_index(np.argmax(Total_Value), Total_Value.shape)
            
            policy_bid[t, qi] = depths_grid[max_idx[0]]
            policy_ask[t, qi] = depths_grid[max_idx[1]]
            
            # Update Value Function for this state
            V[qi] = Total_Value[max_idx[0], max_idx[1]]
            
    _DP_CACHE[key] = (policy_bid, policy_ask)
    return _DP_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Quote policy using pre-computed DP table.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Get optimized policy tables
    policy_bid, policy_ask = _solve_dp(sigma, A, k, gamma, T, steps, Q)
    
    # Map inventory to state index
    # inventory is in [-Q, Q]
    # index = inventory + Q
    q_indices = inventory + Q
    
    # Ensure indices are valid (should be, given problem constraints)
    q_indices = np.clip(q_indices, 0, 2 * Q)
    
    # Get depths for current step t
    # t is the step index n
    bid_depths = policy_bid[t, q_indices]
    ask_depths = policy_ask[t, q_indices]
    
    # Ensure valid depths
    # The DP grid is [0, 4/k], so depths are already valid.
    # Just in case, clip to [0, inf]
    bid_depths = np.clip(bid_depths, 0.0, np.inf)
    ask_depths = np.clip(ask_depths, 0.0, np.inf)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END

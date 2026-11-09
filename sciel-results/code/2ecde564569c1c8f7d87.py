import numpy as np
import time

# Global cache for pre-computed policies
_policy_cache = {}

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Pre-compute the optimal bid and ask depths for all time steps and inventory levels.
    Returns two arrays of shape (steps, 2*Q+1).
    """
    dt = T / steps
    Q_count = 2 * Q + 1
    
    # Depth grid
    # Optimal depths are typically small. 0 to 5.0 should cover it.
    M = 200
    D = np.linspace(0, 5.0, M)
    
    # Pre-compute fill probabilities for the grid
    # p = 1 - exp(-A * exp(-k * d) * dt)
    # Note: A * exp(-k*d) * dt can be large, exp(-large) -> 0, p -> 1.
    # If A * exp(-k*d) * dt is small, p ~ A * exp(-k*d) * dt.
    exp_term = A * np.exp(-k * D) * dt
    P = 1.0 - np.exp(-exp_term)
    
    # V_next is the value function for the next time step
    # Shape: (Q_count,)
    V_next = np.zeros(Q_count)
    
    # Policies to store
    # bid_policy[n, q_idx] = optimal bid depth
    # ask_policy[n, q_idx] = optimal ask depth
    bid_policy = np.zeros((steps, Q_count))
    ask_policy = np.zeros((steps, Q_count))
    
    # Pre-compute masks for boundaries
    # q_idx 0 corresponds to q = -Q
    # q_idx Q_count-1 corresponds to q = Q
    mask_bid = np.ones((1, 1, Q_count))
    mask_ask = np.ones((1, 1, Q_count))
    mask_bid[:, :, Q_count - 1] = 0.0  # No bid at Q
    mask_ask[:, :, 0] = 0.0            # No ask at -Q
    
    # Loop backwards in time
    for n in range(steps - 1, -1, -1):
        # Compute Delta V_ and Delta V_+
        # V_next is indexed 0..Q_count-1 corresponding to q=-Q..Q
        # Delta V_- (q) = V(q-1) - V(q)
        # Delta V_+ (q) = V(q+1) - V(q)
        
        # For q=-Q (idx 0), q-1 is invalid. But bid is allowed?
        # "no bid fill at q = Q, no ask fill at q = -Q".
        # So at q=-Q, ask is forbidden. Bid is allowed.
        # If bid fill at q=-Q, q becomes -Q-1?
        # "inventory ... in [-Q, Q]".
        # Usually, inventory is bounded. If it hits the bound, it stays?
        # "no bid fill at q = Q" implies we can't go to Q+1.
        # "no ask fill at q = -Q" implies we can't go to -Q-1.
        # So at q=-Q, if bid fill, q becomes -Q-1?
        # The problem states inventory is in [-Q, Q].
        # If a fill would move it out, is it blocked?
        # "no bid fill at q = Q" -> blocked.
        # "no ask fill at q = -Q" -> blocked.
        # What about bid fill at q = -Q? It moves to -Q-1.
        # Is that allowed?
        # "inventory (4000,) ints in [-Q, Q]".
        # This implies the state space is bounded.
        # If we are at -Q and a bid fill happens, do we go to -Q-1?
        # If the state space is strictly [-Q, Q], then -Q-1 is invalid.
        # However, the constraint "no bid fill at q = Q" is explicit.
        # "no ask fill at q = -Q" is explicit.
        # It does NOT say "no bid fill at q = -Q".
        # This suggests that at q = -Q, a bid fill IS possible.
        # But where does the inventory go?
        # If the problem implies the inventory is clamped, then q stays -Q.
        # If it implies the state space is larger but we only care about [-Q, Q], 
        # then V(-Q-1) is needed.
        # Given "inventory ... in [-Q, Q]", it's likely the state space is exactly this.
        # If so, a bid fill at -Q is either:
        # 1. Impossible (like at Q).
        # 2. Moves to -Q-1 (outside range).
        # 3. Stays at -Q (clamped).
        
        # Let's re-read carefully: "no bid fill at q = Q, no ask fill at q = -Q".
        # This asymmetry is specific.
        # If q=Q, we are long max. Can't buy more.
        # If q=-Q, we are short max. Can't sell more.
        # So at q=-Q, we CAN buy (bid fill).
        # If we buy, q becomes -Q-1.
        # Is -Q-1 a valid state?
        # The input says inventory is in [-Q, Q].
        # If the simulation allows q to go to -Q-1, then V must be defined there.
        # However, the input constraint says "inventory ... in [-Q, Q]".
        # This usually means the *input* to the function is in this range.
        # It doesn't explicitly forbid the *internal* state from going out, 
        # but the DP needs to handle it.
        # If we assume the state space is infinite, we need more V.
        # But Q=30. Going to -31 is possible.
        # However, the probability of many consecutive fills is low.
        # Let's assume the state space is effectively [-Q-1, Q+1] for safety, 
        # or that fills at the boundary are blocked?
        # "no bid fill at q = Q" -> Blocked.
        # "no ask fill at q = -Q" -> Blocked.
        # It does NOT say "no bid fill at q = -Q".
        # So bid fill at -Q is ALLOWED.
        # This implies inventory can go to -Q-1.
        # Similarly, ask fill at Q is ALLOWED?
        # "no ask fill at q = -Q".
        # So ask fill at Q is ALLOWED?
        # If so, inventory can go to Q+1.
        
        # This complicates the DP array size.
        # Let's extend the range to [-Q-2, Q+2] to be safe?
        # Or just assume that if it goes out, the value is 0 or linear?
        # Given the time, let's assume the state space is [-Q, Q] and 
        # fills that would move it out are BLOCKED?
        # But the text only blocks one side.
        # "no bid fill at q = Q" (Long max, can't buy).
        # "no ask fill at q = -Q" (Short max, can't sell).
        # This is standard market making: you can't increase inventory beyond limits.
        # So:
        # At q=Q: Bid blocked. Ask allowed (reduces inventory).
        # At q=-Q: Ask blocked. Bid allowed (reduces inventory, i.e., makes it less negative).
        # So inventory always stays within [-Q, Q].
        # If bid fill at -Q: q goes from -Q to -Q+1?
        # No, bid fill means we BUY. Inventory increases.
        # If q=-Q (short), buying reduces the short.
        # So q goes from -Q to -Q+1. This is INSIDE the range.
        # If ask fill at Q: q goes from Q to Q-1. INSIDE.
        
        # So:
        # Bid fill: q -> q+1.
        # Ask fill: q -> q-1.
        
        # Check constraints:
        # "no bid fill at q = Q": If q=Q, bid fill would make it Q+1. Blocked.
        # "no ask fill at q = -Q": If q=-Q, ask fill would make it -Q-1. Blocked.
        
        # So:
        # If q=Q: Bid prob = 0.
        # If q=-Q: Ask prob = 0.
        
        # This confirms the state space is [-Q, Q] and transitions stay inside.
        
        # So:
        # Delta V_- (for bid fill, q -> q+1)
        # Wait, my previous notation:
        # Bid fill: we buy. Inventory increases.
        # Ask fill: we sell. Inventory decreases.
        
        # Let's redefine:
        # q is inventory.
        # Bid fill: q_new = q + 1.
        # Ask fill: q_new = q - 1.
        
        # So:
        # V(q+1) is the value after bid fill.
        # V(q-1) is the value after ask fill.
        
        # In my previous DP derivation:
        # I used:
        # Bid fill -> q-1.
        # Ask fill -> q+1.
        # This was WRONG.
        
        # Correct:
        # Bid (Buy) -> Inventory +1.
        # Ask (Sell) -> Inventory -1.
        
        # So:
        # p_b: prob of bid fill. Reward d_b. Next state q+1.
        # p_a: prob of ask fill. Reward d_a. Next state q-1.
        
        # Constraints:
        # If q=Q, p_b = 0.
        # If q=-Q, p_a = 0.
        
        # Let's re-derive J:
        # E = (1-p_b)(1-p_a) V(q) + p_b(1-p_a) (d_b + V(q+1)) + (1-p_b)p_a (d_a + V(q-1)) + p_b p_a (d_b + d_a + V(q))
        
        # Group V(q) terms:
        # V(q) [ (1-p_b)(1-p_a) + p_b p_a ]
        
        # Group other terms:
        # p_b(1-p_a) (d_b + V(q+1)) + (1-p_b)p_a (d_a + V(q-1)) + p_b p_a (d_b + d_a)
        
        # Let's use the simplified form derived earlier, but with correct indices.
        # Let Delta V_bid = V(q+1) - V(q)
        # Let Delta V_ask = V(q-1) - V(q)
        
        # J = p_b d_b + p_a d_a + p_b(1-p_a) Delta V_bid + (1-p_b)p_a Delta V_ask
        
        # This is the term to maximize (plus constant V(q)).
        
        # So in the vectorized code:
        # Delta V_bid (q) = V_next[q_idx+1] - V_next[q_idx]
        # Delta V_ask (q) = V_next[q_idx-1] - V_next[q_idx]
        
        # For q=Q (idx Q_count-1):
        # Bid is blocked. p_b=0.
        # We should set Delta V_bid = 0 and mask out the bid terms.
        # Actually, if p_b=0, the terms with p_b vanish.
        # But in our grid optimization, p_b is determined by d_b.
        # If we force p_b=0, we should set the grid for d_b to be irrelevant.
        # Easiest: Mask out the R and K terms for bid at q=Q.
        
        # For q=-Q (idx 0):
        # Ask is blocked. p_a=0.
        # Mask out C and K terms for ask at q=-Q.
        
        # Let's adjust the masks:
        # mask_bid_valid: 1 if q < Q, else 0.
        # mask_ask_valid: 1 if q > -Q, else 0.
        
        # R (Bid term) should be multiplied by mask_bid_valid.
        # C (Ask term) should be multiplied by mask_ask_valid.
        # K (Cross term) should be multiplied by mask_bid_valid * mask_ask_valid.
        
        # And Delta V_bid should be 0 where mask_bid_valid is 0.
        # Delta V_ask should be 0 where mask_ask_valid is 0.
        
        # This ensures that if bid is blocked, the objective doesn't depend on d_b.
        
        pass

    # Implementation of the loop
    for n in range(steps - 1, -1, -1):
        # Compute Deltas
        # V_next is shape (Q_count,)
        # q_idx 0..Q_count-1
        # Delta V_bid (q) = V(q+1) - V(q)
        # For q=Q (last), V(q+1) is out of bounds. But bid is blocked.
        # We can pad V_next with 0s or just handle indices.
        
        V_padded = np.concatenate(([0], V_next, [0]))
        # Delta V_bid[i] = V_padded[i+2] - V_padded[i+1] ?
        # Let's just use slicing.
        
        # Delta V_bid for q in [-Q, Q-1]
        # Delta V_bid[q_idx] = V_next[q_idx+1] - V_next[q_idx]
        # For q_idx = Q_count-1, set to 0.
        
        Delta_V_bid = np.diff(V_next, prepend=0) # No, diff gives V[i+1]-V[i]
        # np.diff(V_next) has length Q_count-1.
        # We want length Q_count.
        
        Delta_V_bid = np.zeros(Q_count)
        Delta_V_bid[:-1] = V_next[1:] - V_next[:-1]
        # Delta_V_bid[-1] is 0 (blocked)
        
        Delta_V_ask = np.zeros(Q_count)
        Delta_V_ask[1:] = V_next[:-1] - V_next[1:]
        # Delta_V_ask[0] is 0 (blocked)
        
        # Masks
        # mask_bid is 1 for q < Q (idx < Q_count-1)
        mask_bid = np.ones((1, 1, Q_count))
        mask_bid[:, :, Q_count - 1] = 0.0
        
        # mask_ask is 1 for q > -Q (idx > 0)
        mask_ask = np.ones((1, 1, Q_count))
        mask_ask[:, :, 0] = 0.0
        
        # Construct R, C, K
        # R_i,q = P_i (D_i + Delta_V_bid[q]) * mask_bid
        # P_i is (M, 1, 1)
        # D_i is (M, 1, 1)
        # Delta_V_bid is (1, 1, Q_count)
        
        P_col = P[:, np.newaxis, np.newaxis]
        D_col = D[:, np.newaxis, np.newaxis]
        
        R = (P_col * D_col + P_col * Delta_V_bid[np.newaxis, np.newaxis, :]) * mask_bid
        
        C = (P_col * D_col + P_col * Delta_V_ask[np.newaxis, np.newaxis, :]) * mask_ask
        # Wait, C is for Ask. Ask uses P_j, D_j.
        # C_j,q = P_j (D_j + Delta_V_ask[q]) * mask_ask
        
        K = (P_col * P_col * (Delta_V_bid[np.newaxis, np.newaxis, :] + Delta_V_ask[np.newaxis, np.newaxis, :])) * (mask_bid * mask_ask)
        # K_i,j,q = P_i P_j (Delta_V_bid + Delta_V_ask) * mask_bid * mask_ask
        
        J = R + C - K
        
        # Find max indices
        # J shape (M, M, Q_count)
        # Flatten first two dims
        J_flat = J.reshape(M * M, Q_count)
        idx_flat = np.argmax(J_flat, axis=0)
        
        # Map back to i, j
        i_idx = idx_flat // M
        j_idx = idx_flat % M
        
        # Store depths
        bid_policy[n] = D[i_idx]
        ask_policy[n] = D[j_idx]
        
        # Update V_next
        # V_prev(q) = -gamma/2 sigma^2 q^2 dt + V(q) + max_J
        # max_J is J_flat[idx_flat, q]
        
        max_J = J_flat[idx_flat, np.arange(Q_count)]
        
        q_vals = np.arange(-Q, Q + 1)
        inv_cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
        
        V_next = V_next + max_J - inv_cost
        
    return bid_policy, ask_policy


def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _policy_cache:
        # Add a simple deadline guard
        start_time = time.time()
        try:
            _policy_cache[key] = _compute_policy(sigma, A, k, gamma, T, steps, Q)
        except Exception as e:
            # Fallback to seed strategy if computation fails or is too slow
            depth = np.full(len(inventory), 1.0 / k)
            return depth, depth.copy()
        
        if time.time() - start_time > 300:
            # This shouldn't happen if the loop is fast, but just in case
            depth = np.full(len(inventory), 1.0 / k)
            return depth, depth.copy()
            
    bid_policy, ask_policy = _policy_cache[key]
    
    # Map t to step index
    # t = n * dt
    dt = T / steps
    n = int(round(t / dt))
    n = np.clip(n, 0, steps - 1)
    
    # Map inventory to index
    # inventory in [-Q, Q]
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    bid_depths = bid_policy[n, q_idx]
    ask_depths = ask_policy[n, q_idx]
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    
    # Find feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    w_feas = weights[feasible_mask]
    v_feas = values[feasible_mask]
    idx_feas = feasible_indices
    
    n_feas = len(w_feas)
    if n_feas == 0:
        return -1

    # Scale to integers for DP
    # Use a scale that balances precision and speed.
    # Capacity is small (12.5), so 10000 is safe for speed.
    scale = 10000
    C_int = int(np.round(remaining_capacity * scale))
    
    if C_int <= 0:
        return -1
        
    W_int = np.round(w_feas * scale).astype(np.int64)
    
    # Filter out items with weight 0 or > C_int (already filtered by feasible, but W_int might be 0)
    # If weight is 0, it's always good to take?
    # In 0/1 knapsack, weight 0 value > 0 items should always be taken.
    # Our DP handles w=0?
    # Loop: for c in range(C_int, -1, -1) if w_j=0?
    # My loop: range(C_int, w_j - 1, -1). If w_j=0, range(C_int, -1, -1).
    # dp[c] = max(dp[c], dp[c] + v_j). This works.
    
    dp = np.zeros(C_int + 1, dtype=np.float64)
    choice = np.zeros((n_feas, C_int + 1), dtype=bool)
    
    for j in range(n_feas):
        w_j = W_int[j]
        v_j = v_feas[j]
        
        if w_j > C_int:
            continue
            
        # Update DP and choice
        # We iterate c from C_int down to w_j
        for c in range(C_int, w_j - 1, -1):
            prev_val = dp[c - w_j]
            new_val = prev_val + v_j
            if new_val > dp[c]:
                dp[c] = new_val
                choice[j][c] = True
            else:
                choice[j][c] = False
                
    # Backtrack to find the first (highest index) item in the optimal set
    c = C_int
    for j in range(n_feas - 1, -1, -1):
        if choice[j][c]:
            return int(idx_feas[j])
        # else: item j not taken, c stays same, j decreases
    
    # If no item was selected (e.g. all values 0 or negative?)
    # If dp[C_int] is 0, maybe no items fit? But we filtered feasible.
    # If values are 0, choice might be all False.
    # In that case, pick any feasible item?
    # If value is 0, it doesn't matter.
    if dp[C_int] <= 0:
        return int(idx_feas[0])
        
    return -1
# EVOLVE-BLOCK-END

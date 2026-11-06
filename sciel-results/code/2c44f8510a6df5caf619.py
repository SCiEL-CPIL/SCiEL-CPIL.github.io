import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Check if any item fits
    if np.min(remaining_weights) > remaining_capacity:
        return -1

    # Scale to integers for DP
    # Use a scale factor to handle decimals like 12.5
    scale = 100
    cap_int = int(remaining_capacity * scale)
    
    # If capacity is very small or zero after scaling, stop
    if cap_int <= 0:
        return -1

    # Scale weights and values
    # Round to nearest integer
    w_int = np.round(np.array(remaining_weights) * scale).astype(int)
    v_int = np.round(np.array(remaining_values) * scale).astype(int)
    
    # Filter out items that are too heavy even after scaling (should be handled by DP, but good for safety)
    # Actually, DP handles this naturally.
    
    # DP for 0/1 Knapsack
    # dp[j] = max value achievable with capacity j
    dp = np.zeros(cap_int + 1, dtype=int)
    
    # We need to track which items are used to backtrack
    # To save memory/time, we can store the choice or just re-run a simpler backtrack
    # Standard DP:
    # for i in range(n):
    #   for j in range(cap_int, w_int[i]-1, -1):
    #      if dp[j - w_int[i]] + v_int[i] > dp[j]:
    #          dp[j] = dp[j - w_int[i]] + v_int[i]
    
    # This is O(N * C).
    # To backtrack, we need to know the decision.
    # We can store a 2D array of booleans or integers, but that's N*C memory.
    # N=100, C=1250. 125,000 integers. Very small.
    
    # Let's use a list of arrays or a 2D numpy array for choices
    # choice[i][j] = 1 if item i was taken for capacity j, 0 otherwise
    # Actually, we can just recompute or store the parent.
    # Simpler: Store the DP table for all items? No, we only need the final path.
    
    # Let's use the standard backtracking with a stored table.
    # Table dimensions: (n+1) x (cap_int+1)
    # This is 101 * 1251 ~ 126k entries. Fine.
    
    dp_table = np.zeros((n + 1, cap_int + 1), dtype=int)
    
    for i in range(1, n + 1):
        wi = w_int[i-1]
        vi = v_int[i-1]
        for j in range(cap_int + 1):
            # Not take item i
            val_not_take = dp_table[i-1][j]
            # Take item i
            if j >= wi:
                val_take = dp_table[i-1][j-wi] + vi
                if val_take > val_not_take:
                    dp_table[i][j] = val_take
                else:
                    dp_table[i][j] = val_not_take
            else:
                dp_table[i][j] = val_not_take

    # Backtrack to find the items
    # Start from i=n, j=cap_int
    i = n
    j = cap_int
    selected_indices = []
    while i > 0:
        if dp_table[i][j] != dp_table[i-1][j]:
            # Item i-1 was taken
            selected_indices.append(i-1)
            j -= w_int[i-1]
        i -= 1
        
    if not selected_indices:
        return -1
    
    # Return the first selected index
    return selected_indices[0]

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking for the entire evaluation run
_START_TIME = time.time()
_TIME_LIMIT = 110.0  # Leave 10s margin for 120s limit

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _START_TIME
    
    # Fallback to greedy if time is running out
    if time.time() - _START_TIME > _TIME_LIMIT:
        if len(remaining_weights) == 0:
            return -1
        best_idx = -1
        best_ratio = -1
        for i in range(len(remaining_weights)):
            if remaining_weights[i] <= remaining_capacity:
                ratio = remaining_values[i] / remaining_weights[i]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i
        return best_idx

    n = len(remaining_weights)
    if n == 0:
        return -1

    # Check if any item fits
    if min(remaining_weights) > remaining_capacity:
        return -1

    # Scale weights to integers
    SCALE = 100
    C = int(remaining_capacity * SCALE)
    
    # If capacity is effectively 0 after scaling
    if C <= 0:
        return -1

    w_int = [int(w * SCALE) for w in remaining_weights]
    v = np.array(remaining_values, dtype=np.float64)

    # Compute Prefix DP
    # pref[i] is an array of length C+1
    # pref[i][c] = max value using items 0..i-1 with capacity c
    pref = []
    curr_dp = np.zeros(C + 1, dtype=np.float64)
    pref.append(curr_dp)

    for i in range(n):
        w = w_int[i]
        val = v[i]
        
        if w > C:
            # Item too heavy, DP doesn't change
            pref.append(curr_dp.copy())
            continue
        
        # Update DP
        # dp[w:] = max(dp[w:], dp[:-w] + val)
        # Ensure we don't go out of bounds
        if w == 0:
             # Edge case: weight 0. Take it if value > 0?
             # If weight is 0, we can take all such items.
             # But 0/1 knapsack usually implies positive weights.
             # If w=0, dp[c] = max(dp[c], dp[c] + val) -> dp[c] += val if val > 0.
             # Let's assume weights > 0. If not, handle separately.
             if val > 0:
                 curr_dp[:] += val
             pref.append(curr_dp.copy())
             continue
             
        # Standard update
        # We need to be careful with the view.
        # Create a copy of the relevant part to be safe? 
        # NumPy RHS evaluation handles it, but let's be explicit.
        # Actually, the vectorized operation `np.maximum(dp[w:], dp[:-w] + val)` 
        # reads dp[:-w] and dp[w:] and writes to dp[w:].
        # As established, this is safe for 0/1 knapsack.
        
        # Optimization: Only update if val is positive? 
        # If val is negative, we never pick it.
        if val > 0:
            # dp[w:] = np.maximum(dp[w:], dp[:-w] + val)
            # This modifies curr_dp in place.
            curr_dp[w:] = np.maximum(curr_dp[w:], curr_dp[:-w] + val)
        
        pref.append(curr_dp.copy())

    # Compute Suffix DP
    # suff[i] is an array of length C+1
    # suff[i][c] = max value using items i..n-1 with capacity c
    suff = []
    # Initialize from the end
    curr_dp = np.zeros(C + 1, dtype=np.float64)
    suff.insert(0, curr_dp) # suff[n]
    
    # We need suff[i] for i from n down to 0.
    # Let's build it in reverse and then reverse the list or access correctly.
    # suff_list[n] is zeros.
    # suff_list[i] is computed from suff_list[i+1].
    
    suff_list = [None] * (n + 1)
    suff_list[n] = np.zeros(C + 1, dtype=np.float64)
    
    curr_dp = suff_list[n].copy()
    
    for i in range(n - 1, -1, -1):
        w = w_int[i]
        val = v[i]
        
        if w > C:
            suff_list[i] = curr_dp.copy()
            continue
        
        if w == 0:
            if val > 0:
                curr_dp[:] += val
            suff_list[i] = curr_dp.copy()
            continue

        if val > 0:
            curr_dp[w:] = np.maximum(curr_dp[w:], curr_dp[:-w] + val)
        
        suff_list[i] = curr_dp.copy()

    # Now evaluate each item
    best_idx = -1
    best_total_val = -np.inf

    for i in range(n):
        w = w_int[i]
        if w > C:
            continue
            
        C_rem = C - w
        if C_rem < 0:
            continue
            
        # Get prefix and suffix arrays
        p = pref[i]
        s = suff_list[i + 1]
        
        # Calculate max value of sub-problem
        # max_{c=0..C_rem} (p[c] + s[C_rem - c])
        
        # p_part = p[0:C_rem+1]
        # s_part = s[C_rem::-1]  # s[C_rem], s[C_rem-1], ..., s[0]
        
        # We can use numpy to compute this efficiently
        # sums = p[0:C_rem+1] + s[C_rem::-1]
        # sub_val = np.max(sums)
        
        # To avoid creating large temporary arrays if C is large (but it's small here)
        # We can just do it.
        
        sub_val = np.max(p[:C_rem+1] + s[C_rem::-1])
        
        total_val = v[i] + sub_val
        
        if total_val > best_total_val:
            best_total_val = total_val
            best_idx = i

    return best_idx
# EVOLVE-BLOCK-END

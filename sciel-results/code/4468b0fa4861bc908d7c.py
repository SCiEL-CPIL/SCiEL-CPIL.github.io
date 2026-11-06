# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

def construct():
    deadline = time.time() + 50.0  # safety margin before 90s kill
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = [d for d in range(1, n+1) if n % d == 0]
    # Sort in descending order to prefer large divisors (helps find solutions faster)
    divisors.sort(reverse=True)
    
    num_div = len(divisors)
    
    # DP: dp[s] = minimum number of distinct divisors to achieve sum s
    # Initialize with infinity
    INF = float('inf')
    dp = np.full(target + 1, INF, dtype=np.float64)
    dp[0] = 0
    # parent[s] = (prev_sum, divisor_used) to reconstruct solution
    parent = [None] * (target + 1)
    
    # For each divisor, update dp in reverse to avoid using same divisor twice
    for d in divisors:
        if d > target:
            continue
        
        # Vectorized update:
        # We want to update dp[d:target+1] based on dp[0:target+1-d]
        # new_dp[d:s] = min(dp[d:s], dp[0:s-d] + 1)
        # Since we are doing 0/1 knapsack, we must use the state *before* this divisor was considered.
        # The standard iterative loop does this naturally by going backwards.
        # With NumPy, we can compute the candidate values and use np.minimum.
        
        # candidate values for indices d..target
        # dp[0:target+1-d] + 1
        candidates = dp[:target + 1 - d] + 1
        
        # We need to update dp[d:target+1]
        # We only want to update if candidate < current dp
        # Also, we need to track the parent for reconstruction.
        # The vectorized min is fast, but updating parent array element-wise is still needed if we want to reconstruct.
        # However, we can just store the boolean mask of updates and then loop only over those?
        # Or, since target is small, we can just do the vectorized min for dp, and then 
        # reconstruct parent in a second pass or just do the loop for parent if needed.
        
        # Actually, the most efficient way to keep the logic simple and fast:
        # 1. Compute new_dp values.
        # 2. Identify indices where improvement happens.
        # 3. Update dp and parent for those indices.
        
        current_vals = dp[d:target + 1]
        # We need to be careful: dp[:target+1-d] is the "previous" state for the corresponding indices?
        # Yes, because we are iterating d. The values dp[0...target-d] have NOT been updated by divisor d yet 
        # (assuming we process divisors one by one). 
        # Wait, if I use `dp` on the RHS, it refers to the array *before* this divisor is fully processed?
        # No, `dp` is being modified in place. 
        # In the standard loop:
        # for s in range(target, d-1, -1):
        #   if dp[s-d] + 1 < dp[s]: ...
        # Here dp[s-d] is from the *previous* state (before d was added) because s-d < s and we go downwards.
        # So dp[s-d] has not been updated by d yet.
        
        # In NumPy:
        # src = dp[:target+1-d]  # These are values for sums 0..target-d
        # dst = dp[d:target+1]   # These are values for sums d..target
        # We want to update dst with min(dst, src + 1)
        
        # But `src` and `dst` are views of the same array. 
        # If we do `np.minimum(dst, src+1, out=dst)`, does it read `src` correctly?
        # `src` is indices 0..T-d. `dst` is indices d..T.
        # They overlap if d < T-d, i.e., 2d < T.
        # If they overlap, `src` contains values that might be in `dst`?
        # No, `src` is lower indices, `dst` is higher indices.
        # The values in `src` (indices 0..T-d) are NOT modified by the update to `dst` (indices d..T) 
        # UNLESS the update to `dst` writes to an index that is also in `src`.
        # Indices in `dst`: d, d+1, ..., T.
        # Indices in `src`: 0, 1, ..., T-d.
        # Overlap region: indices from d to T-d.
        # When updating `dst[i]` (which is `dp[d+i]`), we read `src[i]` (which is `dp[i]`).
        # Does the update to `dp[d+i]` affect `dp[i]`? No, `dp[i]` is not in `dst` (unless i >= d, which it isn't for i < d).
        # Wait, if i is in the overlap, e.g., i=10, d=5, T=100.
        # dst index 10 corresponds to dp[15]. src index 10 corresponds to dp[10].
        # We update dp[15] based on dp[10].
        # Does updating dp[15] change dp[10]? No.
        # So the read of `src` is safe from the writes to `dst`?
        # Yes, because `src` indices are always smaller than `dst` indices (offset by d).
        # Specifically, `src[j]` is `dp[j]`. `dst[j]` is `dp[j+d]`.
        # We write to `dp[j+d]`. We read from `dp[j]`.
        # Since `j < j+d`, the write happens at a higher index.
        # The loop in Python goes `s` from high to low.
        # Here, the vectorized operation is "parallel".
        # Does the parallel update cause issues?
        # If I update `dp[10]` and `dp[15]` simultaneously?
        # `dp[15]` depends on `dp[10]`.
        # If `dp[10]` is also being updated in the same vectorized step?
        # `dp[10]` is in `dst` only if `10 >= d`.
        # If `d <= 10`, then `dp[10]` is in `dst`.
        # `dp[10]` in `dst` is updated based on `dp[10-d]`.
        # So `dp[10]` changes.
        # But `dp[15]` (in `dst`) is updated based on `dp[10]` (in `src`).
        # If `dp[10]` is updated *before* `dp[15]` reads it, we might use the new value (which includes divisor d).
        # This would violate the "distinct divisors" constraint (using d twice).
        
        # Therefore, we CANNOT do a simple in-place vectorized update if the ranges overlap.
        # We must use a temporary array for the source or copy the relevant slice.
        
        src_copy = dp[:target + 1 - d].copy()
        candidates = src_copy + 1
        
        # Now we can safely update dp[d:target+1]
        # We need to find where candidates < dp[d:target+1]
        dst_view = dp[d:target + 1]
        mask = candidates < dst_view
        
        if np.any(mask):
            # Update dp
            dst_view[mask] = candidates[mask]
            
            # Update parent
            # Indices in original dp: d + j
            # Mask indices j
            # We need to iterate over the set bits in mask to update parent
            # Since target is small, this loop is fast.
            indices = np.where(mask)[0]
            for j in indices:
                s = d + j
                parent[s] = (j, d) # prev_sum is j, divisor is d
                
    if dp[target] == INF:
        # Fallback: return empty or raise
        return []
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s
    
    return chosen
# EVOLVE-BLOCK-END

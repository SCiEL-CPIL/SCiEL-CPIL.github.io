"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_range(n):
    return reduce(_lcm, range(1, n + 1), 1)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin: kill at 90s, use 50s for DP
    
    n = 30
    L = _lcm_range(n)
    weights = [L // (i + 1) for i in range(n)]
    
    # Trivial fallback: 1/29 - 1/30 = 1/870
    # Coeffs: 0...0, 1, -1 at indices 28, 29
    best_coeffs = [0] * 28 + [1, -1]
    best_N = weights[28] - weights[29]
    best_abs_N = abs(best_N)
    
    # Split the problem into Core (large weights) and Tail (small weights)
    # Weights are decreasing: w[0] > w[1] > ... > w[29]
    # Let's split at index 15.
    # Core: indices 0..14 (15 variables)
    # Tail: indices 15..29 (15 variables)
    
    split_idx = 15
    core_weights = weights[:split_idx]
    tail_weights = weights[split_idx:]
    
    # Generate all possible sums for the Core
    # 3^15 = 14,348,907 states. This is feasible.
    # We store the set of all possible core sums.
    # To save memory/time, we can just store the set of sums.
    # The sums are integers.
    
    core_sums = set()
    # Start with sum 0 (all coeffs 0)
    core_sums.add(0)
    
    # Iteratively build the set of sums
    # For each weight in core_weights, update the set
    # new_sums = {s + c*w for s in old_sums for c in {-1, 0, 1}}
    # This is O(3^k * K) which is fine.
    
    for w in core_weights:
        new_sums = set()
        for s in core_sums:
            new_sums.add(s - w)
            new_sums.add(s)
            new_sums.add(s + w)
        core_sums = new_sums
        
        # Check time
        if time.time() > deadline - 5:
            # If we run out of time, we can't complete the DP.
            # Fallback to the trivial solution or a quick random search.
            # But 3^15 should be fast enough.
            break

    if time.time() > deadline - 5:
        # Failed to complete DP, return best found so far (trivial)
        return best_coeffs

    # Now, iterate over all possible Tail configurations
    # Tail has 15 variables, so 3^15 = 14,348,907 combinations.
    # For each tail sum T, we want to find a core sum C in core_sums
    # such that |C + T| is minimized and non-zero.
    
    # To make this fast, we can sort core_sums.
    # Then for a given T, we want to find C in core_sums such that C is close to -T.
    # We can use binary search (bisect) to find the candidate C.
    
    import bisect
    sorted_core_sums = sorted(core_sums)
    
    # We need to map back to coefficients?
    # The task requires returning the coefficients.
    # The DP set only stores sums, not the path (coefficients).
    # To recover coefficients, we would need to store the path or backtrack.
    # Storing paths for 14M states is memory heavy (14M * 15 ints).
    # Alternative:
    # 1. Find the best (C, T) pair using the sets.
    # 2. Once we have the target C and T, we can reconstruct the coefficients
    #    by solving the subset sum for C using core_weights and T using tail_weights.
    #    Since we know the target sum, we can do a recursive search or another DP
    #    specifically for that target, which is much faster than full enumeration.
    
    best_total_N = best_N
    best_C = None
    best_T = None
    
    # Generate all tail sums
    # We can do this iteratively as well, or just loop through 3^15.
    # Iterative is better.
    
    tail_sums = {0}
    for w in tail_weights:
        new_sums = set()
        for s in tail_sums:
            new_sums.add(s - w)
            new_sums.add(s)
            new_sums.add(s + w)
        tail_sums = new_sums
        if time.time() > deadline - 5:
            break

    if time.time() > deadline - 5:
        return best_coeffs

    # Now we have sets of Core Sums and Tail Sums.
    # We want to find C in core_sums and T in tail_sums
    # minimizing |C + T| subject to C + T != 0.
    
    # Since tail_sums is also a set of ~14M elements, iterating over all pairs is too slow.
    # But we can iterate over tail_sums and for each T, search in sorted_core_sums.
    
    # Optimization: 
    # The range of C is roughly [-sum(core_w), sum(core_w)].
    # The range of T is roughly [-sum(tail_w), sum(tail_w)].
    # sum(core_w) is much larger than sum(tail_w).
    # So for a given T, the optimal C is near -T.
    
    # We iterate over tail_sums. 14M iterations.
    # Each iteration does a bisect (log(14M) ~ 24 ops).
    # Total ~ 14M * 24 = 336M ops. This might be too slow in Python (60s limit).
    # We need to be smarter or limit the search.
    
    # Alternative: 
    # The smallest non-zero sum is likely to be small.
    # We can check for small values of K = 1, 2, 3...
    # Does there exist C in core_sums and T in tail_sums such that C + T = K?
    # This is equivalent to: does (K - T) exist in core_sums for some T in tail_sums?
    # Or: does (K - C) exist in tail_sums for some C in core_sums?
    
    # Let's try checking small K values first.
    # If we find a K, we can reconstruct.
    
    # To speed up the check for a specific K:
    # We can iterate over the smaller set (tail_sums) and check if K-T is in core_sums (O(1) lookup).
    
    # How large can K be?
    # The trivial solution has |N| = L/870 ~ 2.6e12.
    # We expect to find much smaller.
    # Let's check K up to, say, 10000?
    # But if the minimal K is large, we miss it.
    
    # Better approach:
    # Iterate over tail_sums (14M).
    # For each T, we want to find C in core_sums such that C is close to -T.
    # We can use bisect on sorted_core_sums.
    # To make this fit in time, we can sample or limit the tail_sums iteration?
    # No, we need completeness.
    
    # Let's optimize the Python loop.
    # Using a list for sorted_core_sums and bisect is fast.
    # The bottleneck is the 14M loop.
    # We can try to do this in C via numpy? No, sets are not numpy friendly.
    
    # Let's try to reduce the size of the sets.
    # Maybe split 14/16?
    # 3^14 = 4.7M. 3^16 = 43M.
    # If we use Core 14, Tail 16.
    # Core sums: 4.7M.
    # Tail sums: 43M.
    # Iterating over Tail (43M) and bisecting Core (log 4.7M ~ 22) -> 43M * 22 ~ 1B ops. Too slow.
    
    # If we use Core 15, Tail 15.
    # Both 14M.
    # Iterate over Tail (14M), bisect Core (log 14M ~ 24).
    # 14M * 24 ~ 336M ops.
    # In Python, a loop of 14M takes ~1-2 seconds if simple.
    # Bisect is C-coded, so it's fast.
    # The overhead is the Python loop and set lookup.
    # Let's try it. If it's too slow, we fall back.
    
    # Actually, we can do better.
    # We don't need to check ALL tail sums if we just want the MINIMUM.
    # But we don't know the minimum.
    
    # Let's try the "Check small K" approach first, as it's much faster if a small solution exists.
    # If no small K is found, we do the full search.
    
    # Check K from 1 to 1000
    # For each K, iterate tail_sums and check if K-T in core_sums.
    # 1000 * 14M = 14B ops. Too slow.
    
    # So we must do the "Find closest" approach.
    
    # Let's implement the closest search efficiently.
    
    # We will iterate over tail_sums.
    # To speed up, we can convert core_sums to a list and use bisect.
    
    # Wait, if we just want the best C+T, we can do:
    # For each T in tail_sums:
    #   target = -T
    #   Find C in core_sums closest to target.
    #   Check C and C_next.
    #   Calculate |C+T| and |C_next+T|.
    #   Update best.
    
    # This is O(3^15 * log(3^15)).
    
    # Let's run it.
    
    found_solution = False
    best_K_val = best_abs_N
    best_C_found = None
    best_T_found = None
    
    # To avoid iterating 14M times if we find a very small K early,
    # we can't easily break because a later T might yield a smaller K.
    # But if we find K=1, we can stop immediately.
    
    for T in tail_sums:
        target = -T
        # Find insertion point in sorted_core_sums
        pos = bisect.bisect_left(sorted_core_sums, target)
        
        # Check pos and pos-1
        candidates = []
        if pos < len(sorted_core_sums):
            candidates.append(sorted_core_sums[pos])
        if pos > 0:
            candidates.append(sorted_core_sums[pos-1])
            
        for C in candidates:
            total = C + T
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_K_val:
                best_K_val = abs_total
                best_C_found = C
                best_T_found = T
                if best_K_val == 1:
                    # Can't get better than 1
                    found_solution = True
                    break
        if found_solution:
            break
            
        # Time check
        if time.time() > deadline - 5:
            break

    if best_K_val < best_abs_N:
        # We found a better N.
        # Now we need to reconstruct the coefficients for C and T.
        
        # Function to find coefficients for a target sum given weights
        def find_coeffs(weights, target):
            # Recursive search with memoization or backtracking
            # Since weights are distinct and large, backtracking should be fast
            # if we prune.
            # n=15, 3^15 is small enough for a simple DFS if we prune.
            # Actually, 3^15 is 14M. A DFS visiting 14M nodes is okay.
            
            coeffs = [0] * len(weights)
            
            def dfs(idx, current_sum):
                if idx == len(weights):
                    if current_sum == target:
                        return True
                    return False
                
                # Pruning: if current_sum + sum(remaining_weights) < target, return False
                # if current_sum - sum(remaining_weights) > target, return False
                # Precompute suffix sums for pruning.
                
                # Try -1, 0, 1
                for val in (-1, 0, 1):
                    coeffs[idx] = val
                    if dfs(idx + 1, current_sum + val * weights[idx]):
                        return True
                return False
            
            # To make DFS faster, we can use the fact that we know a solution exists
            # (because C was in the set generated by these weights).
            
            # Let's use a simple iterative check or a smarter DP for reconstruction.
            # Since we already did the DP to generate the set, we could have stored the path.
            # But we didn't.
            # Let's just run the DFS. 14M nodes max.
            
            # Optimization: Order weights by size (descending) to prune faster.
            # But we need to return coeffs in original order.
            # So we map back.
            
            # For simplicity and safety, let's just use the set membership check
            # to guide a greedy or backtracking search.
            
            # Actually, since we have the set of ALL possible sums, we can backtrack
            # by checking if (target - val*w[i]) is in the set of sums for remaining weights.
            
            # Let's build the sets for suffixes.
            # This is expensive to build again.
            
            # Alternative: Just use the fact that we have the full set of core_sums.
            # We can reconstruct by walking back through the weights.
            
            # Let's just use a simple recursive search with pruning.
            # It should be fast enough for n=15.
            
            # To ensure speed, we'll use a list of remaining sums? No.
            
            # Let's try a different reconstruction:
            # We know C is achievable.
            # We can try to match C by subtracting weights one by one.
            
            # Let's use the "Check if target - w[i]*val is achievable by remaining" logic.
            # We need the set of achievable sums for the suffix.
            
            # We can generate these sets on the fly in reverse?
            
            # Let's just use the original `core_sums` set and work backwards?
            # No, `core_sums` is the set of ALL sums.
            
            # Let's use a simple DFS. 15 levels, branching 3.
            # With pruning based on range, it should be very fast.
            
            suffix_sums = [0] * (len(weights) + 1)
            for i in range(len(weights) - 1, -1, -1):
                suffix_sums[i] = suffix_sums[i+1] + abs(weights[i])
            
            def dfs_recon(idx, current_sum):
                if idx == len(weights):
                    return current_sum == target
                if current_sum + suffix_sums[idx] < target:
                    return False
                if current_sum - suffix_sums[idx] > target:
                    return False
                    
                for val in (-1, 0, 1):
                    coeffs[idx] = val
                    if dfs_recon(idx + 1, current_sum + val * weights[idx]):
                        return True
                return False
            
            if dfs_recon(0, 0):
                return coeffs
            return None

        core_coeffs = find_coeffs(core_weights, best_C_found)
        tail_coeffs = find_coeffs(tail_weights, best_T_found)
        
        if core_coeffs and tail_coeffs:
            new_coeffs = core_coeffs + tail_coeffs
            # Verify
            check_N = sum(c * w for c, w in zip(new_coeffs, weights))
            if check_N == best_C_found + best_T_found:
                best_coeffs = new_coeffs
                best_N = check_N
                best_abs_N = abs(best_N)

    # Convert best_N to (num, den)
    num = best_N
    den = L
    g = gcd(abs(num), den)
    if g:
        num //= g
        den //= g
    
    if den < 0:
        num = -num
        den = -den
    
    if num == 0:
        return [0] * 28 + [1, -1]
        
    return best_coeffs

# EVOLVE-BLOCK-END

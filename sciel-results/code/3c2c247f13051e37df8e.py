"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math
    import itertools
    
    # Denominators are 2..48, indices 0..46
    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Compute LCM of all denominators 2..48
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in denoms:
        L = lcm(L, d)
    
    # Convert each 1/d to integer numerator over L
    # val[i] = L // denoms[i]
    vals = [L // d for d in denoms]
    
    # We want a subset of indices such that sum(vals[i]) < L, minimizing L - sum(vals[i])
    # i.e., maximize sum(vals[i]) subject to sum < L
    # This is a subset sum problem with target L-1 (max sum <= L-1)
    
    # Meet-in-the-middle: split 47 items into two halves
    # First half: 0..23 (24 items), second half: 24..46 (23 items)
    # 2^24 = 16,777,216 which might be too large in memory/time
    # Let's split more evenly: 23 and 24? Or 24 and 23?
    # Actually 47 items: split into 23 and 24.
    # 2^23 = 8,388,608 subset sums per half. That's manageable.
    
    half1_size = 23
    half1 = vals[:half1_size]
    half2 = vals[half1_size:]
    
    # Generate all subset sums for half1
    # Use iterative approach
    sums1 = [0]
    for v in half1:
        sums1 += [s + v for s in sums1]
    
    # Generate all subset sums for half2
    sums2 = [0]
    for v in half2:
        sums2 += [s + v for s in sums2]
    
    # Sort sums2 for binary search
    sums2.sort()
    
    # For each sum in sums1, find the largest sum in sums2 such that sum1 + sum2 < L
    # i.e., sum2 < L - sum1
    best_total = 0
    best_sum1 = 0
    best_sum2 = 0
    
    import bisect
    
    for s1 in sums1:
        if s1 >= L:
            continue
        # Need s2 < L - s1
        limit = L - s1
        # Find largest s2 in sums2 with s2 < limit
        # bisect_left gives first index where sums2[idx] >= limit
        idx = bisect.bisect_left(sums2, limit)
        if idx > 0:
            s2 = sums2[idx - 1]
            total = s1 + s2
            if total > best_total:
                best_total = total
                best_sum1 = s1
                best_sum2 = s2
    
    # Now we need to recover which indices were selected
    # Reconstruct subset for half1 that gives best_sum1
    # And subset for half2 that gives best_sum2
    
    def find_subset(values, target):
        """Find a subset of values that sums to target. Returns list of indices."""
        n = len(values)
        # Use DP or meet-in-the-middle again? For 23 items, we can do it directly.
        # Actually, let's use a simple approach: generate all subsets and check
        # But 2^23 is 8M which is fine.
        
        # Better: use the same sums generation but track indices
        # Let's do a DFS/backtracking approach with memoization or just iterate
        
        # Simple approach: since we know the target, use DP
        # dp[s] = True if s is achievable, and store parent info
        # But sums can be very large (up to L ~ 10^18), so DP on sum won't work
        
        # Instead, use meet-in-the-middle within each half? No, halves are already 23 items.
        # Let's just do a direct search: for 23 items, 2^23 = 8M, we can iterate.
        
        # Actually, let's use a different approach: generate all subset sums with their index sets
        # But that's memory heavy. Let's use a recursive approach with pruning.
        
        # Alternative: since we have the list of values and target, use a greedy-ish approach
        # with backtracking. For 23 items, this should be fast enough.
        
        result = []
        remaining = target
        for i in range(n - 1, -1, -1):
            if values[i] <= remaining:
                result.append(i)
                remaining -= values[i]
            if remaining == 0:
                break
        
        # This greedy might not work. Let's do proper backtracking.
        
        def backtrack(start, current_sum, current_indices):
            if current_sum == target:
                return True
            if start == n or current_sum > target:
                return False
            # Try including values[start]
            if current_sum + values[start] <= target:
                if backtrack(start + 1, current_sum + values[start], current_indices + [start]):
                    return True
            # Try excluding
            if backtrack(start + 1, current_sum, current_indices):
                return True
            return False
        
        # This might be slow. Let's use a better approach.
        # Since we generated sums1 from half1, we can regenerate with index tracking.
        
        # Let's just do a simple iterative approach: for each subset, check if sum matches
        # For 23 items, 8M subsets is fine.
        
        for mask in range(1 << n):
            s = 0
            indices = []
            for i in range(n):
                if mask & (1 << i):
                    s += values[i]
                    indices.append(i)
                    if s > target:
                        break
            if s == target:
                return indices
        
        return []
    
    # Find subset for half1
    indices1 = find_subset(half1, best_sum1)
    # Find subset for half2
    indices2 = find_subset(half2, best_sum2)
    
    # Combine indices (offset half2 indices by half1_size)
    selected = indices1 + [i + half1_size for i in indices2]
    
    # Verify
    total_frac = sum(Fraction(1, denoms[i]) for i in selected)
    assert total_frac < 1, f"Sum {total_frac} is not < 1"
    assert len(selected) > 0, "Empty selection"
    
    return selected
# EVOLVE-BLOCK-END

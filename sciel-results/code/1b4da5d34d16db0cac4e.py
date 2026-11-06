import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    N = 479001600  # 12!
    # Prime factorization of N: 2^8 * 3^5 * 5^2 * 7^2 * 11^1
    # We represent a divisor by its exponent vector (e2, e3, e5, e7, e11)
    # where 0 <= e2 <= 8, 0 <= e3 <= 5, 0 <= e5 <= 2, 0 <= e7 <= 2, 0 <= e11 <= 1
    
    # 1. Generate all valid divisors (value > 12) with their exponent vectors
    divs = []
    for e2 in range(9):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(3):
                    for e11 in range(2):
                        val = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if val > 12:
                            divs.append((val, (e2, e3, e5, e7, e11)))
    
    divs.sort(key=lambda x: x[0])
    n_divs = len(divs)
    
    # 2. Binary Search for the minimum M
    # Low bound: 13
    # High bound: N
    low = 13
    high = N
    
    best_M = N
    best_factors = [N]
    found = False
    
    # Pre-compute suffix products of largest divisors for pruning?
    # Or just use a simple bound: if target > (max_val)^(12-count), fail.
    # Actually, since divisors are distinct, the max product of k distinct divisors from a set S
    # is the product of the k largest elements in S.
    # Let's precompute the sorted list of values.
    vals = [d[0] for d in divs]
    
    # For pruning, we need to know if it's possible to form 'target' using distinct divisors
    # from the available set (indices start_idx to end_idx).
    # A necessary condition: target <= product of the largest (12-count) divisors in [start_idx, end_idx].
    # Also, target must be divisible by the product of some subset.
    
    # Let's use a DFS with memoization on (target, start_idx, count).
    # However, 'target' can be large.
    # The state space is determined by the remaining prime exponents.
    # Initial exponents: (8, 5, 2, 2, 1)
    # A state can be represented by the tuple of remaining exponents.
    # This is much smaller than 'target' value space.
    # Let's map (target, start_idx, count) to (rem_exponents, start_idx, count).
    # Actually, 'target' determines the remaining exponents uniquely if we started from N.
    # So we can just use 'target' as the key for exponents, but we also need start_idx to ensure distinctness.
    
    # Let's refine the DFS:
    # dfs(target, start_idx, count)
    # Returns a list of factors or None.
    # To avoid recomputation, we can cache (target, start_idx) -> success/failure?
    # But we need the path.
    # We can cache (target, start_idx) -> bool (success) to prune.
    # If we know it's possible, we still need to find the path.
    # However, if we iterate candidates in a deterministic order (e.g., ascending value),
    # the first path found is deterministic.
    # So we can cache (target, start_idx) -> bool.
    
    # Let's implement the binary search loop.
    
    # Optimization: Instead of binary searching over [13, N], which is huge,
    # we can binary search over the sorted list of possible max values.
    # The possible max values are just the divisors themselves.
    # Let's binary search on the index in `vals`.
    
    # We want the smallest M such that can_decompose(M) is True.
    # M must be one of the divisors? Yes, the largest factor is one of the factors, so it's a divisor.
    
    # Let's binary search on the index of M in `vals`.
    # Low index: 0 (val=13)
    # High index: n_divs - 1 (val=N)
    
    # Wait, M doesn't have to be the largest divisor in the whole set, just the bound.
    # But the optimal M will be equal to one of the factors, hence a divisor.
    
    # Let's define can_decompose(idx_max) where idx_max is the max index allowed in divs.
    # This means all factors must be from divs[0:idx_max+1].
    
    # We'll binary search on idx_max.
    
    # Helper to check if a solution exists using divs[0:idx_max+1]
    # We use DFS with memoization.
    
    # To make it fast, we should try to find a solution quickly.
    # Order of trying candidates:
    # If we want to minimize the largest factor, we are already bounding it.
    # Inside DFS, does the order matter?
    # We want to find *any* solution.
    # Trying larger factors first might reduce the depth (fewer factors needed).
    # Trying smaller factors first might lead to deeper trees.
    # Given the constraint "distinct", and we want to minimize the MAX,
    # the binary search handles the max.
    # Inside the search for a fixed max, finding any valid set is the goal.
    # Let's try iterating candidates in descending order of value (largest first).
    # Why? Because we want to consume the 'target' quickly.
    # Also, larger factors have fewer divisors generally? No.
    # But picking a large factor reduces the remaining target significantly.
    
    # Let's stick to the structure but optimize the DFS.
    
    # Memoization key: (target, start_idx)
    # Note: 'count' is not strictly needed for the success/failure memo if we handle the base case correctly,
    # BUT the limit of 12 factors depends on count.
    # However, if we pick factors, count increases.
    # The state (target, start_idx) implies a specific set of available numbers.
    # The number of factors picked so far is not in the key.
    # We MUST include count in the key, or ensure that the path length is bounded.
    # Actually, if we reach target=1, we check if count <= 12.
    # If we don't include count, we might reuse a memo entry where count was 11,
    # but now count is 12, and adding one more would be 13 (invalid).
    # So key must be (target, start_idx, count).
    # Or, since count is small (<=12), and target decreases, the state space is manageable.
    
    # Let's implement can_decompose(idx_max):
    
    def can_decompose(idx_max):
        # Available divisors are divs[0 : idx_max + 1]
        # We want to pick distinct indices from this range.
        # To ensure distinctness, we pick indices in increasing order.
        # So in DFS, we iterate i from start_idx to idx_max.
        
        memo = {}
        
        def dfs(target, start_idx, count):
            # Pruning 1: Time
            if time.time() > deadline:
                return None
            
            # Pruning 2: Target reached
            if target == 1:
                if count <= 12:
                    return [] # Empty list of remaining factors (path is built by caller)
                else:
                    return None # Too many factors
            
            # Pruning 3: No more candidates or too many factors
            if count >= 12:
                return None
            if start_idx > idx_max:
                return None
            
            # Pruning 4: Max possible product
            # The largest product we can form with (12 - count) distinct factors
            # from divs[start_idx : idx_max + 1] is the product of the largest (12-count) values.
            # Let k = 12 - count.
            # If k > idx_max - start_idx + 1, impossible.
            k = 12 - count
            available_count = idx_max - start_idx + 1
            if k > available_count:
                return None
            
            # Calculate max possible product
            # We take the largest k values from divs[start_idx : idx_max + 1]
            # Since divs is sorted ascending, these are at the end of the slice.
            # indices: idx_max, idx_max-1, ..., idx_max-k+1
            # But we must ensure these indices are >= start_idx.
            # Since k <= available_count, the smallest index is idx_max - k + 1 >= start_idx.
            
            # Optimization: Precompute suffix products?
            # For a fixed idx_max, the set of available elements changes with start_idx.
            # It's hard to precompute for all start_idx efficiently without O(N^2) space/time.
            # Let's just compute it on the fly. k is small (<=12).
            max_prod = 1
            for j in range(k):
                idx = idx_max - j
                if idx < start_idx:
                    break
                max_prod *= divs[idx][0]
                if max_prod > target:
                    break
            
            if max_prod < target:
                return None
            
            # Pruning 5: Memoization
            key = (target, start_idx, count)
            if key in memo:
                return memo[key]
            
            # Search
            # Iterate i from start_idx to idx_max
            # We want to find a divisor d of target such that d is in divs[i]
            # and dfs(target//d, i+1, count+1) succeeds.
            
            # To find a solution quickly, should we go ascending or descending i?
            # If we go descending (largest factors first), we reduce target faster.
            # Let's try descending.
            for i in range(idx_max, start_idx - 1, -1):
                d_val, d_vec = divs[i]
                
                # If d_val > target, it cannot divide target (unless target is 0, which it isn't).
                # Since we are going descending, if d_val > target, all smaller i will have smaller d_val.
                # Wait, divs is sorted ascending. So i=idx_max is the largest.
                # If divs[idx_max][0] > target, we still might find a divisor smaller than target.
                # We cannot break immediately. We must check d_val % target == 0? No, target % d_val == 0.
                # If d_val > target, target % d_val != 0 (for target > 0).
                # So we can skip all d_val > target?
                # No, we iterate i from idx_max down to start_idx.
                # If d_val > target, we just skip it.
                # But since divs is sorted, once d_val <= target, all subsequent (smaller i) will also be <= target?
                # No, smaller i means smaller value. So if d_val <= target, all j < i have d_val_j <= d_val <= target.
                # So if we start from the top and d_val > target, we just keep going down until d_val <= target.
                
                if d_val > target:
                    continue
                    
                if target % d_val == 0:
                    # Check if d_val is a valid divisor of target in terms of prime exponents?
                    # Yes, if target % d_val == 0, it is a divisor.
                    
                    path = dfs(target // d_val, i + 1, count + 1)
                    if path is not None:
                        res = [d_val] + path
                        memo[key] = res
                        return res
            
            memo[key] = None
            return None
        
        return dfs(N, 0, 0)

    # Binary search on idx_max
    # We want the smallest idx_max such that can_decompose(idx_max) is not None.
    # Note: idx_max corresponds to the largest factor allowed.
    # The value of the largest factor will be divs[idx_max][0] if it's used, or smaller if not used but bounded by it.
    # Actually, if we allow up to idx_max, the largest factor in the solution is <= divs[idx_max][0].
    # We want to minimize this largest factor.
    
    low_idx = 0
    high_idx = n_divs - 1
    
    best_idx = n_divs - 1
    best_path = None
    found = False
    
    while low_idx <= high_idx:
        if time.time() > deadline - 5:
            break
            
        mid_idx = (low_idx + high_idx) // 2
        
        # Try to find a solution with max index mid_idx
        path = can_decompose(mid_idx)
        
        if path is not None:
            # Found a solution. The largest factor is max(path).
            # But we want to minimize the largest factor.
            # The bound is divs[mid_idx][0].
            # It's possible the actual max in path is smaller than divs[mid_idx][0].
            # But if a solution exists with bound M, it also exists with bound M' > M.
            # We are searching for the smallest M.
            # So if can_decompose(mid_idx) is True, the optimal M is <= divs[mid_idx][0].
            # We record this solution and search for a smaller M.
            best_idx = mid_idx
            best_path = path
            found = True
            high_idx = mid_idx - 1
        else:
            low_idx = mid_idx + 1
    
    if found and best_path:
        return best_path
    else:
        # Fallback
        return [N]

# Quick sanity check in head:
# N = 479001600
# Divisors generated.
# Binary search on index.
# DFS with memoization.
# Should be fast enough.

# One potential issue: The memoization key (target, start_idx, count).
# target can be large, but the number of distinct targets reachable is limited by the divisors of N.
# start_idx is up to ~900.
# count is up to 12.
# Total states ~ (Num Divisors of N) * 900 * 12.
# Num divisors of N is 9*6*3*3*2 = 972.
# 972 * 900 * 12 ~ 10^7 states.
# This might be too slow for Python if we visit many of them.
# However, the DFS prunes heavily.
# Also, we only run this for O(log N) values of mid_idx.
# And the memo is cleared for each mid_idx.
# If the search space is too big, it might time out.

# Let's add a more aggressive pruning:
# If target == 1, return [].
# If count == 12, return None (unless target == 1 handled above).
# Also, if the smallest possible product of (12-count) distinct divisors from [start_idx, idx_max] is > target, fail?
# No, we need product == target.
# If min_prod > target, fail.
# Min prod is product of smallest (12-count) divisors in range.
# indices: start_idx, start_idx+1, ...
# Let's add this check.

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    N = 479001600
    
    divs = []
    for e2 in range(9):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(3):
                    for e11 in range(2):
                        val = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if val > 12:
                            divs.append((val, (e2, e3, e5, e7, e11)))
    
    divs.sort(key=lambda x: x[0])
    n_divs = len(divs)
    vals = [d[0] for d in divs]
    
    def can_decompose(idx_max):
        memo = {}
        
        def dfs(target, start_idx, count):
            if time.time() > deadline:
                return None
            
            if target == 1:
                return [] if count <= 12 else None
            
            if count >= 12 or start_idx > idx_max:
                return None
            
            k = 12 - count
            available = idx_max - start_idx + 1
            if k > available:
                return None
            
            # Pruning: Min and Max product
            # Max product: largest k values in [start_idx, idx_max]
            max_prod = 1
            for j in range(k):
                idx = idx_max - j
                max_prod *= vals[idx]
                if max_prod > target:
                    break
            if max_prod < target:
                return None
            
            # Min product: smallest k values in [start_idx, idx_max]
            min_prod = 1
            for j in range(k):
                idx = start_idx + j
                min_prod *= vals[idx]
                if min_prod > target:
                    break
            if min_prod > target:
                return None
            
            key = (target, start_idx, count)
            if key in memo:
                return memo[key]
            
            for i in range(idx_max, start_idx - 1, -1):
                d_val = vals[i]
                if d_val > target:
                    continue
                if target % d_val == 0:
                    path = dfs(target // d_val, i + 1, count + 1)
                    if path is not None:
                        res = [d_val] + path
                        memo[key] = res
                        return res
            
            memo[key] = None
            return None
        
        return dfs(N, 0, 0)

    low_idx = 0
    high_idx = n_divs - 1
    best_idx = n_divs - 1
    best_path = None
    found = False
    
    while low_idx <= high_idx:
        if time.time() > deadline - 5:
            break
        mid_idx = (low_idx + high_idx) // 2
        path = can_decompose(mid_idx)
        if path is not None:
            best_idx = mid_idx
            best_path = path
            found = True
            high_idx = mid_idx - 1
        else:
            low_idx = mid_idx + 1
    
    if found and best_path:
        return best_path
    return [N]

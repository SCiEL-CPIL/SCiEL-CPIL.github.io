import time
import math
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    N = 479001600  # 12!
    
    # 1. Generate all divisors of N
    # N = 2^8 * 3^5 * 5^2 * 7^2 * 11
    # We can generate them by iterating up to sqrt(N)
    divisors = set()
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.add(i)
            divisors.add(N // i)
    
    # Filter divisors > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    n_divs = len(valid_divs)
    
    if n_divs == 0:
        return [N] # Fallback, though N > 12 always

    # Helper to check if N can be decomposed into distinct factors from valid_divs
    # where all factors <= M and count <= 12
    def can_decompose(M):
        # Find the index up to which divisors are <= M
        # valid_divs is sorted ascending
        # We want indices [0, end_idx] where valid_divs[end_idx] <= M
        # bisect_right returns the insertion point, so index - 1 is the last valid
        end_idx = bisect.bisect_right(valid_divs, M)
        
        if end_idx == 0:
            return False, None

        # We will search in the subarray valid_divs[0:end_idx]
        # But we must ensure factors are distinct. 
        # The DFS will pick indices in increasing order, ensuring distinctness.
        
        # We need to store the path to return the solution if successful
        path = []
        
        def dfs(target, start_idx, count):
            if time.time() > deadline:
                return False
            
            if target == 1:
                if 1 <= count <= 12:
                    return True
                return False
                
            if count >= 12:
                return False
                
            if start_idx >= end_idx:
                return False
                
            # Pruning: 
            # The maximum number of factors we can still pick is (12 - count).
            # The maximum product we can achieve with the remaining candidates 
            # (from start_idx to end_idx-1) picking the largest ones first.
            # If even the product of the largest (12-count) available divisors 
            # (that are > current last picked, which is implied by start_idx) 
            # is less than target, we fail.
            
            # However, calculating the exact max product is O(k). 
            # A simpler bound: 
            # The largest divisor in the range is valid_divs[end_idx - 1].
            # If target > (valid_divs[end_idx - 1]) ** (12 - count), it's impossible?
            # Not quite, because they are distinct.
            # But if target > valid_divs[end_idx - 1] * valid_divs[end_idx - 2] * ... 
            # This is expensive.
            
            # Let's stick to the basic iteration first. 
            # The number of divisors is ~900. 
            # Depth is <= 12.
            # Branching factor is limited by divisors of 'target'.
            
            # Optimization: 
            # Iterate i from start_idx to end_idx - 1
            for i in range(start_idx, end_idx):
                d = valid_divs[i]
                
                # Since valid_divs is sorted, if d > target, we can break
                # because any subsequent divisor will be larger and cannot divide target 
                # (unless target is 0, which it isn't, or if we allow 1, but d>12).
                # Wait, if d > target, then d cannot divide target (since target % d != 0 for d > target > 0).
                if d > target:
                    break
                    
                if target % d == 0:
                    path.append(d)
                    if dfs(target // d, i + 1, count + 1):
                        return True
                    path.pop()
                    
            return False

        # Initial call
        # We want to see if we can form N.
        # start_idx = 0 (smallest divisor > 12)
        # count = 0
        if dfs(N, 0, 0):
            return True, path[:]
            
        return False, None

    # 2. Binary Search for the minimum M
    # Low bound: 13 (smallest valid factor)
    # High bound: N (single factor N)
    low = 13
    high = N
    
    best_M = N
    best_factors = [N]
    found = False
    
    # We can optimize the search space.
    # The geometric mean of k factors is N^(1/k).
    # If k=1, M=N.
    # If k=2, M=sqrt(N) ~ 21886.
    # If k=3, M=cbrt(N) ~ 782.
    # If k=4, M=4th root ~ 263.
    # If k=5, M ~ 137.
    # If k=6, M ~ 95.
    # If k=7, M ~ 73.
    # If k=8, M ~ 58.
    # If k=9, M ~ 48.
    # If k=10, M ~ 41.
    # If k=11, M ~ 35.
    # If k=12, M ~ 31.
    
    # Since factors must be > 12, and distinct, the minimum possible max factor
    # is constrained by the smallest possible set of distinct integers > 12.
    # 13 * 14 * 15 ...
    # Let's just binary search [13, N].
    
    while low <= high:
        if time.time() > deadline - 5:
            break
            
        mid = (low + high) // 2
        
        possible, factors = can_decompose(mid)
        
        if possible:
            best_M = mid
            best_factors = factors
            found = True
            high = mid - 1
        else:
            low = mid + 1
            
    if not found:
        # Fallback if binary search fails or times out before finding anything
        # [N] is always a valid solution (1 factor, N > 12, distinct)
        return [N]
        
    return best_factors

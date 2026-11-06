import math
from functools import lru_cache

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order to help with pruning (larger divisors first)
    divisors.sort(reverse=True)
    
    # Use memoized DFS to find minimum number of divisors
    # State: (index, remaining_sum)
    # Returns: minimum number of divisors needed from divisors[index:] to sum to remaining_sum
    # Or returns -1 if impossible
    
    # We'll use a recursive approach with memoization
    # Since target is 4979 and we have ~60 divisors, this might be feasible
    
    @lru_cache(maxsize=None)
    def dfs(idx, rem):
        if rem == 0:
            return 0
        if idx >= len(divisors) or rem < 0:
            return -1
        
        # Option 1: Skip current divisor
        best = dfs(idx + 1, rem)
        
        # Option 2: Use current divisor
        if divisors[idx] <= rem:
            sub = dfs(idx + 1, rem - divisors[idx])
            if sub != -1:
                candidate = 1 + sub
                if best == -1 or candidate < best:
                    best = candidate
        
        return best
    
    # Find minimum number
    min_count = dfs(0, target)
    
    if min_count == -1:
        # Fallback: should not happen
        return []
    
    # Reconstruct the solution
    chosen = []
    idx = 0
    rem = target
    while rem > 0:
        # Check if we can achieve rem using divisors[idx:]
        # If we skip divisors[idx], can we still achieve min_count?
        # We need to track which path was taken
        
        # Let's re-do with a different approach: store the choice
        break
    
    # Better approach: use the DP to reconstruct
    # Let's use a different DP that stores the actual choices
    
    # Actually, let me use a simpler approach with bitsets or just iterate
    # Since we know the minimum count, we can try to find it greedily with backtracking
    
    # Alternative: use the DFS result to guide reconstruction
    # We'll trace back which divisors were chosen
    
    # Let's use a different strategy: iterate through divisors and decide
    # whether to include them based on whether the remaining sum is achievable
    
    chosen = []
    rem = target
    for i, d in enumerate(divisors):
        if d > rem:
            continue
        
        # Check if we can reach rem - d with remaining divisors
        # and if that would give us the optimal count
        # This requires knowing the optimal count from each state
        
        # Simpler: just use the fact that we want minimum count
        # Try including d if it leads to a valid solution with min_count terms
        
        # Let's just do a simple greedy with verification
        # Try to pick largest possible divisors that still allow completion
        
        pass
    
    # Let me restart with a cleaner approach using DP with path reconstruction
    
    # Clear cache and use a different method
    dfs.cache_clear()
    
    # Use DP with parent tracking
    # dp[idx][rem] = minimum number of divisors from divisors[idx:] to sum to rem
    # parent[idx][rem] = whether we used divisors[idx]
    
    # But this might use too much memory: 60 * 4980 = ~300k entries, which is fine
    
    m = len(divisors)
    
    # Initialize dp table
    # dp[i][r] = min divisors from divisors[i:] to sum to r, or -1 if impossible
    # We'll use a 2D array
    
    # Actually, let's use a 1D DP and store the path separately
    # Or use the recursive approach with path reconstruction
    
    # Let's use a different approach: since we want minimum count,
    # and we know the answer is small (probably < 10), we can use
    # iterative deepening or just trust the DFS and reconstruct
    
    # For reconstruction, let's modify the DFS to also track the path
    
    # Clear and redefine
    dfs.cache_clear()
    
    # Use a dict to store the best path
    best_path = {}
    
    def dfs_with_path(idx, rem, path):
        if rem == 0:
            return 0
        if idx >= len(divisors) or rem < 0:
            return -1
        
        # Option 1: Skip current divisor
        best = dfs_with_path(idx + 1, rem, path)
        best_path_count = best
        
        # Option 2: Use current divisor
        if divisors[idx] <= rem:
            sub = dfs_with_path(idx + 1, rem - divisors[idx], path + [divisors[idx]])
            if sub != -1:
                candidate = 1 + sub
                if best_path_count == -1 or candidate < best_path_count:
                    best_path_count = candidate
                    best_path[(idx, rem)] = divisors[idx]
        
        return best_path_count
    
    # This won't work well for reconstruction. Let me use a simpler approach.
    
    # Since the target is not too large and the number of divisors is small,
    # let's just use the original DP approach but optimize it
    
    # Go back to the DP approach but make it faster
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    prev_sum = [-1] * (target + 1)
    item_used = [-1] * (target + 1)
    
    for d in divisors:
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev_sum[s] = s - d
                item_used[s] = d
    
    if dp[target] == INF:
        return []
    
    chosen = []
    s = target
    while s > 0:
        d = item_used[s]
        if d == -1:
            break
        chosen.append(d)
        s = prev_sum[s]
    
    return chosen

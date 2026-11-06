import math
import time

def construct():
    start_time = time.time()
    n = 5040
    target = 4979
    
    # Get divisors
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    # Sort in descending order for better pruning in DFS, but DP handles any order.
    # For DP, order doesn't matter for correctness, only for reconstruction logic.
    # Let's keep them sorted ascending for 0-based indexing simplicity in reconstruction if needed,
    # or descending. The current code uses descending. Let's stick with descending.
    divisors.sort(reverse=True)
    
    num_divs = len(divisors)
    
    # DP: dp[s] = min number of divisors to sum to s
    # We use 1D DP for space efficiency, but we need to reconstruct the set.
    # To reconstruct with 1D DP, we can store the "last item" used to achieve the optimal count for each sum.
    # However, 1D DP overwrites previous states. 
    # Standard 0/1 knapsack reconstruction with 1D array requires careful backtracking or storing history.
    # Since target is ~5000 and num_divs is 60, 2D DP is very small (60 * 5000 = 300k ints).
    # Let's use 2D DP for simplicity and robustness of reconstruction.
    
    INF = float('inf')
    # dp[i][s] = min count using first i divisors (indices 0..i-1) to reach sum s
    dp = [[INF] * (target + 1) for _ in range(num_divs + 1)]
    dp[0][0] = 0
    
    for i in range(1, num_divs + 1):
        d = divisors[i-1]
        for s in range(target + 1):
            # Option 1: Don't take divisors[i-1]
            dp[i][s] = dp[i-1][s]
            # Option 2: Take divisors[i-1] if s >= d
            if s >= d:
                if dp[i-1][s-d] + 1 < dp[i][s]:
                    dp[i][s] = dp[i-1][s-d] + 1
                    
    min_count = dp[num_divs][target]
    
    if min_count == INF:
        # Should not happen for this instance, but handle gracefully
        return []
        
    # Reconstruct
    result = []
    s = target
    for i in range(num_divs, 0, -1):
        if dp[i][s] != dp[i-1][s]:
            # We took divisors[i-1]
            result.append(divisors[i-1])
            s -= divisors[i-1]
            
    # The loop iterates i from num_divs down to 1.
    # divisors are sorted descending.
    # i=num_divs corresponds to the smallest divisor (last in list).
    # i=1 corresponds to the largest divisor (first in list).
    # So result is built from smallest to largest?
    # Let's trace:
    # i=60 (smallest div), if taken, append.
    # i=1 (largest div), if taken, append.
    # So result list will have smaller divisors first if they were taken later in the iteration?
    # No, the iteration is i=60, 59, ..., 1.
    # If divisors[59] (smallest) is taken, it's appended first.
    # If divisors[0] (largest) is taken, it's appended last.
    # So result is sorted ascending?
    # The task doesn't require specific order, just the set.
    # "at most 256 distinct positive divisors ... whose sum is target"
    # "Minimize the number of selected divisors"
    # Return sorted for consistency.
    
    return sorted(result, reverse=True)

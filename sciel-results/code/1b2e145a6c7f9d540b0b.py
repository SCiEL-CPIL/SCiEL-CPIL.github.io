import time
import heapq

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    m = len(divisors)
    start_time = time.time()
    deadline = start_time + 60.0  # 60 second deadline with safety margin

    # Dijkstra's algorithm on state space of (sum, set of used divisors)
    # State: (sum, used_mask) where used_mask is a bitmask of used divisors
    # Since m can be up to 60, we can't use a simple bitmask for all divisors
    # Instead, we use a different approach: BFS by number of divisors
    
    # Let's use a BFS approach: find minimum k such that there exists k distinct divisors summing to target
    # For each k, check if any subset of size k sums to target
    
    # Better approach: Use DP with bitsets or meet-in-the-middle
    # Since target is 4979 and m is at most 60, let's try a smarter DP
    
    # Actually, let's stick with the DP approach but optimize it
    # The original DP is O(m * target) = O(60 * 5000) = 300k operations, which is fast
    
    INF = 10**9
    # dp[s] = min number of divisors to sum to s, using divisors[0..i]
    # We need to track which divisors were used for backtracking
    
    # Use a dictionary-based approach to save memory
    # dp[s] = (min_count, last_divisor_used)
    
    # Let's use a different DP: 
    # dp[i][s] = min number of divisors from divisors[i:] to sum to s
    # This allows us to use only future divisors
    
    # Actually, let's just use the original DP but with better memory management
    # The original approach is fine for this size
    
    dp = [[INF] * (target + 1) for _ in range(m + 1)]
    choice = [[False] * (target + 1) for _ in range(m + 1)]
    
    dp[0][0] = 0
    
    for i in range(1, m + 1):
        d = divisors[i - 1]
        for s in range(target + 1):
            # Option 1: Skip divisor i-1
            dp[i][s] = dp[i - 1][s]
            choice[i][s] = False
            
            # Option 2: Take divisor i-1
            if s >= d:
                if dp[i - 1][s - d] + 1 < dp[i][s]:
                    dp[i][s] = dp[i - 1][s - d] + 1
                    choice[i][s] = True
        
        # Check deadline
        if time.time() > deadline:
            break
    
    min_count = dp[m][target]
    
    if min_count == INF:
        # Fallback if no solution found
        # Try a greedy approach as a last resort
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        if remainder == 0:
            return chosen
        raise ValueError("No solution found")
    
    # Backtrack to find the selected divisors
    chosen = []
    s = target
    for i in range(m, 0, -1):
        if choice[i][s]:
            d = divisors[i - 1]
            chosen.append(d)
            s -= d
    
    # Verify
    assert len(set(chosen)) == len(chosen), "Not distinct"
    assert all(n % d == 0 for d in chosen), "Not all divisors of 5040"
    assert sum(chosen) == target, f"Sum mismatch: {sum(chosen)} != {target}"
    assert len(chosen) <= 256, "Too many divisors"
    
    return chosen

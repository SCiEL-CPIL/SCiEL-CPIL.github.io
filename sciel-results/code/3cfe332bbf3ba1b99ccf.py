"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 5040
    target = 4979

    # Generate divisors efficiently
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    
    divisors_desc = sorted(divisors, reverse=True)

    # Greedy solution
    remainder = target
    chosen_greedy = []
    for d in divisors_desc:
        if d <= remainder:
            chosen_greedy.append(d)
            remainder -= d
        if remainder == 0:
            break
    
    if remainder != 0:
        # Fallback if greedy fails (should not happen)
        return [1] * target # Invalid but safe fallback structure, though sum won't match if target>256? 
        # Actually, if greedy fails, we must use DP.

    best_solution = list(chosen_greedy)
    best_len = len(best_solution)

    if best_len == 1:
        return best_solution

    # DP for minimum count
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [(-1, -1)] * (target + 1)

    for d in divisors:
        # Optimization: if d > target, skip (not possible here as max div is 5040 > 4979? No 5040 > 4979)
        # Actually 5040 is a divisor. 5040 > 4979. So it cannot be used.
        # We can filter divisors > target.
        if d > target:
            continue
            
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
                
        # Time check
        if time.time() > deadline - 5.0:
            break

    if dp[target] == INF:
        return best_solution

    if dp[target] >= best_len:
        return best_solution

    # Reconstruct
    solution = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        if prev_s == -1:
            break
        solution.append(d)
        s = prev_s
    
    if len(solution) < best_len and sum(solution) == target:
        best_solution = solution
        best_len = len(solution)
        
    return best_solution
# EVOLVE-BLOCK-END

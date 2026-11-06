"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _get_divisors(n):
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    return divisors

# Precompute divisors for n=5040
N = 5040
TARGET = 4979
DIVISORS = _get_divisors(N)
# Sort descending to potentially find solutions with larger divisors first (though DP order doesn't affect min count, it can affect which solution is found)
DIVISORS.sort(reverse=True)

def construct():
    deadline = time.time() + 50.0
    
    # DP: dp[s] = minimum number of distinct divisors to achieve sum s
    INF = 10**9
    dp = [INF] * (TARGET + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, divisor_used) to reconstruct solution
    parent = [None] * (TARGET + 1)
    
    # For each divisor, update dp in reverse to avoid using same divisor twice
    for d in DIVISORS:
        if d > TARGET:
            continue
        # Optimization: if d is small, the range is large.
        # We iterate s from TARGET down to d.
        for s in range(TARGET, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
        
        # Time check (unlikely to be needed for 60 divisors)
        if time.time() > deadline:
            break
            
    if dp[TARGET] == INF:
        return []
    
    # Reconstruct the solution
    chosen = []
    s = TARGET
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s
    
    return chosen
# EVOLVE-BLOCK-END

import time
import sys

_N = 5040
_TARGET = 4979
_DEADLINE = 80.0

def _divisors_of(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)

def construct():
    start = time.time()
    divisors = _divisors_of(_N)
    # Filter divisors that are too large to be useful
    divisors = [d for d in divisors if d <= _TARGET]
    
    # DP for minimum cardinality subset sum
    # dp[s] = minimum number of divisors to sum to s
    # parent[s] = the last divisor added to reach s with the current optimal count
    # If multiple divisors lead to the same optimal count, any is fine, 
    # but we must ensure we don't reuse the same divisor.
    # The 1D DP with reverse iteration ensures 0/1 usage.
    
    INF = 10**9
    dp = [INF] * (_TARGET + 1)
    parent = [-1] * (_TARGET + 1)
    
    dp[0] = 0
    
    # Iterate through each divisor
    for d in divisors:
        # Iterate sums backwards to ensure each divisor is used at most once
        # We only need to update sums >= d
        for s in range(_TARGET, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
    
    if dp[_TARGET] == INF:
        # Fallback: Greedy solution (should not happen for practical numbers like 5040 
        # if target is reasonable, but safety first)
        remainder = _TARGET
        chosen = []
        # Use descending order for greedy to minimize count usually
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        if time.time() > _DEADLINE - 2:
            return chosen if sum(chosen) == _TARGET else []
        return chosen

    # Reconstruct the solution
    result = []
    s = _TARGET
    while s > 0:
        d = parent[s]
        if d == -1:
            # This shouldn't happen if dp[_TARGET] < INF
            break
        result.append(d)
        s -= d
        
    # Verify
    if sum(result) == _TARGET and len(set(result)) == len(result):
        return result
    
    # If reconstruction failed for some reason (logic error), fallback to greedy
    remainder = _TARGET
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen

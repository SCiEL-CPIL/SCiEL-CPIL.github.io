"""Find minimum-count subset of distinct divisors of 5040 summing to 4979."""
import time

def construct():
    start = time.time()
    deadline = 85.0  # safety margin under 90s kill

    n = 5040
    target = 4979

    # Generate all divisors of n
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1

    # Keep only divisors <= target (5040 > 4979 so it's excluded)
    divs = [d for d in divs if d <= target]
    # Sort descending for better pruning potential (though DP order doesn't matter for correctness)
    divs.sort(reverse=True)

    # Quick check: is target itself a divisor?
    if n % target == 0:
        return [target]

    # DP: dp[s] = minimum number of divisors to reach sum s
    # parent[s] = the divisor that was last added to reach sum s
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [-1] * (target + 1)

    # 0/1 knapsack: iterate each divisor once, sums in descending order
    for d in divs:
        if time.time() > deadline:
            break
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d

    best = dp[target]

    if best != INF:
        # Reconstruct the solution
        chosen = []
        s = target
        while s > 0:
            d = parent[s]
            if d <= 0:
                # Should not happen if dp is valid, but safety break
                break
            chosen.append(d)
            s -= d

        # Verify the solution
        if len(chosen) <= 256 and len(set(chosen)) == len(chosen):
            if all(n % d == 0 and d > 0 for d in chosen):
                if sum(chosen) == target:
                    return chosen

    # Fallback: greedy (should not be reached if DP works)
    remainder = target
    chosen = []
    for d in divs:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    return chosen

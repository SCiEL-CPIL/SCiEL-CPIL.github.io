"""Rewritten program using exact DP for minimum-cardinality subset sum."""
import time


def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 5040
    target = 4979

    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    num_divs = len(divisors)

    # DP: dp[s] = (min_count, prev_sum, divisor_index_used)
    # We want minimum number of divisors summing to target.
    INF = float('inf')
    dp_count = [INF] * (target + 1)
    dp_prev = [-1] * (target + 1)   # previous sum
    dp_div = [-1] * (target + 1)    # index of divisor used to reach this state
    dp_count[0] = 0

    # Process divisors in descending order to potentially find solutions faster
    # Standard 0/1 knapsack style: for each divisor, update sums in reverse.
    for idx, d in enumerate(divisors):
        if d > target:
            continue
        # Check deadline
        if time.time() > deadline:
            break
        # reverse to avoid reusing same divisor
        for s in range(target, d - 1, -1):
            if dp_count[s - d] + 1 < dp_count[s]:
                dp_count[s] = dp_count[s - d] + 1
                dp_prev[s] = s - d
                dp_div[s] = idx

    if dp_count[target] == INF:
        # Fallback: greedy approach
        rem = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= rem:
                chosen.append(d)
                rem -= d
            if rem == 0:
                break
        if rem == 0:
            return chosen
        return []

    # Reconstruct the set of divisors
    result = []
    s = target
    while s > 0:
        idx = dp_div[s]
        result.append(divisors[idx])
        s = dp_prev[s]

    result.sort()
    # Verify
    if sum(result) == target and len(set(result)) == len(result):
        return result
    # Fallback
    return []


# EVOLVE-BLOCK-END

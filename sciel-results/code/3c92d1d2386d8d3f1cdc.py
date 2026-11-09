"""Rewritten program: exact minimum-cardity representation of 4979 as a sum of
distinct divisors of 5040, using DP with parent tracking and a time guard."""
import time

def construct():
    t0 = time.time()
    DEADLINE = 55.0  # safety margin under 60s

    n = 5040
    target = 4979

    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)

    # Sort in descending order: larger divisors first helps the DP fill high sums
    # with fewer items earlier, and the 0/1 knapsack reverse-scan still works.
    divisors.sort(reverse=True)
    m = len(divisors)

    INF = float('inf')
    dp_count = [INF] * (target + 1)
    dp_parent = [(-1, -1)] * (target + 1)  # (prev_sum, divisor_value)
    dp_count[0] = 0

    for d in divisors:
        if d > target:
            continue
        # Reverse scan to enforce 0/1 (each divisor used at most once)
        for s in range(target, d - 1, -1):
            if dp_count[s - d] + 1 < dp_count[s]:
                dp_count[s] = dp_count[s - d] + 1
                dp_parent[s] = (s - d, d)

        # Time guard
        if time.time() - t0 > DEADLINE:
            break

    # Reconstruct
    if dp_count[target] == INF:
        # Fallback greedy (should not happen for this instance)
        rem = target
        chosen = []
        for d in divisors:
            if d <= rem:
                chosen.append(d)
                rem -= d
        if rem != 0:
            return []  # invalid, but avoids crash
        return chosen

    chosen = []
    s = target
    while s > 0:
        prev_s, d = dp_parent[s]
        chosen.append(d)
        s = prev_s

    # Verify
    assert len(set(chosen)) == len(chosen), "divisors not distinct"
    assert sum(chosen) == target, f"sum {sum(chosen)} != {target}"
    assert all(n % d == 0 for d in chosen), "not all divisors of 5040"
    assert len(chosen) <= 256, "too many divisors"

    return chosen

"""Exact minimum-cardity representation of 4979 as a sum of distinct divisors of 5040,
using 0/1 knapsack DP with parent tracking and a wall-clock deadline guard."""
import math
import time


def construct():
    t0 = time.time()
    DEADLINE = 55.0  # safety margin under the 90s evaluator kill

    n = 5040
    target = 4979

    # Generate all positive divisors of 5040
    divisors = []
    r = math.isqrt(n)
    for i in range(1, r + 1):
        if n % i == 0:
            divisors.append(i)
            j = n // i
            if j != i:
                divisors.append(j)

    # Descending order: larger divisors first helps the DP reach high sums with
    # fewer items, and the reverse-scan still enforces the 0/1 constraint.
    divisors.sort(reverse=True)

    INF = float("inf")
    dp_count = [INF] * (target + 1)
    dp_parent = [(-1, -1)] * (target + 1)  # (prev_sum, divisor_value)
    dp_count[0] = 0

    for d in divisors:
        if d > target:
            continue
        # Reverse scan enforces 0/1 (each divisor used at most once)
        for s in range(target, d - 1, -1):
            c = dp_count[s - d] + 1
            if c < dp_count[s]:
                dp_count[s] = c
                dp_parent[s] = (s - d, d)

        # Time guard
        if time.time() - t0 > DEADLINE:
            break

    # Reconstruct the minimum-cardity solution
    if dp_count[target] == INF:
        # Fallback: greedy over descending divisors (should not trigger for this instance)
        rem = target
        chosen = []
        for d in divisors:
            if d <= rem:
                chosen.append(d)
                rem -= d
        if rem != 0:
            return []
        return chosen

    chosen = []
    s = target
    while s > 0:
        prev_s, d = dp_parent[s]
        chosen.append(d)
        s = prev_s

    # Exact verification
    assert len(set(chosen)) == len(chosen), "divisors not distinct"
    assert sum(chosen) == target, f"sum {sum(chosen)} != {target}"
    assert all(n % d == 0 for d in chosen), "not all divisors of 5040"
    assert len(chosen) <= 256, "too many divisors"

    return chosen

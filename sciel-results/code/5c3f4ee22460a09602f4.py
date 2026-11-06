from collections import deque
import math

def construct():
    n = 5040
    target = 4979

    # Generate all positive divisors of n
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()

    # BFS on subset-sum states:
    # State: (sum_so_far, used_mask) is too large; instead we do DP over divisors
    # But to guarantee minimal count, we can do BFS on (sum, index) where we process
    # divisors in order and track minimum steps.
    #
    # Better approach: DP where dp[s] = min number of divisors to reach sum s,
    # using each divisor at most once. We iterate divisors and update in reverse.
    # This is the standard 0/1 knapsack for minimum items.
    #
    # To make it structurally distinct, we use a different formulation:
    # We track for each achievable sum, the minimum number of items,
    # and we use a dictionary-based sparse DP with early termination checks.

    INF = float('inf')
    # dp maps sum -> (min_count, bitmask or list of divisors)
    # For reconstruction, we store the parent pointer
    # dp[s] = (count, prev_s, divisor)
    dp = {0: (0, None, None)}

    for d in divisors:
        # Collect all states that can be extended by d
        new_entries = []
        for s, (cnt, prev_s, prev_d) in list(dp.items()):
            new_s = s + d
            if new_s <= target:
                new_cnt = cnt + 1
                if new_s not in dp or new_cnt < dp[new_s][0]:
                    new_entries.append((new_s, new_cnt, s, d))

        # Apply updates
        for new_s, new_cnt, prev_s, d_used in new_entries:
            dp[new_s] = (new_cnt, prev_s, d_used)

        # Early termination: if we've reached target with few items, we could stop
        # but we need to ensure minimal, so we continue unless we can prove minimality.
        # Since we process all divisors, the final dp[target] gives the minimum.

    if target not in dp:
        raise ValueError("No solution found")

    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        cnt, prev_s, d_used = dp[s]
        chosen.append(d_used)
        s = prev_s

    return chosen

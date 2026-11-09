# EVOLVE-BLOCK-START
import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s wall-clock kill

    def is_valid_set(s):
        # s: list of distinct positive ints
        if len(s) < 3:
            return True
        n = max(s)
        # dp[total] is a bitmask of which elements contributed to that total
        dp = [0] * (n + 1)
        dp[0] = 1
        for i, x in enumerate(s):
            bit = 1 << i
            # iterate descending to avoid reusing x
            for t in range(n, x - 1, -1):
                prev = dp[t - x]
                if prev:
                    dp[t] |= prev | bit
        for i, a in enumerate(s):
            target = 2 * a
            if target > n:
                continue
            # need a subset of OTHER elements summing to target
            if dp[target] & (~ (1 << i)):
                return False
        return True

    def greedy_build(order, limit=48):
        selected = []
        for x in order:
            if len(selected) >= limit:
                break
            cand = selected + [x]
            if is_valid_set(cand):
                selected.append(x)
        return selected

    # Deterministic baseline: ascending
    best = greedy_build(list(range(1, 513)))

    rng = random.Random(12345)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Randomized orderings to escape local traps
        order = list(range(1, 513))
        if attempts % 3 == 1:
            rng.shuffle(order)
        elif attempts % 3 == 2:
            # bias towards larger numbers first
            order = list(range(512, 0, -1))
            rng.shuffle(order)
        else:
            # bias towards smaller numbers first
            order = list(range(1, 513))
            rng.shuffle(order)

        res = greedy_build(order)
        if len(res) > len(best):
            best = res
            if len(best) >= 48:
                break

    return best
# EVOLVE-BLOCK-END

import time

N = 5040
TARGET = 4979

# Precompute all divisors of 5040
_divisors = []
for d in range(1, N + 1):
    if N % d == 0:
        _divisors.append(d)

# Only keep divisors <= TARGET (larger ones can't be part of a valid sum)
_usable = sorted([d for d in _divisors if d <= TARGET], reverse=True)
_usable_set = set(_usable)


def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin before 90s kill

    # DP: dp[s] = minimum number of distinct divisors needed to sum to s
    # Using -1 to indicate unreachable
    INF = 10**9
    dp = [-1] * (TARGET + 1)
    dp[0] = 0
    # Also track which divisors were used to achieve each sum (for reconstruction)
    # parent[s] = (prev_sum, divisor_used)
    parent = [None] * (TARGET + 1)

    # 0/1 knapsack style: for each divisor, update sums in reverse
    for d in _usable:
        # Process sums from high to low to avoid reusing the same divisor
        for s in range(TARGET, d - 1, -1):
            if dp[s - d] != -1 and (dp[s] == -1 or dp[s - d] + 1 < dp[s]):
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)

    if dp[TARGET] == -1:
        # Fallback: greedy (shouldn't happen for this instance)
        rem = TARGET
        chosen = []
        for d in _usable:
            if d <= rem:
                chosen.append(d)
                rem -= d
                if rem == 0:
                    break
        return chosen

    # Reconstruct the solution
    chosen = []
    s = TARGET
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s

    # Verify
    assert len(set(chosen)) == len(chosen), "Divisors not distinct"
    assert all(N % d == 0 for d in chosen), "Not all are divisors of 5040"
    assert sum(chosen) == TARGET, f"Sum mismatch: {sum(chosen)} != {TARGET}"
    assert len(chosen) <= 256, f"Too many divisors: {len(chosen)}"

    return chosen


# Pre-verify at module level to ensure correctness
_result = construct()
assert _result is not None
assert sum(_result) == TARGET
assert all(N % d == 0 for d in _result)
assert len(set(_result)) == len(_result)
assert len(_result) <= 256

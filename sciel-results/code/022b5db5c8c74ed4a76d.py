def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0

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
    m = len(divisors)

    if time.time() - start_time > DEADLINE:
        return []

    # 0/1 knapsack DP minimizing number of items
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [(-1, -1)] * (target + 1)
    dp[0] = 0

    for i in range(m):
        if time.time() - start_time > DEADLINE:
            break
        d = divisors[i]
        if d > target:
            continue
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, i)

    if dp[target] == INF:
        return []

    # Reconstruct
    result = []
    s = target
    while s > 0:
        prev_s, idx = parent[s]
        if idx == -1:
            break
        result.append(divisors[idx])
        s = prev_s

    return result

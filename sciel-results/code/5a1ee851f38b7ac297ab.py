def construct():
    import time
    deadline = time.time() + 85

    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)

    if n <= 2:
        return primes

    dp = [[0] * n for _ in range(n)]

    for i in range(n - 1):
        k_min = i
        running_max = -1
        for j in range(i + 1, n):
            threshold = 2 * primes[i] - primes[j]
            while k_min > 0 and primes[k_min - 1] >= threshold:
                k_min -= 1
                if dp[k_min][i] > running_max:
                    running_max = dp[k_min][i]
            if running_max > 0:
                dp[i][j] = running_max + 1
            else:
                dp[i][j] = 2
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    max_len = 0
    best_i = 0
    best_j = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if max_len <= 1:
        return primes[:1]

    seq = [primes[best_i], primes[best_j]]
    i, j = best_i, best_j
    while len(seq) < max_len:
        target_len = len(seq) - 1
        gap_ij = primes[j] - primes[i]
        found = False
        for k in range(i - 1, -1, -1):
            if dp[k][i] == target_len and (primes[i] - primes[k]) <= gap_ij:
                seq.append(primes[k])
                i, j = k, i
                found = True
                break
        if not found:
            break

    seq.reverse()
    return seq

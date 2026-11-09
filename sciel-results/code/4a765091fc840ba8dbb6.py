import time
from math import isqrt

def construct():
    deadline = time.time() + 85.0

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(n + 1) if is_prime[i]]

    primes = sieve(10000)
    n = len(primes)

    if n == 0:
        return []
    if n == 1:
        return primes

    # dp[i] = list of (length, last_gap, prev_index) Pareto-optimal states
    dp = [[] for _ in range(n)]
    for i in range(n):
        dp[i].append((1, 0, -1))

    for i in range(1, n):
        if time.time() > deadline:
            break
        for j in range(i):
            gap = primes[i] - primes[j]
            for (length, last_gap, prev_idx) in dp[j]:
                if gap >= last_gap:
                    new_state = (length + 1, gap, j)
                    dominated = False
                    for (l2, g2, p2) in dp[i]:
                        if l2 >= new_state[0] and g2 <= new_state[1]:
                            dominated = True
                            break
                    if not dominated:
                        dp[i] = [(l, g, p) for (l, g, p) in dp[i]
                                  if not (new_state[0] >= l and new_state[1] <= g)]
                        dp[i].append(new_state)

    best_len = 0
    best_i = 0
    for i in range(n):
        for (length, last_gap, prev_idx) in dp[i]:
            if length > best_len:
                best_len = length
                best_i = i

    # Reconstruct
    result = [primes[best_i]]
    cur = best_i
    cur_len = best_len

    while cur_len > 1:
        cur_state = None
        for s in dp[cur]:
            if s[0] == cur_len:
                cur_state = s
                break
        if cur_state is None:
            break
        prev_idx = cur_state[2]
        if prev_idx == -1:
            break
        result.append(primes[prev_idx])
        old_cur = cur
        cur = prev_idx
        cur_len -= 1

    result.reverse()
    return result

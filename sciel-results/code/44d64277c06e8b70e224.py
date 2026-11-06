import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin under 90s wall-clock

    # --- Sieve of Eratosthenes: primes <= 10000 ---
    LIMIT = 10000
    sieve = bytearray(b'\x01') * (LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(LIMIT ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    primes = [i for i in range(2, LIMIT + 1) if sieve[i]]
    n = len(primes)
    if n == 0:
        return []
    if n == 1:
        return [primes[0]]

    min_prime = primes[0]
    max_prime = primes[-1]

    sys.setrecursionlimit(100000)

    # Memo: key (index, last_gap) -> (best_len_from_here, path_from_here)
    memo = {}
    # Best complete path found so far (list of primes, increasing, nondecreasing gaps)
    best_path = [primes[0]]
    best_len = 1

    def dfs(i, last_gap):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return (0, [])
        key = (i, last_gap)
        cached = memo.get(key)
        if cached is not None:
            return cached

        p = primes[i]

        # Upper bound: even taking the smallest possible gap each step,
        # how many more primes can we reach before exceeding LIMIT?
        min_gap = p - min_prime  # smallest gap from p to any smaller prime
        if min_gap <= 0:
            min_gap = 1
        max_extra = 0
        t = p
        while t + min_gap <= LIMIT:
            t += min_gap
            max_extra += 1
            if max_extra > 500:  # safety cap
                break

        # If even the optimistic bound can't beat the current best, prune.
        # (cur_len is accounted for by caller; here we return length from i onward)
        # We'll let caller handle the global comparison; prune only when hopeless.

        best_here_len = 1
        best_here_path = [p]

        for j in range(i + 1, n):
            gap = primes[j] - p
            if gap < last_gap:
                continue
            sub_len, sub_path = dfs(j, gap)
            if sub_len + 1 > best_here_len:
                best_here_len = sub_len + 1
                best_here_path = [p] + sub_path
            # early stop: cannot do better than a full jump to the end
            if best_here_len >= n - i:
                break

        # Record global best
        if best_here_len > best_len:
            best_len = best_here_len
            best_path = best_here_path

        memo[key] = (best_here_len, best_here_path)
        return memo[key]

    # Start from every prime as a fresh sequence (last_gap = 0)
    for i in range(n):
        if time.time() > deadline:
            break
        sub_len, sub_path = dfs(i, 0)
        if sub_len > best_len:
            best_len = sub_len
            best_path = sub_path

    return list(best_path)
# EVOLVE-BLOCK-END

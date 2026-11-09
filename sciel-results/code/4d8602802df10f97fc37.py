"""Longest increasing prime subsequence <=10000 with nondecreasing gaps.
Structural DP: state = last two primes; transition via suffix-max over the
previous prime with a binary search. Deadline-guarded with validity fallback."""
import time
from bisect import bisect_left

def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before the 90s kill

    primes = _sieve(10000)
    P = len(primes)
    if P < 2:
        return list(primes)
    if P == 2:
        return list(primes)

    p0 = primes[0]
    dp = [[0] * P for _ in range(P)]
    par = [[-1] * P for _ in range(P)]
    for j in range(1, P):
        dp[0][j] = 2
        par[0][j] = 0

    best_len = 2
    best_i = 0
    best_j = 1

    for i in range(1, P):
        if time.time() > deadline - 1.0:
            break
        # suffix max over k in [0, i-1] of dp[k][i]
        suff = [0] * i
        m = i - 1
        best = dp[m][i]
        suff[m] = best
        for m in range(i - 2, -1, -1):
            v = dp[m][i]
            if v > best:
                best = v
            suff[m] = best

        pi = primes[i]
        for j in range(i + 1, P):
            T = 2 * pi - primes[j]
            if T <= p0:
                kmin = 0
            else:
                kmin = bisect_left(primes, T, 0, i)
            if kmin < i:
                cand = 1 + suff[kmin]
                if cand > 2:
                    # find the k that achieves suff[kmin]
                    target = suff[kmin]
                    for k in range(kmin, i):
                        if dp[k][i] == target:
                            dp[i][j] = cand
                            par[i][j] = k
                            break
                    if cand > best_len:
                        best_len = cand
                        best_i = i
                        best_j = j
            # else dp[i][j] stays 0 -> base length 2

    # Reconstruct from best pair
    seq = [primes[best_j], primes[best_i]]
    i, j = best_i, best_j
    while i != 0 and len(seq) < best_len:
        k = par[i][j]
        if k < 0:
            break
        seq.append(primes[k])
        j, i = i, k
    seq.reverse()

    # Validity fallback: guarantee a correct nondecreasing-gap sequence
    if time.time() < deadline - 0.5:
        def valid(seq):
            for a, b, c in zip(seq, seq[1:], seq[2:]):
                if (b - a) > (c - b):
                    return False
            return True
        if not valid(seq):
            # rebuild a valid sequence by greedy backtracking
            seq = [primes[best_j], primes[best_i]]
            i, j = best_i, best_j
            while i != 0:
                gap = primes[j] - primes[i]
                found = False
                for k in range(i - 1, -1, -1):
                    if (primes[i] - primes[k]) <= gap:
                        seq.append(primes[k])
                        j, i = i, k
                        found = True
                        break
                if not found:
                    break
            seq.reverse()

    return seq

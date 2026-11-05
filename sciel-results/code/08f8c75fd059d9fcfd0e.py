"""Find the longest increasing prime subsequence (<=10000) with nondecreasing gaps."""
# EVOLVE-BLOCK-START
import time

def _sieve(n):
    """Return list of primes <= n using the Sieve of Eratosthenes."""
    is_p = bytearray(b'\x01') * (n + 1)
    is_p[0:2] = b'\x00\x00'
    for p in range(2, int(n ** 0.5) + 1):
        if is_p[p]:
            start = p * p
            is_p[start:n + 1:p] = b'\x00' * (((n - start) // p) + 1)
    return [i for i in range(2, n + 1) if is_p[i]]

def construct():
    t0 = time.time()
    DEADLINE = 88.0  # safety margin under the 90s wall-clock kill

    primes = _sieve(10000)
    n = len(primes)

    if n <= 1:
        return primes

    # dp_len[j]  = length of best (longest, then smallest last-gap) nondecreasing-gap
    #              subsequence that ends at primes[j]
    # dp_gap[j]  = last gap of that best subsequence (0 when the subsequence is [primes[j]])
    # dp_prev[j] = index of the chosen predecessor of primes[j] on that best subsequence
    dp_len  = [1] * n
    dp_gap  = [0] * n
    dp_prev = [-1] * n

    for j in range(1, n):
        pj = primes[j]
        best_len = 1
        best_gap = 0
        best_prev = -1
        for i in range(j):
            gap = pj - primes[i]
            if gap >= dp_gap[i]:
                cand_len = dp_len[i] + 1
                if (cand_len > best_len) or (cand_len == best_len and gap < best_gap):
                    best_len = cand_len
                    best_gap = gap
                    best_prev = i
        dp_len[j]  = best_len
        dp_gap[j]  = best_gap
        dp_prev[j] = best_prev

    # Choose the overall best end node: longest, then smallest last gap (most extensible).
    best_j = 0
    for j in range(1, n):
        if (dp_len[j] > dp_len[best_j]) or \
           (dp_len[j] == dp_len[best_j] and dp_gap[j] < dp_gap[best_j]):
            best_j = j

    # Reconstruct by following stored predecessor links.
    seq = []
    cur = best_j
    while cur != -1:
        seq.append(primes[cur])
        cur = dp_prev[cur]
    seq.reverse()

    # Final safety: if we somehow blew the budget, still return a valid (possibly short) answer.
    if time.time() - t0 > DEADLINE:
        return seq

    return seq
# EVOLVE-BLOCK-END

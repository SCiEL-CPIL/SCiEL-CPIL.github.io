"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _sieve(n):
    """Return list of primes <= n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    primes = _sieve(10000)
    n = len(primes)
    
    # dp[i] = (length, prev_index) for longest valid subsequence ending at primes[i]
    # A valid subsequence p_{i_k} < ... < p_{i_1} < p_i has gaps
    # g_k = p_{i_k} - p_{i_{k-1}} with g_1 <= g_2 <= ... <= g_k
    # dp[i] stores the best (longest) chain ending at i, and the index of the
    # previous prime in that chain (to check gap constraint).
    
    # For efficiency, we store for each i the best chain length and the
    # previous prime index. But we need to be careful: among chains of the
    # same length ending at i, we want the one with the SMALLEST last gap
    # (i.e., smallest primes[i] - primes[prev]) because that makes it easier
    # to extend. Actually, for extending, we need the gap from prev to i to be
    # >= the gap before prev. So we want to track, for each i, the chain that
    # maximizes length, and among ties, minimizes the last gap to make future
    # extension easier.
    
    # Let's use: dp_len[i] = max length of valid subsequence ending at i
    # dp_prev[i] = index of previous prime in that subsequence (or -1 if length 1)
    # To handle ties properly, among subsequences of the same max length ending at i,
    # we prefer the one with the smallest gap (primes[i] - primes[dp_prev[i]])
    # because a smaller last gap makes it easier for a future prime k > i to satisfy
    # primes[k] - primes[i] >= primes[i] - primes[dp_prev[i]].
    
    dp_len = [1] * n
    dp_prev = [-1] * n
    
    for i in range(1, n):
        best_len = 1
        best_prev = -1
        best_gap = float('inf')  # gap from prev to i; we want smallest among ties
        
        pi = primes[i]
        
        for j in range(i):
            if time.time() > deadline:
                break
            gap_ji = pi - primes[j]
            if dp_len[j] == 1:
                # Can always extend a length-1 chain
                cand_len = 2
                if cand_len > best_len:
                    best_len = cand_len
                    best_prev = j
                    best_gap = gap_ji
                elif cand_len == best_len and gap_ji < best_gap:
                    best_prev = j
                    best_gap = gap_ji
            else:
                prev_j = dp_prev[j]
                gap_prev = primes[j] - primes[prev_j]
                if gap_ji >= gap_prev:
                    cand_len = dp_len[j] + 1
                    if cand_len > best_len:
                        best_len = cand_len
                        best_prev = j
                        best_gap = gap_ji
                    elif cand_len == best_len and gap_ji < best_gap:
                        best_prev = j
                        best_gap = gap_ji
        
        dp_len[i] = best_len
        dp_prev[i] = best_prev
    
    # Find the index with maximum dp_len
    max_len = max(dp_len)
    end_idx = dp_len.index(max_len)
    
    # Reconstruct
    seq = []
    idx = end_idx
    while idx != -1:
        seq.append(primes[idx])
        idx = dp_prev[idx]
    seq.reverse()
    
    return seq
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 10000
    sieve = np.ones(N + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.where(sieve)[0]
    m = len(primes)
    
    if m < 2:
        return primes.tolist()
    
    dp = np.zeros((m, m), dtype=np.int32)
    
    for i in range(m):
        cur_lo = i
        max_val = 0
        for j in range(i + 1, m):
            threshold = 2 * primes[i] - primes[j]
            new_lo = np.searchsorted(primes, threshold, side='left')
            if new_lo < cur_lo:
                if cur_lo > 0:
                    segment_max = np.max(dp[new_lo:cur_lo, i])
                    if segment_max > max_val:
                        max_val = segment_max
                cur_lo = new_lo
            if cur_lo < i:
                dp[i][j] = max_val + 1
            else:
                dp[i][j] = 1
    
    best_len = int(np.max(dp))
    best_i, best_j = np.unravel_index(np.argmax(dp), dp.shape)
    
    seq = [int(primes[best_j]), int(primes[best_i])]
    i, j = int(best_i), int(best_j)
    while dp[i][j] > 1:
        threshold = 2 * primes[i] - primes[j]
        lo = np.searchsorted(primes, threshold, side='left')
        if lo < i:
            candidates = dp[lo:i, i]
            best_offset = np.argmax(candidates)
            i_prev = lo + best_offset
            if dp[i_prev][i] == dp[i][j] - 1:
                seq.append(int(primes[i_prev]))
                i, j = int(i_prev), i
            else:
                break
        else:
            break
    
    seq.reverse()
    return seq
# EVOLVE-BLOCK-END

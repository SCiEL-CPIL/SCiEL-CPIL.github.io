import time
import numpy as np

_DEADLINE = time.time() + 85.0


def construct():
    global _DEADLINE
    deadline = _DEADLINE
    LIMIT = 10000

    # Sieve primes up to LIMIT using NumPy.
    sieve = np.ones(LIMIT + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for x in range(2, int(LIMIT ** 0.5) + 1):
        if sieve[x]:
            sieve[x * x::x] = False
    primes = np.array([i for i in range(2, LIMIT + 1) if sieve[i]], dtype=np.int32)
    N = len(primes)

    # Maximum possible gap between two primes <= LIMIT is LIMIT - 2.
    G = LIMIT - 2

    # dp[i][g]: max length of valid sequence ending at prime i with last gap exactly g.
    # Use -1 to indicate "unreachable".
    dp = np.full((N, G + 1), -1, dtype=np.int32)
    pref = np.full((N, G + 1), -1, dtype=np.int32)
    pred = np.full((N, G + 1), -1, dtype=np.int32)

    # Base case: a single prime is a valid sequence of length 1 with no gap.
    dp[:, 0] = 1
    pref[:, 0] = 1

    best_len = 1
    best_i = 0
    best_g = 0

    for i in range(1, N):
        if time.time() > deadline:
            break
        pi = primes[i]
        
        # Vectorized transition
        # gaps[j] = pi - primes[j] for j in 0..i-1
        gaps = pi - primes[:i]
        
        # Get previous best values: pref[j, gaps[j]]
        # indices: row indices 0..i-1, col indices gaps[0]..gaps[i-1]
        prev_vals = pref[:i, gaps]
        
        # Valid if prev_val >= 0
        valid = prev_vals >= 0
        vals = np.where(valid, prev_vals + 1, -1)
        
        # Update dp[i, gaps] with vals
        # np.maximum.at handles duplicate indices, but here gaps are unique for fixed i
        # because primes are unique.
        np.maximum.at(dp[i], gaps, vals)
        
        # Update pred[i, gaps] with j indices
        # Only update if the transition was valid and actually improved dp?
        # Since each gap comes from a unique j, and dp starts at -1,
        # if vals > -1, it will update dp.
        # We can just set pred for all valid transitions.
        pred[i, gaps[valid]] = np.arange(i)[valid]
        
        # Compute prefix maximums for row i
        pref[i] = np.maximum.accumulate(dp[i])
        
        # Update global best
        row_max = int(np.max(dp[i]))
        if row_max > best_len:
            best_len = row_max
            best_g = int(np.argmax(dp[i]))
            best_i = i

    # Reconstruct the sequence by walking predecessors.
    seq = []
    if best_len > 0:
        i = best_i
        g = best_g
        while i >= 0:
            seq.append(int(primes[i]))
            if i == 0:
                break
            j = int(pred[i][g])
            if j == -1:
                break
            # Find the specific gap h at predecessor j such that dp[j][h] == pref[j][g]
            # and h <= g.
            target_val = int(pref[j][g])
            h = g
            while h >= 0:
                if int(dp[j][h]) == target_val:
                    break
                h -= 1
            if h < 0:
                h = g
            i = j
            g = h
        seq.reverse()

    # Safety fallback
    if not seq:
        from math import isqrt
        vals = []
        for p in range(2, 10001):
            if all(p % d for d in range(2, isqrt(p) + 1)):
                if len(vals) < 2 or p - vals[-1] >= vals[-1] - vals[-2]:
                    vals.append(p)
        seq = vals

    return seq

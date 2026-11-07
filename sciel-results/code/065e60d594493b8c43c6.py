"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

def _sigma_sieve(n):
    """Compute sigma(k) for k=1..n using a numpy-accelerated sieve."""
    sigma = np.zeros(n + 1, dtype=np.int64)
    for d in range(1, n + 1):
        sigma[d::d] += d
    return sigma

def _sigma_exact(n):
    """Exact sigma(n) via trial division (with isqrt cutoff)."""
    if n <= 0:
        return 0
    s = 1
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            term = 1
            pk = 1
            while x % p == 0:
                x //= p
                pk *= p
                term += pk
            s *= term
        p += 1 if p == 2 else 2
    if x > 1:
        s *= (1 + x)
    return s

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # Start with the known 3-way collision as fallback
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # Search for collisions using numpy-accelerated sieve
    for limit in [10_000_000, 20_000_000, 40_000_000, 60_000_000]:
        if time.time() > deadline - 30:
            break
        try:
            sigma = _sigma_sieve(limit)
            # Vectorized: compute v = k * sigma[k] for k=1..limit
            ks = np.arange(1, limit + 1, dtype=np.int64)
            vals = ks * sigma[1:]
            
            # Filter: v <= 10^18 and k <= sigma(k) (i.e., k <= sqrt(v))
            max_M = np.int64(10**18)
            mask = (vals <= max_M) & (ks <= sigma[1:])
            
            valid_vals = vals[mask]
            valid_ks = ks[mask]
            del vals
            
            # Find duplicates using sorting
            if len(valid_vals) > 0:
                order = np.argsort(valid_vals, kind='stable')
                sorted_vals = valid_vals[order]
                sorted_ks = valid_ks[order]
                del valid_vals, valid_ks
                
                # Find group boundaries where values change
                diffs = np.diff(sorted_vals)
                boundaries = np.where(diffs > 0)[0] + 1
                group_starts = np.concatenate([[0], boundaries])
                group_ends = np.concatenate([boundaries, [len(sorted_vals)]])
                group_sizes = group_ends - group_starts
                
                # Find the largest group
                max_idx = int(np.argmax(group_sizes))
                if group_sizes[max_idx] > best_count:
                    gs = int(group_starts[max_idx])
                    ge = int(group_ends[max_idx])
                    val = int(sorted_vals[gs])
                    group_ks = sorted_ks[gs:ge].tolist()
                    # Verify k <= isqrt(val)
                    sq = math.isqrt(val)
                    valid_ks_group = [k for k in group_ks if k <= sq]
                    if len(valid_ks_group) > best_count:
                        best_count = len(valid_ks_group)
                        best_M = val
                        best_k = valid_ks_group[:1000]
                
                del sorted_vals, sorted_ks, diffs, boundaries, group_starts, group_ends, group_sizes
            del sigma
            del ks
        except (MemoryError, Exception):
            break

    # Verify with exact factorization for safety
    verified = []
    for k in best_k:
        if k * _sigma_exact(k) == best_M and k <= math.isqrt(best_M):
            verified.append(k)
    
    if len(verified) > 0:
        best_k = verified
    else:
        # Fallback to known solution
        best_M = 333312
        best_k = [336, 372, 434]

    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k[:1000]]

    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END

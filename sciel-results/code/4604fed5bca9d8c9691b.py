"""Rewritten program: sieve-based enumeration to find M with many preimages k where k*sigma(k)=M."""
import math
import time
import sys

def _sigma_sieve(limit):
    """Compute sigma(n) for all n in 1..limit using a divisor-sum sieve.
    Returns a list sigma where sigma[i] = sum of divisors of i."""
    sigma = [0] * (limit + 1)
    for d in range(1, limit + 1):
        for m in range(d, limit + 1, d):
            sigma[m] += d
    return sigma

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill
    
    # Strategy: enumerate k up to some limit, compute M = k*sigma(k), group by M.
    # Find the M with the most distinct k values, subject to k <= isqrt(M) and M <= 10^18.
    
    # We need k*sigma(k) <= 10^18 and k <= isqrt(k*sigma(k)).
    # The constraint k <= isqrt(M) means k^2 <= M = k*sigma(k), i.e., k <= sigma(k).
    # This is always true for k >= 1 (sigma(k) >= k+1 for k>1, sigma(1)=1 so k=1: 1<=1 ok).
    # Actually sigma(k) >= k for all k, with equality only for k=1. For k>1, sigma(k) > k.
    # So k <= sigma(k) is always satisfied. Good.
    
    # The binding constraint is M = k*sigma(k) <= 10^18.
    # For k up to ~10^9, sigma(k) ~ O(k log log k), so k*sigma(k) ~ O(k^2 log log k).
    # k^2 * log log k <= 10^18 => k <= ~10^9 / sqrt(log log k) ~ 10^9.
    # But we can't sieve up to 10^9 in 130s easily. Let's be practical.
    
    # A sieve up to N takes O(N log N) time. For N=10^7, that's ~10^7 * 16 ~ 1.6*10^8 ops, feasible.
    # For N=5*10^7, ~5*10^7 * 17 ~ 8.5*10^8, might be tight.
    # Let's try N up to what we can handle.
    
    # Try multiple sieve sizes, starting large, and see what fits in time.
    # Actually, let's just pick a reasonable N and go.
    
    # For N = 10^7: sieve takes maybe 5-10 seconds in Python with naive loops.
    # But Python loops are slow. Let's use a more efficient approach.
    
    # Alternative: use numpy for the sieve.
    try:
        import numpy as np
        HAVE_NUMPY = True
    except ImportError:
        HAVE_NUMPY = False
    
    if HAVE_NUMPY:
        # Use numpy for faster sieve
        # Try N = 20_000_000 first, fall back if too slow
        for N in [20_000_000, 10_000_000, 5_000_000]:
            if time.time() - start_time > DEADLINE - 20:
                break
            try:
                sigma = np.zeros(N + 1, dtype=np.int64)
                for d in range(1, N + 1):
                    sigma[d::d] += d
                
                elapsed = time.time() - start_time
                if elapsed > DEADLINE - 30:
                    continue
                
                # Now group by M = k * sigma(k)
                # k ranges from 1 to N
                ks = np.arange(1, N + 1, dtype=np.int64)
                Ms = ks * sigma[1:N+1]
                
                # Filter: M <= 10^18
                valid = Ms <= 10**18
                Ms_valid = Ms[valid]
                ks_valid = ks[valid]
                
                # Also need k <= isqrt(M), which as discussed is always true for k>1.
                # But let's double-check: k <= isqrt(M) means k^2 <= M = k*sigma(k), i.e. k <= sigma(k).
                # This is always true. So no additional filter needed.
                
                # Group by M value, find the M with most distinct k
                # Use a dictionary
                from collections import defaultdict
                groups = defaultdict(list)
                for k_val, m_val in zip(ks_valid.tolist(), Ms_valid.tolist()):
                    groups[m_val].append(k_val)
                
                # Find best M
                best_M = None
                best_ks = []
                for m, k_list in groups.items():
                    if len(k_list) > len(best_ks):
                        best_M = m
                        best_ks = k_list
                
                if best_M is not None:
                    # Verify
                    best_ks = sorted(set(best_ks))
                    if len(best_ks) >= 3:
                        return {'M': int(best_M), 'k': [int(x) for x in best_ks]}
                        
            except (MemoryError, Exception):
                continue
    
    # Fallback: pure Python sieve with smaller N
    for N in [5_000_000, 2_000_000, 1_000_000]:
        if time.time() - start_time > DEADLINE - 20:
            break
        try:
            sigma = _sigma_sieve(N)
            
            if time.time() - start_time > DEADLINE - 30:
                continue
            
            from collections import defaultdict
            groups = defaultdict(list)
            for k in range(1, N + 1):
                M = k * sigma[k]
                if M <= 10**18:
                    groups[M].append(k)
            
            best_M = None
            best_ks = []
            for m, k_list in groups.items():
                if len(k_list) > len(best_ks):
                    best_M = m
                    best_ks = k_list
            
            if best_M is not None:
                best_ks = sorted(set(best_ks))
                if len(best_ks) >= 3:
                    return {'M': int(best_M), 'k': [int(x) for x in best_ks]}
                    
        except (MemoryError, Exception):
            continue
    
    # Ultimate fallback: return the seed result
    return {'M': 333312, 'k': [336, 372, 434]}

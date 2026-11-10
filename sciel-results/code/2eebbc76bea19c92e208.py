import numpy as np
import time
from math import isqrt

def _sigma_sieve(N):
    """Compute sigma(n) for all n up to N using SPF sieve."""
    spf = np.arange(N + 1, dtype=np.int64)
    sqrt_N = int(np.sqrt(N))
    for i in range(2, sqrt_N + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    rem = np.zeros(N + 1, dtype=np.int64)
    cnt = np.zeros(N + 1, dtype=np.int64)
    sigma = np.ones(N + 1, dtype=np.int64)

    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m >= 2 and spf[m] == p:
            rem[n] = rem[m]
            cnt[n] = cnt[m] + 1
        else:
            rem[n] = m
            cnt[n] = 1

        a = cnt[n]
        geo = 1
        for _ in range(a):
            geo = geo * p + 1
        sigma[n] = sigma[rem[n]] * geo

    return sigma

def construct():
    start = time.time()
    deadline = 140.0

    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # Strategy: Sieve up to large N, compute M = k*sigma(k), group by M
    # Focus on finding the M with most preimages

    N = 50_000_000
    try:
        sigma = _sigma_sieve(N)
        
        k_vals = np.arange(1, N + 1, dtype=np.int64)
        M_vals = k_vals * sigma[1:]
        
        mask = M_vals <= 10**18
        M_valid = M_vals[mask]
        k_valid = k_vals[mask]

        unique_M, inverse_indices, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
        max_count = int(np.max(counts))
        if max_count > best_count:
            idx = int(np.argmax(counts))
            target_M = int(unique_M[idx])
            k_solutions = k_valid[inverse_indices == idx]
            best_M = target_M
            best_k = [int(x) for x in k_solutions]
            best_count = max_count

    except MemoryError:
        pass
    except Exception:
        pass

    # Try to extend if time permits
    if time.time() - start < deadline - 30:
        N2 = 80_000_000
        try:
            sigma2 = _sigma_sieve(N2)
            k_vals2 = np.arange(1, N2 + 1, dtype=np.int64)
            M_vals2 = k_vals2 * sigma2[1:]
            mask2 = M_vals2 <= 10**18
            M_valid2 = M_vals2[mask2]
            k_valid2 = k_vals2[mask2]
            
            unique_M2, inverse_indices2, counts2 = np.unique(M_valid2, return_inverse=True, return_counts=True)
            max_count2 = int(np.max(counts2))
            if max_count2 > best_count:
                idx2 = int(np.argmax(counts2))
                best_M = int(unique_M2[idx2])
                best_k = [int(x) for x in k_valid2[inverse_indices2 == idx2]]
                best_count = max_count2
        except MemoryError:
            pass
        except Exception:
            pass

    return {'M': best_M, 'k': best_k}

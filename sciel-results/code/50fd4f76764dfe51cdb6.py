import math
import time
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, use 135s
    deadline = start_time + 135

    # We want to find M with many solutions k*sigma(k)=M
    # k <= sqrt(M) is always true for k>=1
    # We search k in [1, N]

    # Increase N to find more collisions
    # Using 30M to balance time and memory. int32 for sigma saves memory.
    N = 30_000_000

    try:
        # Sieve for sigma
        # sigma[k] fits in int32 for k up to 30e6 (max sigma ~ 120e6 < 2^31)
        sigma = np.zeros(N + 1, dtype=np.int32)

        for d in range(1, N + 1):
            sigma[d::d] += d

        # Check time
        if time.time() > deadline - 30:
            # If too slow, fallback
            return {'M': 333312, 'k': [336, 372, 434]}

        # Compute M = k * sigma(k)
        # k needs to be int64 to avoid overflow in multiplication
        k_arr = np.arange(1, N + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:]

        # Find most frequent M
        unique_M, counts = np.unique(M_arr, return_counts=True)

        max_count = np.max(counts)

        # If no collisions found (max_count == 1), we still return one solution
        # But we want to maximize, so we pick the first one with max count
        best_M_idx = np.argmax(counts)
        best_M = int(unique_M[best_M_idx])

        # Get all k that map to best_M
        indices = np.where(M_arr == best_M)[0]
        k_sols_np = k_arr[indices]

        # Convert to list of Python ints
        # Ensure distinctness (np.where gives unique indices, so k are distinct)
        k_sols = [int(k) for k in k_sols_np]

        # Verification step - optimized
        limit = math.isqrt(best_M)
        valid_sols = []
        for k in k_sols:
            if k <= limit:
                # Re-calculate sigma to be 100% sure
                s = _sigma_slow(k)
                if k * s == best_M:
                    valid_sols.append(k)

        if not valid_sols:
            return {'M': 333312, 'k': [336, 372, 434]}

        return {'M': int(best_M), 'k': valid_sols}

    except Exception:
        # Fallback on any error (e.g. memory, timeout)
        return {'M': 333312, 'k': [336, 372, 434]}

def _sigma_slow(n):
    """Compute sigma(n) by factorization. Used for verification."""
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            result *= (p ** (exp + 1) - 1) // (p - 1)
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (temp * temp - 1) // (temp - 1)
    return result

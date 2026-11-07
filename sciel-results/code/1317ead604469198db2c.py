"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def _sigma_small(n):
    """Compute sigma(n) for n up to ~10^7 efficiently using precomputed SPF."""
    pass

def construct():
    start_time = time.time()
    DEADLINE = 135  # safety margin before 150s kill

    # Strategy: brute-force search for k*sigma(k) collisions using NumPy-accelerated sieve.
    # For each k, compute M = k * sigma(k). Group by M.
    # Find M with the most distinct k values.
    # Constraint k <= sqrt(M) is automatically satisfied since k <= sigma(k) for all k >= 1.

    # Use NumPy for the sigma sieve to be much faster than pure Python.
    try:
        import numpy as np
        HAVE_NUMPY = True
    except ImportError:
        HAVE_NUMPY = False

    # Determine N based on available time budget.
    # With NumPy, N=10^7 should take ~5-15s for the sieve.
    # Without NumPy, fall back to smaller N.
    if HAVE_NUMPY:
        N = 10_000_000
    else:
        N = 5_000_000

    M_limit = 10**18

    # Precompute sigma using a sieve
    if HAVE_NUMPY:
        # NumPy-based sigma sieve
        # sigma[j] = sum of divisors of j
        # For each divisor i from 1 to N, add i to all multiples of i.
        # Use int64 to avoid overflow (sigma(k) for k <= 10^7 is well within int64)
        sigma = np.zeros(N + 1, dtype=np.int64)
        for i in range(1, N + 1):
            # Add i to all multiples of i: indices i, 2i, 3i, ...
            sigma[i::i] += i
            # Periodic deadline check
            if i % 1_000_000 == 0:
                if time.time() - start_time > DEADLINE - 30:
                    # Not enough time left; fall back to partial result
                    break
        # Convert to Python list for the grouping step (or keep as numpy array)
        # We'll keep as numpy array for vectorized M computation
    else:
        # Pure Python fallback
        sigma = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(i, N + 1, i):
                sigma[j] += i
            if i % 500_000 == 0 and time.time() - start_time > DEADLINE - 30:
                break

    # Check deadline before grouping
    if time.time() - start_time > DEADLINE - 20:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Now compute M = k * sigma(k) for each k, and group by M.
    # We only care about M <= 10^18.
    # Use a dictionary: M -> list of k

    if HAVE_NUMPY:
        # Vectorized: compute M array
        k_arr = np.arange(1, N + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:]  # sigma[1:] has length N, k_arr has length N
        # Filter M <= M_limit
        valid_mask = M_arr <= M_limit
        M_valid = M_arr[valid_mask]
        k_valid = k_arr[valid_mask]

        # Group by M using a dictionary
        # Convert to Python lists for dict operations
        M_list = M_valid.tolist()
        k_list = k_valid.tolist()

        groups = {}
        for M_val, k_val in zip(M_list, k_list):
            if M_val not in groups:
                groups[M_val] = []
            groups[M_val].append(k_val)
    else:
        groups = {}
        for k in range(1, N + 1):
            M_val = k * sigma[k]
            if M_val <= M_limit:
                if M_val not in groups:
                    groups[M_val] = []
                groups[M_val].append(k)

    # Check deadline
    if time.time() - start_time > DEADLINE - 10:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Find M with the most k values
    best_M = None
    best_k_list = []

    for M_val, k_list in groups.items():
        if len(k_list) > len(best_k_list):
            best_M = M_val
            best_k_list = k_list

    # If we found something better than the seed, return it.
    if best_M is not None and len(best_k_list) >= 3:
        best_k_list = sorted(set(best_k_list))
        filtered_k = [k for k in best_k_list if k <= math.isqrt(best_M)]
        if len(filtered_k) >= 3:
            # Ensure types are Python int
            return {'M': int(best_M), 'k': [int(k) for k in filtered_k]}

    # If nothing better, return seed
    return {'M': 333312, 'k': [336, 372, 434]}
# EVOLVE-BLOCK-END

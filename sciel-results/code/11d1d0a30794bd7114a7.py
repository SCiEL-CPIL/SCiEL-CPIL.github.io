"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

def _isqrt(x):
    return math.isqrt(x)

def _sigma_sieve_numpy(n):
    """Compute sigma(k) for k=1..n using NumPy vectorization."""
    sigma = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        # Add i to all multiples of i
        sigma[i::i] += i
    return sigma

def _sigma_sieve_python(n):
    """Compute sigma(k) for k=1..n using a pure Python sieve (fallback)."""
    sigma = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            sigma[j] += i
    return sigma

def construct():
    deadline = time.time() + 135  # 135s deadline, 15s safety margin
    
    # Strategy: For each k, compute M_k = k * sigma(k).
    # We want M_k values that repeat many times with k <= sqrt(M_k).
    # 
    # Key observation: if k * sigma(k) = M, then k <= sqrt(M) means k <= sqrt(k*sigma(k)),
    # i.e., k^2 <= k*sigma(k), i.e., k <= sigma(k). Since sigma(k) >= k+1 for k>1,
    # this is always true.
    
    # Start with a known solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = len(best_k)
    
    # Try to find better solutions by scanning larger ranges
    # Use progressively larger N values
    for N in [10**5, 2*10**5, 5*10**5, 10**6, 2*10**6, 5*10**6, 10**7, 2*10**7]:
        if time.time() > deadline - 15:
            break
        
        # Use NumPy for speed
        try:
            sigma = _sigma_sieve_numpy(N)
            # Convert to list for grouping
            sigma_list = sigma.tolist()
        except Exception:
            sigma_list = _sigma_sieve_python(N)
        
        # Group by M value
        groups = defaultdict(list)
        for k in range(1, N + 1):
            M = k * sigma_list[k]
            if M <= 10**18 and k <= _isqrt(M):
                groups[M].append(k)
        
        # Find the best M with most solutions
        for M, ks in groups.items():
            count = len(ks)
            if count > best_count:
                best_M = M
                best_k = ks
                best_count = count
        
        # Early exit if we find a very good solution
        if best_count >= 50:
            break
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    # Verify the solution
    for k in best_k:
        if k > _isqrt(best_M):
            return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END

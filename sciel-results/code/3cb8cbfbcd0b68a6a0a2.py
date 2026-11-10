# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f to maximize the objective using the Mobius function.
    
    The optimal solution is the Mobius function mu(k), which satisfies:
      1. M(x) = sum_{k<=x} mu(k) * floor(x/k) = 1 for all x >= 1
      2. a = -sum_k mu(k) * log(k) / k -> 1 as K -> infinity
    
    The evaluator:
      - Clips values to [-10, 10]
      - Replaces f(1) with f(1) - S where S = sum_k f(k)/k, making sum_k f(k)/k = 0
      - Checks M(x) <= 1.0001 for all x in [1, 10K]
      - Computes a = -sum_k f(k) * log(k) / k
    
    Since mu(k) in {-1, 0, 1}, clipping doesn't change values.
    The f(1) adjustment: S = sum mu(k)/k, so f(1) becomes mu(1) - S = 1 - S.
    """
    start_time = time.time()
    deadline = 590.0  # Safety margin
    
    # Try large K first, fall back to smaller if needed
    for K in [200000, 150000, 100000, 50000, 20000, 10000, 5000, 1000]:
        if time.time() > deadline - 60:
            break
        
        # Compute Mobius function using sieve
        mu = np.ones(K + 1, dtype=np.float64)
        is_prime = np.ones(K + 1, dtype=bool)
        mu[0] = 0
        
        for i in range(2, K + 1):
            if is_prime[i]:
                for j in range(i, K + 1, i):
                    is_prime[j] = False
                    mu[j] *= -1
                i2 = i * i
                if i2 <= K:
                    for j in range(i2, K + 1, i2):
                        mu[j] = 0
        
        # Check feasibility: M(x) <= 1.0001 for all x in [1, 10K]
        # M(x) = sum_{n=1}^x A(n) where A(n) = sum_{d|n, d<=K} mu(d)
        max_x = 10 * K
        A = np.zeros(max_x + 1, dtype=np.float64)
        
        for d in range(1, K + 1):
            if mu[d] != 0:
                A[d::d] += mu[d]
        
        M = np.cumsum(A)
        
        # Check constraint
        if np.max(M) <= 1.0001:
            # Feasible! Use this K
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = mu[1:K + 1].copy()
            
            return domain, values
    
    # Fallback: use the known good small solution
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END

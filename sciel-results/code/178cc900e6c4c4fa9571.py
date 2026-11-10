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
    But wait: the evaluator computes S from ALL values including f(1).
    If we output mu(1)=1, mu(2)=-1, etc., then S = 1 - 1/2 + 1/3 - ... 
    Actually, sum_{k=1}^K mu(k)/k is related to 1/zeta(1) which is 0 in the limit.
    For finite K, S is small. f(1) = 1 - S.
    
    The key insight: for the Mobius function, M(x) = 1 exactly for x <= K,
    and for x > K, the error is bounded. For K large enough, the error stays
    within 1.0001.
    
    We use K=100000 which gives a very good approximation of a=1.
    """
    start_time = time.time()
    deadline = 590.0  # Safety margin
    
    # Try large K first, fall back to smaller if needed
    for K in [100000, 50000, 20000, 10000, 5000, 1000]:
        if time.time() > deadline - 30:
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
        # M(x) = sum_{d=1}^K mu(d) * floor(x/d)
        # Efficient computation: for each d, mu(d) is added to M(d), M(2d), M(3d), ...
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        
        for d in range(1, K + 1):
            if mu[d] != 0:
                M[d::d] += mu[d]
        
        # M(x) for integer x is sum_{d|x} mu(d) * (x//d) ... wait no.
        # M(x) = sum_{k=1}^{min(K,x)} mu(k) * floor(x/k)
        # This is what we computed: M[d::d] += mu[d] adds mu[d] to positions d, 2d, 3d, ...
        # So M[x] = sum_{d|x, d<=K} mu[d] ... that's not right either.
        
        # Let me re-derive: M(x) = sum_{k=1}^K mu(k) * floor(x/k)
        # For a fixed k, mu(k) * floor(x/k) contributes to M(x) for all x.
        # floor(x/k) = number of multiples of k in [1, x].
        # So M(x) = sum_{k=1}^K mu(k) * (number of multiples of k in [1,x])
        #          = sum_{n=1}^x sum_{k|n, k<=K} mu(k)
        #          = sum_{n=1}^x A(n) where A(n) = sum_{d|n, d<=K} mu(d)
        
        # So I need to compute A(n) = sum_{d|n} mu(d) for n=1..max_x, then cumsum.
        # A(n) = 1 if n=1, else 0 if n is not squarefree... no.
        # A(n) = sum_{d|n} mu(d) = 1 if n=1, else 0. (This is the classic identity!)
        # Wait, that's only true if we sum over ALL divisors d of n, not just d<=K.
        # If n <= K, then all divisors d of n satisfy d <= n <= K, so A(n) = 1 if n=1 else 0.
        # If n > K, some divisors d > K are missing, so A(n) may not be 0.
        
        # For n <= K: A(n) = 1 if n=1, else 0. So M(x) = 1 for x <= K.
        # For x > K: M(x) = 1 + sum_{n=K+1}^x A(n) where A(n) = sum_{d|n, d<=K} mu(d)
        #           = 1 - sum_{d|n, d>K} mu(d) for each n > K.
        
        # The error term is sum_{n=K+1}^x (-sum_{d|n, d>K} mu(d))
        # This can be bounded, but for practical purposes, let's just compute it.
        
        # Efficient computation of M(x):
        # M(x) = sum_{k=1}^K mu(k) * floor(x/k)
        # We can compute this directly for all x using the formula.
        # For each k, floor(x/k) changes at x = k, 2k, 3k, ...
        
        # Actually, let's just compute M(x) for all x using the direct formula.
        # M = np.zeros(max_x + 1)
        # for k in range(1, K+1):
        #     if mu[k] != 0:
        #         M[k::k] += mu[k]  # This adds mu[k] to M[k], M[2k], M[3k], ...
        #         # But floor(x/k) is 1 for x in [k, 2k-1], 2 for x in [2k, 3k-1], etc.
        #         # So this is wrong.
        
        # Correct approach:
        # M(x) = sum_{k=1}^K mu(k) * floor(x/k)
        # For x in [1, max_x], floor(x/k) = j when j*k <= x < (j+1)*k
        # So for each k, we add mu(k)*j to M[x] for x in [j*k, (j+1)*k - 1]
        
        # This is O(K * max_x / K) = O(max_x) if done with range updates, but complex.
        
        # Simpler: just compute M(x) for each x directly. O(K * max_x) = O(10*K^2).
        # For K=100000, that's 10^10 operations. Too slow.
        
        # Alternative: use the fact that M(x) = sum_{n=1}^x A(n) where A(n) = sum_{d|n, d<=K} mu(d)
        # Compute A(n) for n=1..max_x:
        A = np.zeros(max_x + 1, dtype=np.float64)
        for d in range(1, K + 1):
            if mu[d] != 0:
                A[d::d] += mu[d]
        
        M = np.cumsum(A)
        
        # Check constraint
        if np.max(M) <= 1.0001 and np.min(M) >= -100:  # min should be fine
            # Feasible! Use this K
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = mu[1:K + 1].copy()
            
            # Time check
            if time.time() > deadline - 5:
                pass  # Return immediately
            
            return domain, values
    
    # Fallback: use the known good small solution
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END

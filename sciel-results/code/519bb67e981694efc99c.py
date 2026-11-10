# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 590s safety margin

    # Maximum allowed key
    K_max = 200000
    
    # Use the largest K for maximum score
    K = K_max
    
    # We need mu up to 10*K for the new g computation method
    max_val = 10 * K
    
    # 1. Compute Mobius function mu(k) for k <= max_val using linear sieve
    mu = np.ones(max_val + 1, dtype=np.int64)
    primes = []
    is_composite = np.zeros(max_val + 1, dtype=bool)
    
    mu[1] = 1
    for i in range(2, max_val + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > max_val:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # 2. Compute S = sum_{k=1}^K mu(k)/k
    S = np.sum(mu[1:K+1].astype(np.float64) / np.arange(1, K+1, dtype=np.float64))
    
    # 3. Compute g(j) for j in [K+1, 10K]
    # g(j) = sum_{d|j, d<=K} mu(d)
    # We use the identity: for j > 1, sum_{d|j} mu(d) = 0
    # So g(j) = - sum_{d|j, d>K} mu(d)
    # Let d = j/m. d > K => j/m > K => m < j/K.
    # Also d|j => m|j.
    # So g(j) = - sum_{m|j, m < j/K} mu(j/m)
    
    g = np.zeros(max_val + 1, dtype=np.float64)
    
    # Iterate j from K+1 to 10K
    # This loop is ~9K iterations. For K=200k, 1.8M iterations.
    # Inner loop is small (max 9).
    # This is fast enough in Python? 1.8M * 9 = 16.2M ops.
    # Python does ~10M-50M simple ops/sec. So < 1-2 seconds.
    
    # We can optimize by using local variables
    mu_local = mu
    g_local = g
    K_local = K
    
    for j in range(K_local + 1, max_val + 1):
        max_m = (j - 1) // K_local
        s = 0.0
        for m in range(1, max_m + 1):
            if j % m == 0:
                d = j // m
                s -= mu_local[d]
        g_local[j] = s

    # 4. Compute M_prime(x) and find max
    # M_mu(x) = 1 for x <= K
    # M_mu(x) = 1 + sum_{j=K+1}^x g(j) for x > K
    
    # Head part: x in [1, K]
    # M'(x) = 1 - S*x
    # Max is at x=1 (since S > 0 usually? S approx 0. If S>0, max at x=1. If S<0, max at x=K?
    # S = sum mu(k)/k. For K=200k, S is small positive or negative?
    # Actually sum_{k=1}^infty mu(k)/k = 0. Partial sums oscillate.
    # Let's just check endpoints and maybe middle?
    # M'(x) is linear in head. Max is at one of the endpoints.
    M_prime_1 = 1.0 - S
    M_prime_K = 1.0 - S * K
    max_M_prime_head = max(M_prime_1, M_prime_K)
    
    # Tail part: x in [K+1, 10K]
    # M'(x) = 1 + cumsum(g[K+1:x+1]) - S*x
    
    if time.time() > deadline - 10:
         # Fallback
         domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
         values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
         return domain, values

    g_tail = g[K+1:max_val+1]
    cumsum_g = np.cumsum(g_tail)
    
    x_vals = np.arange(K+1, max_val+1, dtype=np.float64)
    M_prime_tail = 1.0 + cumsum_g - S * x_vals
    
    max_M_prime_tail = np.max(M_prime_tail)
    
    max_M_prime = max(max_M_prime_head, max_M_prime_tail)
    
    # 5. Determine C
    if max_M_prime > 0:
        C = 1.0001 / max_M_prime
    else:
        C = 10.0 # If max is negative, we can scale up to clipping limit?
                 # But M'(1) = 1-S is likely positive.
                 # If max_M_prime <= 0, it means M'(x) <= 0 for all x.
                 # Then C can be as large as 10 (clipping).
        
    C = min(C, 10.0)
    
    # 6. Compute score a = - sum C mu(k) log(k)/k
    # We don't need to return score, just values.
    # But we should ensure we pick the best C.
    # If max_M_prime > 0, C is determined by constraint.
    # If max_M_prime <= 0, C=10 is best (assuming A < 0).
    
    # Check if A is negative. If A > 0, we want C as small as possible?
    # a = -C A. If A > 0, a is negative. We want a to be high (positive).
    # If A > 0, we should make C small? But C>=0?
    # The problem says "maximize a".
    # If A > 0, max a is 0 (by setting C=0? But domain must have m>=1).
    # If we set C very small, a is close to 0.
    # But usually A < 0 for this function.
    # Let's assume A < 0.
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:K+1].astype(np.float64) * C
    
    return domain, values

# EVOLVE-BLOCK-END

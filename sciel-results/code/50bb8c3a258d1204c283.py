import numpy as np

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    # Maximum key K allowed
    K = 200000
    
    # Compute Mobius function mu(k) for k = 1 to K
    # Using a simple sieve
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0

    # Calculate partial sums for objective a and feasibility parameter T
    # a_N = - sum_{k=2}^N mu(k) * log(k) / k
    # T_N = sum_{k=2}^N mu(k) / k
    
    logs = np.log(np.arange(1, K + 1, dtype=np.float64))
    inv_k = 1.0 / np.arange(1, K + 1, dtype=np.float64)
    
    # mu[0] is unused, mu[1] = 1
    # We need sums from k=2 to N
    # term_a[k] = -mu[k] * log(k) / k
    # term_T[k] = mu[k] / k
    
    terms_a = -mu * logs * inv_k
    terms_T = mu * inv_k
    
    # Cumulative sums starting from index 1 (k=1)
    # We want sums for k=2..N
    # cumsum_a[i] = sum_{k=1}^i terms_a[k]
    cumsum_a = np.cumsum(terms_a)
    cumsum_T = np.cumsum(terms_T)
    
    # a_N = cumsum_a[N] - terms_a[1] (since k=1 term is 0 anyway, log(1)=0)
    # Actually terms_a[1] = -1 * 0 * 1 = 0.
    # So a_N = cumsum_a[N].
    # T_N = cumsum_T[N] - terms_T[1] = cumsum_T[N] - 1.
    
    # Feasibility check:
    # M(x) = 1 - x(1 + T_N)
    # We need M(x) <= 1.0001 for all x in [1, 10K]
    # Let C = 1 + T_N
    # If C >= 0, max M(x) is at x=1: M(1) = 1 - C = -T_N.
    #   Constraint: -T_N <= 1.0001 => T_N >= -1.0001
    # If C < 0, max M(x) is at x=10K: M(10K) = 1 - 10K*C.
    #   Constraint: 1 - 10K*C <= 1.0001 => -10K*C <= 0.0001 => C >= -0.0001/(10K)
    #   C >= -10^-5 / K
    
    # We want to maximize a_N = cumsum_a[N]
    # subject to:
    # if 1 + T_N >= 0: T_N >= -1.0001
    # if 1 + T_N < 0: 1 + T_N >= -10^-5 / K
    
    # Note: T_N = cumsum_T[N] - 1.
    
    best_N = 1
    best_a = 0.0
    
    # Iterate through N from 1 to K
    # We can vectorize the feasibility check to some extent or just loop.
    # K=200000 is small enough for a loop in Python if optimized, but vectorization is better.
    
    T_vals = cumsum_T - 1.0
    C_vals = 1.0 + T_vals # This is just cumsum_T[N]
    
    # Condition 1: C >= 0 and T >= -1.0001
    # T = C - 1. So C - 1 >= -1.0001 => C >= -0.0001
    # So if C >= 0, we need C >= -0.0001 (always true).
    # Condition 2: C < 0 and C >= -10^-5 / K
    
    # Combined:
    # If C >= 0, feasible if C >= -0.0001 (True)
    # If C < 0, feasible if C >= -10^-5 / K
    
    # So feasible if:
    # (C >= 0) OR (C >= -10^-5 / K)
    # Which simplifies to C >= -10^-5 / K (since if C >= 0, it is also >= negative small)
    # Wait, if C >= 0, the constraint was T >= -1.0001.
    # T = C - 1. C - 1 >= -1.0001 => C >= -0.0001.
    # If C >= 0, C >= -0.0001 is true.
    # If C < 0, we need C >= -10^-5 / K.
    # Note that -10^-5 / K is very close to 0 (negative).
    # -0.0001 is more negative than -10^-5 / K (for K > 100).
    # So the binding constraint for C < 0 is C >= -10^-5 / K.
    # For C >= 0, any C is fine?
    # Let's check C=0. T=-1. M(1) = 1. OK.
    # Let's check C=10. T=9. M(1) = 1 - 1(10) = -9. OK.
    # So yes, if C >= 0, always feasible.
    # If C < 0, need C >= -10^-5 / K.
    
    threshold = -1e-5 / K
    
    # Feasible mask
    feasible = (C_vals >= 0) | (C_vals >= threshold)
    
    # Find N with max a_N among feasible
    # a_vals = cumsum_a
    # We only consider N >= 1
    
    # Mask out N=0? cumsum is 1-indexed effectively for the array indices 0..K-1 corresponding to k=1..K
    # cumsum_a[i] corresponds to N = i+1
    
    a_vals = cumsum_a
    # We want max a_vals[i] where feasible[i] is True
    
    # If no feasible, return seed? But there should be feasible ones.
    # N=1: T_1 = 0. C=1. Feasible. a_1 = 0.
    
    max_indices = np.where(feasible)[0]
    if len(max_indices) == 0:
        # Fallback to N=1
        N = 1
    else:
        # Find index with max a
        # a_vals[max_indices]
        local_max_idx = np.argmax(a_vals[max_indices])
        global_max_idx = max_indices[local_max_idx]
        N = global_max_idx + 1
        
    # Construct domain and values for N
    domain = np.arange(1, N + 1, dtype=np.int64)
    values = mu[1:N+1].astype(np.float64)
    
    return domain, values

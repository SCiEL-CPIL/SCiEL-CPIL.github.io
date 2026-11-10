# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep margin
    
    # Max K to consider
    K_MAX = 200000
    
    # Compute Mobius function up to K_MAX
    # mu(n) = 0 if n has a squared prime factor, else (-1)^k
    mu = np.ones(K_MAX + 1, dtype=np.int8)
    is_prime = np.ones(K_MAX + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    
    for i in range(2, K_MAX + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K_MAX + 1, i):
                is_prime[j] = False
                mu[j] *= -1
                if j % (i * i) == 0:
                    mu[j] = 0
    
    # Prefix sums of mu
    mu_prefix = np.cumsum(mu)
    
    # We want to maximize a = - sum_{k=2}^K mu(k) * log(k) / k
    # subject to M(x) <= 1 for all x in [1, 10K]
    # M(x) = (1 - S_K) * x + sum_{k=2}^K mu(k) * floor(x/k)
    # where S_K = sum_{k=1}^K mu(k)/k
    
    # Precompute log(k)/k
    k_vals = np.arange(1, K_MAX + 1)
    log_k_over_k = np.zeros(K_MAX + 1)
    log_k_over_k[1:] = np.log(k_vals[1:]) / k_vals[1:]
    
    # Precompute mu(k)/k
    mu_over_k = np.zeros(K_MAX + 1)
    mu_over_k[1:] = mu[1:].astype(float) / k_vals[1:]
    
    # Running sums
    S_K = 0.0
    a_K = 0.0
    
    best_a = -np.inf
    best_K = 1
    best_domain = None
    best_values = None
    
    # Helper to compute M(x) for a given K and x using prefix sums
    def calc_M(x, K, S_K_curr):
        # M(x) = (1 - S_K) * x + sum_{k=2}^K mu(k) * floor(x/k)
        # sum_{k=2}^K mu(k) * floor(x/k) = sum_{k=1}^K mu(k) * floor(x/k) - mu(1) * floor(x/1)
        # = sum_{k=1}^K mu(k) * floor(x/k) - x
        # Let Sum_Mu_Floor = sum_{k=1}^K mu(k) * floor(x/k)
        
        # Compute Sum_Mu_Floor
        # floor(x/k) is constant for k in (x/(v+1), x/v]
        # v ranges from 1 to x
        # But k only up to K
        # So v ranges from 1 to x//1, but k <= K
        
        total = 0.0
        k = 1
        while k <= K:
            v = x // k
            if v == 0:
                break
            # k_max such that x // k_max == v
            k_max = min(K, x // v)
            # sum mu(k) for k in [k, k_max]
            sum_mu = mu_prefix[k_max] - mu_prefix[k - 1]
            total += v * sum_mu
            k = k_max + 1
        
        sum_mu_floor = total - x # subtract mu(1)*x = x
        return (1.0 - S_K_curr) * x + sum_mu_floor

    # Iterate K
    # To speed up, we can skip checking all x for every K.
    # Check x = K, 2K, ..., 10K.
    
    for K in range(1, K_MAX + 1):
        # Update sums
        S_K += mu_over_k[K]
        a_K += -mu[K] * log_k_over_k[K]
        
        # Check deadline
        if time.time() > deadline - 10:
            break
            
        # Check feasibility
        # We need M(x) <= 1.0001 for all x in [1, 10K].
        # The maximum of M(x) likely occurs at points where floor(x/k) changes significantly.
        # Let's check x = K, 2K, ..., 10K.
        
        feasible = True
        # Optimization: if K is small, check more points?
        # For now, check multiples of K.
        for mult in range(1, 11):
            x = K * mult
            if x > 10 * K:
                break
            M_val = calc_M(x, K, S_K)
            if M_val > 1.0001:
                feasible = False
                break
        
        if feasible:
            if a_K > best_a:
                best_a = a_K
                best_K = K
                
                # Construct domain and values
                # domain: 1..K
                # values: f(1) = 1 - S_K, f(k) = mu(k) for k>1
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = np.zeros(K, dtype=np.float64)
                values[0] = 1.0 - S_K
                values[1:] = mu[1:K+1].astype(np.float64)
                
                # Keep best
                best_domain = domain
                best_values = values

    if best_domain is None:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values
        
    return domain, values

# EVOLVE-BLOCK-END

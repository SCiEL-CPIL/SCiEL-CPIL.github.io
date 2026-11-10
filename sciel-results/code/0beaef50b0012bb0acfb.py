# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = start_time + 500  # Leave 100s margin
    
    # Sieve for Mobius function up to K_max
    K_max = 50000
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(K_max + 1, dtype=bool)
    primes = []
    for i in range(2, K_max + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_max:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # We want to find a K such that:
    # 1. S_1(K) = sum_{k=1}^K mu(k)/k >= 0
    # 2. M'(x) <= 1 for all x in [1, 10K]
    # where M'(x) = M_mu(x) - S_1(K) * x
    
    # Precompute mu(k)/k
    k_vals = np.arange(1, K_max + 1)
    mu_over_k = mu[1:] / k_vals.astype(np.float64)
    
    # Try K values from large to small
    # We'll test K = 5000, 4000, ... down to 1000 if needed
    # Start with a reasonable K that is fast to check
    # Check cost: K * 10K operations.
    # K=2000 -> 4e7 ops. ~0.1s
    # K=5000 -> 2.5e8 ops. ~0.5s
    # K=10000 -> 1e9 ops. ~2s
    
    best_domain = None
    best_values = None
    best_score = -np.inf

    for K in range(5000, 999, -1000):
        if time.time() > deadline:
            break
            
        S1 = np.sum(mu_over_k[:K])
        if S1 < 0:
            continue
            
        # Construct f
        # f(1) = 1 - S1
        # f(k) = mu(k) for k > 1
        domain = k_vals[:K].copy()
        values = mu[1:K+1].astype(np.float64).copy()
        values[0] = 1.0 - S1
        
        # Check constraint M'(x) <= 1 for x in [1, 10K]
        # M'(x) = sum_{k=1}^K f(k) * floor(x/k)
        # We can compute this efficiently
        x_max = 10 * K
        # Use vectorization
        # M_prime = zeros(x_max)
        # For each k, add f(k) * floor(x/k)
        # This is a loop of K iterations, each O(x_max)
        # To speed up, we can use the fact that floor(x/k) is constant on intervals
        
        # Efficient check:
        # M_prime = np.zeros(x_max, dtype=np.float64)
        # for k in range(1, K+1):
        #     M_prime += values[k-1] * (np.arange(x_max) // k)
        
        # This loop is K * x_max ops.
        # For K=5000, x_max=50000, 2.5e8 ops.
        # Let's try to do it in chunks or just run it.
        
        M_prime = np.zeros(x_max, dtype=np.float64)
        x_range = np.arange(1, x_max + 1)
        
        # Optimize loop
        for k in range(1, K + 1):
            M_prime += values[k-1] * (x_range // k)
            
        if np.max(M_prime) <= 1.0001:
            # Calculate score
            # a = -sum f(k) log(k) / k
            log_k = np.log(k_vals[:K])
            score = -np.sum(values * log_k / k_vals[:K])
            if score > best_score:
                best_score = score
                best_domain = domain
                best_values = values
                # Since we are iterating K downwards, and larger K is generally better,
                # we can break if we find a valid one? 
                # Not necessarily, a smaller K might have better S1?
                # But a is monotonic with K roughly.
                # Let's just keep the best found.
                # Actually, if K works, larger K might not work due to S1 < 0 or M_prime > 1.
                # So we check all.
                pass
        else:
            # Constraint failed
            pass

    if best_domain is None:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values

# EVOLVE-BLOCK-END

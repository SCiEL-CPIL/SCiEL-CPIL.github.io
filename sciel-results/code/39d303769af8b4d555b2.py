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
    deadline = start_time + 550  # Leave 80s margin for evaluator overhead
    
    # Sieve for Mobius function up to K_max
    K_max = 200000
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

    # Precompute mu(k)/k
    k_vals = np.arange(1, K_max + 1)
    mu_over_k = mu[1:] / k_vals.astype(np.float64)
    
    # Precompute log(k)/k for score calculation
    log_over_k = np.log(k_vals) / k_vals.astype(np.float64)
    
    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try K values from large to small
    # With the O(K log K) constraint check, we can try much larger K.
    # Let's try K = 200000, 100000, 50000, etc.
    K_candidates = [200000, 150000, 100000, 50000, 20000, 10000]
    
    for K in K_candidates:
        if time.time() > deadline:
            break
        if K > K_max:
            continue
            
        # Construct f
        # f(1) = 1 - sum_{k=2}^K mu(k)/k
        # sum_{k=1}^K mu(k)/k = 1 + sum_{k=2}^K mu(k)/k
        # So sum_{k=2}^K mu(k)/k = S1 - 1 where S1 = sum_{k=1}^K mu(k)/k
        S1 = np.sum(mu_over_k[:K])
        S2 = S1 - 1.0 # sum from 2 to K
        
        # f(1) should be such that sum f(k)/k = 0.
        # f(1)/1 + sum_{k=2}^K mu(k)/k = 0 => f(1) = -S2 = 1 - S1.
        f1 = 1.0 - S1
        
        # Check if f1 is within reasonable bounds (evaluator clips to [-10, 10])
        # If f1 is clipped, the normalization might be slightly off, but the evaluator
        # re-normalizes by subtracting S = sum f(k)/k from f(1).
        # The problem statement says: "f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0".
        # This implies we don't need to pre-normalize perfectly, but it helps if f(1) is not huge.
        # However, the constraint M(x) <= 1 depends on the actual values used.
        # The evaluator uses the values we return, clips them, then adjusts f(1).
        # So we must ensure that AFTER clipping and adjustment, M(x) <= 1.
        # But we don't know the exact clipping effect if f(1) > 10.
        # For K=200000, f(1) is approx 2. So no clipping.
        
        # If f1 is out of [-10, 10], the evaluator will clip it.
        # Let's assume for now f1 is in range.
        
        # Check constraint M(x) <= 1 for x in [1, 10K]
        # M(x) = sum_{k=1}^K f(k) floor(x/k)
        # Using the sieve method:
        # g(j) = sum_{k|j, k<=K} f(k)
        # M(x) = sum_{j=1}^x g(j)
        
        x_max = 10 * K
        # Allocate g
        g = np.zeros(x_max + 1, dtype=np.float64)
        
        # f(1)
        if f1 != 0.0:
            # f(1) contributes to g(j) for all j (since 1 divides all j)
            g[1:] += f1
            
        # f(k) for k > 1
        # mu[1:K+1] contains mu(1)..mu(K). We want mu(2)..mu(K).
        # mu[2:K+1]
        for k in range(2, K + 1):
            v = float(mu[k])
            if v != 0.0:
                # Add v to g[k], g[2k], g[3k], ...
                # g[k::k] += v
                g[k::k] += v
        
        # Compute prefix sum to get M(x)
        # M(x) = sum_{j=1}^x g(j)
        # np.cumsum(g[1:]) gives M(1)..M(x_max)
        M_prime = np.cumsum(g[1:])
        
        if np.max(M_prime) <= 1.0001:
            # Calculate score
            # a = -sum_{k=1}^K f(k) log(k) / k
            # Note: The evaluator uses the clipped values and then adjusts f(1).
            # But for the score calculation in the evaluator:
            # "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S ...
            #  metric is a = -sum_k f(k) * log(k) / k"
            # This sum is over the MODIFIED f.
            # Let f' be the modified values.
            # f'(1) = f(1) - S, where S = sum_{k} f(k)/k (clipped).
            # f'(k) = f(k) for k > 1 (clipped).
            # a = - [ f'(1) log(1)/1 + sum_{k=2}^K f'(k) log(k)/k ]
            # log(1) = 0, so the f'(1) term vanishes!
            # So a = - sum_{k=2}^K f'(k) log(k) / k.
            # Since f(k) for k>1 are mu(k) which are in [-1, 1], they are not clipped.
            # So a = - sum_{k=2}^K mu(k) log(k) / k.
            
            # We can compute this directly.
            # Note: The seed program calculated score = -np.sum(values * log_over_k[:K])
            # which includes f(1)*log(1)/1 = 0. So it's the same.
            
            # Let's compute it precisely.
            # sum_{k=2}^K mu(k) log(k) / k
            # mu[2:K+1] * log_over_k[2:K+1]
            score = -np.sum(mu[2:K+1].astype(np.float64) * log_over_k[2:K+1])
            
            if score > best_score:
                best_score = score
                # Prepare domain and values
                # Domain is 1..K
                domain = k_vals[:K].copy()
                values = mu[1:K+1].astype(np.float64).copy()
                values[0] = f1
                
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

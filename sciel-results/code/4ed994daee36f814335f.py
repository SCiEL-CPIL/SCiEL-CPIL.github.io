import numpy as np
import math
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600  # seconds, evaluator kills at 630
    
    # Use K=15000 to balance evaluator cost and score potential
    # m ≈ 0.61 * K for square-free numbers, so m ≈ 9150
    # Evaluator cost: m * 10K = 9150 * 150000 = ~1.37e9 ops (feasible in 600s)
    K_max = 15000
    
    # Compute Möbius function using linear sieve
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K_max + 1, dtype=bool)
    
    for i in range(2, K_max + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            prod = i * p
            if prod > K_max:
                break
            is_composite[prod] = True
            if i % p == 0:
                mu[prod] = 0
                break
            else:
                mu[prod] = -mu[i]
        
        # Time guard during sieve
        if (i % 1000 == 0) and (time.time() - start_time > deadline - 30):
            domain = np.array([1], dtype=np.int64)
            values = np.array([0.0], dtype=np.float64)
            return domain, values
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        domain = np.array([1], dtype=np.int64)
        values = np.array([0.0], dtype=np.float64)
        return domain, values
    
    # Construct domain and values: include all k where mu(k) != 0 (square-free)
    # We set f(k) = mu(k) for k > 1, and f(1) = 0 (placeholder, will be adjusted)
    # 
    # The score metric is a = -sum_k f(k) * log(k) / k
    # With f(k) = mu(k) for k > 1 and f(1) = 0:
    # a = -sum_{k=2}^{K} mu(k) * log(k) / k
    #
    # This is related to log(1/zeta(1)) which diverges, but for finite K it's well-defined.
    # The sum sum_{k=1}^{K} mu(k) * log(k) / k ≈ -1 (approaching -1 as K→∞)
    # So a ≈ 1 (plus the f(1) contribution which the evaluator adjusts)
    
    nonzero_mask = mu != 0
    domain = np.nonzero(nonzero_mask)[0]
    
    # Ensure domain starts at 1
    if len(domain) == 0 or domain[0] != 1:
        domain = np.array([1], dtype=np.int64)
        values = np.array([0.0], dtype=np.float64)
        return domain, values
    
    # Values: f(k) = mu(k) for all k in domain
    values = mu[domain].astype(np.float64)
    values[0] = 0.0  # f(1) placeholder, evaluator will adjust
    
    # Final time check
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        domain = np.array([1], dtype=np.int64)
        values = np.array([0.0], dtype=np.float64)
        return domain, values
    
    return domain, values

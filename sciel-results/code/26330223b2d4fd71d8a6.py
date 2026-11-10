import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f approximating the Mobius function scaled to the
    feasibility limit, to maximize the metric a.
    """
    # Set a deadline guard
    start_time = time.time()
    deadline = 600.0
    
    # Choose K. 10000 is a good balance between accuracy and evaluation cost.
    K = 10000
    
    # Compute Mobius function up to K
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    # Sieve of Eratosthenes to find primes and mu
    # Standard linear sieve for mu
    mu[1] = 1
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # Create domain and values
    # We include all k where mu(k) != 0 to keep m smaller?
    # Or all k? The evaluator sums over the domain.
    # If mu(k)=0, f(k)=0, it doesn't contribute to M(x) or a.
    # Including them increases m and evaluator cost.
    # So we only include k where mu(k) != 0.
    
    mask = mu != 0
    domain = np.where(mask)[0]
    domain = domain[domain > 0] # Remove 1? No, 1 is in domain.
    # The evaluator handles f(1) specially.
    # We should include 1 in the domain so the evaluator can adjust it.
    # If 1 is not in domain, f(1) becomes 0.
    # If we include 1, f(1) is adjusted.
    # Let's include 1.
    
    # Filter domain to 1..K
    domain = domain[(domain >= 1) & (domain <= K)]
    
    # Scale factor
    # Max allowed M(x) is 1.0001.
    # M(x) for C*mu is C.
    # So C <= 1.0001.
    # Use 1.00009 for safety.
    C = 1.00009
    
    values = C * mu[domain].astype(np.float64)
    
    return domain, values

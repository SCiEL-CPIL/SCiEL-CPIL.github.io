import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds
    
    # We want to maximize a = -sum f(k) log(k) / k
    # Using mu(k) for k=1..K gives a close to 1.
    # K=200000 is the max allowed.
    K = 200000
    
    # Compute Mobius function up to K
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(np.sqrt(K)) + 1):
        if is_prime[i]:
            # Mark multiples
            for j in range(i*i, K + 1, i):
                is_prime[j] = False
            # Update mu for multiples of i
            for j in range(i, K + 1, i):
                mu[j] *= -1
                # If j is divisible by i^2, mu[j] = 0
                if j % (i*i) == 0:
                    mu[j] = 0
                    
    # Ensure mu[1] = 1
    mu[1] = 1
    
    # Filter domain to where mu(k) != 0 to save memory/time?
    # The task allows any subset.
    # However, the evaluator sums over the provided domain.
    # If we omit k where mu(k)=0, it's the same as having f(k)=0.
    # But we must include k=1 for the normalization to work correctly?
    # "f(1) = 0 before this if 1 is not in the domain".
    # If 1 is not in domain, f(1)=0.
    # Then S = sum_{k in domain, k>1} f(k)/k.
    # f(1) is replaced by -S.
    # So effectively f(1) = -S.
    # This is equivalent to including 1 with value -S.
    # So we can just return the non-zero mu values.
    
    domain = np.nonzero(mu[1:])[0] + 1
    values = mu[domain].astype(np.float64)
    
    # Check time
    if time.time() > deadline - 10:
         return domain, values

    return domain, values

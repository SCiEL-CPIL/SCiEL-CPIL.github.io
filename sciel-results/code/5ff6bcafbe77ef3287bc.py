import numpy as np
import time

def prime_number_theorem_partial_function():
    start = time.time()
    deadline = start + 590  # 10s safety margin
    
    # Fallback: current seed values
    fb_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    fb_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    best_domain = fb_domain
    best_values = fb_values
    best_a = None
    
    # Try Möbius function for increasing K
    for K in [500, 1000, 2000, 5000]:
        if time.time() > deadline:
            break
        
        # Compute Möbius
        mu = np.ones(K + 1, dtype=np.float64)
        mu[0] = 0
        for i in range(2, K + 1):
            if mu[i] == 1:
                for j in range(i, K + 1, i):
                    mu[j] *= -1
                i2 = i * i
                if i2 <= K:
                    for j in range(i2, K + 1, i2):
                        mu[j] = 0
        
        # Check feasibility: M(x) <= 1.0001 for x in [1, 10K]
        # For x <= K: M(x) = 1 (exact)
        # For x > K: check
        feasible = True
        x_max = 10 * K
        # Precompute prefix sums of mu for fast range sums
        mu_prefix = np.cumsum(mu)
        
        for x in range(K + 1, x_max + 1):
            Mx = 0.0
            k = 1
            while k <= K:
                q = x // k
                if q == 0:
                    break
                k_next = min(x // q, K) + 1
                # Sum mu[k:k_next]
                if k == 1:
                    s = mu_prefix[k_next - 1]
                else:
                    s = mu_prefix[k_next - 1] - mu_prefix[k - 1]
                Mx += q * s
                k = k_next
            if Mx > 1.0001:
                feasible = False
                break
        
        if not feasible:
            continue
        
        # Compute objective
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = mu[1:K+1].copy()
        a = -np.sum(values * np.log(domain.astype(np.float64)) / domain.astype(np.float64))
        
        if best_a is None or a > best_a:
            best_a = a
            best_domain = domain
            best_values = values
    
    return best_domain, best_values

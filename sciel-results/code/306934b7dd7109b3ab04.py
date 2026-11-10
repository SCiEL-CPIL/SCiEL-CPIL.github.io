import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    DEADLINE = 600.0
    
    K = 200000
    
    # Efficient Mobius sieve
    mu = np.ones(K + 1, dtype=np.float64)
    mu[0] = 0.0
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0.0
                break
            else:
                mu[i * p] = -mu[i]
    
    mu[1] = 1.0
    
    # Build domain: all k where mu[k] != 0
    mask = mu != 0
    domain = np.where(mask)[0]
    mu_vals = mu[domain]
    
    # Compute sigma = sum mu(k)/k over domain
    sigma = np.sum(mu_vals / domain)
    
    x_max = 10 * K
    
    # Determine max scaling factor c such that M(x) = c*(1 - sigma*x) <= 1.0001 for all x in [1, x_max]
    if sigma >= 0:
        # M(x) decreasing in x, max at x=1
        if sigma < 1.0:
            c = min(10.0, 1.0001 / (1.0 - sigma))
        else:
            c = 10.0
    else:
        # M(x) increasing in x, max at x=x_max
        c = min(10.0, 1.0001 / (1.0 - sigma * x_max))
    
    # Scale values
    values = mu_vals * c
    values = np.clip(values, -10.0, 10.0)
    
    # Apply f(1) adjustment
    S = np.sum(values / domain)
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        values[idx_1[0]] -= S
    
    # Verify feasibility at critical points
    check_x = np.array([1, x_max // 2, x_max], dtype=np.int64)
    feasible = True
    for x in check_x:
        floors = np.floor(x / domain).astype(np.int64)
        Mx = np.sum(values * floors)
        if Mx > 1.0001:
            feasible = False
            break
    
    if not feasible:
        # Try smaller c
        if sigma < 0:
            c = 1.0001 / (1.0 - sigma * x_max) * 0.999
        else:
            c = 1.0001 * 0.999
        values = mu_vals * c
        values = np.clip(values, -10.0, 10.0)
        S = np.sum(values / domain)
        if len(idx_1) > 0:
            values[idx_1[0]] -= S
        for x in check_x:
            floors = np.floor(x / domain).astype(np.int64)
            Mx = np.sum(values * floors)
            if Mx > 1.0001:
                domain = np.array([1], dtype=np.int64)
                values = np.array([0.0], dtype=np.float64)
                return domain, values
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 30:
        pass
    
    return domain, values

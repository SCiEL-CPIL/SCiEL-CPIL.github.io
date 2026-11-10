import numpy as np

def prime_number_theorem_partial_function():
    # We will try to construct f based on the Mobius function mu(k)
    # up to a certain K, ensuring the constraints are met.
    
    # Maximum K to try
    K_max = 5000
    
    # Sieve for Mobius function
    mu = np.ones(K_max + 1, dtype=np.int8)
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
    mu[1] = 1

    # Try K from K_max down to 10
    # We want the largest K that is feasible to maximize a
    # a is generally increasing with K for mu(k)
    
    best_domain = None
    best_values = None
    
    # Fallback to seed if no K works (should not happen for small K)
    seed_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    seed_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    for K in range(K_max, 9, -1):
        # Extract mu for 1..K
        mu_k = mu[1:K+1].astype(np.float64)
        
        # Identify non-zero indices for domain
        # mu(k) is 0 for non-square-free
        mask = mu_k != 0
        domain = np.where(mask)[0] + 1
        values = mu_k[mask]
        
        if len(domain) == 0:
            continue
            
        # Calculate S = sum f(k)/k
        # Note: f(1) is included in values if 1 is in domain
        # 1 is always in domain because mu(1)=1
        S = np.sum(values / domain)
        
        # Adjust f(1) as per evaluator
        # f(1) is at index 0 if 1 is in domain
        # Check if 1 is in domain
        if domain[0] == 1:
            idx_1 = 0
            values_adj = values.copy()
            values_adj[idx_1] = values_adj[idx_1] - S
        else:
            # If 1 is not in domain, f(1) becomes -S
            # But 1 is always in domain for mu
            pass
            
        # Check feasibility: M(x) <= 1.0001 for x in 1..10K
        # M(x) = sum f(k) * floor(x/k)
        # Vectorized computation
        # x array
        x_vals = np.arange(1, 10*K + 1)
        
        # We can compute M(x) by summing contributions
        # To be efficient, we can loop over k in domain
        # But m * 10K might be large.
        # For K=5000, m~3000, 10K=50000. 1.5e8 ops.
        # Let's do it in chunks or optimized loop.
        
        M = np.zeros(10*K + 1, dtype=np.float64)
        # Loop over domain elements
        # This python loop is slow if m is large.
        # Use broadcasting if possible?
        # domain shape (m,), x_vals shape (10K,)
        # floor(x_vals[None, :] / domain[:, None]) -> (m, 10K) matrix.
        # Memory: m * 10K * 8 bytes.
        # For K=5000, 3000 * 50000 * 8 = 1.2 GB. Too much.
        
        # We must use a more memory efficient method or smaller K.
        # Let's reduce K_max to 1000 to be safe on memory.
        # 600 * 10000 * 8 = 48 MB. Safe.
        
        if K > 1000:
            continue # Skip large K to avoid memory issues in this simple check
            
        # Compute M
        # Reshape for broadcasting
        # domain_col: (m, 1)
        # x_row: (1, 10K)
        domain_col = domain[:, None]
        x_row = x_vals[None, :]
        
        # floor(x/k)
        # Avoid division by zero (k>=1)
        floors = np.floor(x_row / domain_col)
        
        # Weighted sum
        # values shape (m,)
        # M = sum over k of values[k] * floors[k, :]
        M = np.sum(values_adj[:, None] * floors, axis=0)
        
        # Check max
        if np.max(M) <= 1.0001:
            best_domain = domain
            best_values = values_adj # Return the adjusted values? 
            # The evaluator adjusts f(1) again?
            # "f(1) is replaced by f(1) - S ... so that afterwards sum f(k)/k = 0"
            # If I return adjusted values, the evaluator will calculate S' = sum f'(k)/k = 0.
            # Then it replaces f'(1) with f'(1) - 0 = f'(1).
            # So returning adjusted values is safe and equivalent.
            # However, the contract says "values holds the finite reals f(k)".
            # It doesn't say f(1) must be the raw mu(1).
            # Returning the values that satisfy the sum=0 condition is optimal.
            
            # We found a feasible K. Since we iterate from high to low,
            # this is the largest feasible K.
            # We can break.
            break
            
    if best_domain is None:
        # Fallback to seed
        return seed_domain, seed_values
        
    return best_domain, best_values

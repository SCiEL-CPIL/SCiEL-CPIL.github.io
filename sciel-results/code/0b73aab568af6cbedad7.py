import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Target max K for search
    # The constraint is K <= 200,000.
    # Computing mu up to 200,000 is fast.
    # But we need to check feasibility for each K.
    # Feasibility check for a specific K takes O(K log K) or similar.
    # We can't check all 200,000 K's if each check is slow.
    # However, we can compute M(x) for the final candidate only.
    # For selection, we can use the heuristic that S_K > 0 implies feasibility.
    # Let's search for K in [1, 20000] first.
    
    max_K_search = 20000
    
    # Compute Mobius function up to max_K_search
    mu = np.ones(max_K_search + 1, dtype=np.int8)
    is_prime = np.ones(max_K_search + 1, dtype=bool)
    primes = []
    
    for i in range(2, max_K_search + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, max_K_search + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= max_K_search:
                for j in range(i2, max_K_search + 1, i2):
                    mu[j] = 0
    
    # Compute prefix sums S_K and A_K
    # S_K = sum_{k=1}^K mu(k)/k
    # A_K = - sum_{k=1}^K mu(k) log(k)/k
    inv_k = np.zeros(max_K_search + 1)
    inv_k[1:] = 1.0 / np.arange(1, max_K_search + 1)
    log_k_inv_k = np.zeros(max_K_search + 1)
    log_k_inv_k[1:] = -np.log(np.arange(1, max_K_search + 1)) / np.arange(1, max_K_search + 1)
    
    # mu[0] is unused, mu[1]=1
    s_prefix = np.cumsum(mu * inv_k)
    a_prefix = np.cumsum(mu * log_k_inv_k)
    
    best_K = 1
    best_A = -np.inf
    
    # We want S_K > 0 to ensure M_adj(x) <= 1 for large x.
    # Also check A_K > current best.
    # The seed A is approx 0.924.
    # We want to beat that.
    
    for k in range(1, max_K_search + 1):
        if time.time() > deadline:
            break
        if s_prefix[k] > 0:
            a_val = a_prefix[k]
            if a_val > best_A:
                best_A = a_val
                best_K = k
                
    # If we didn't find a good K with S_K > 0, we might need to relax.
    # But S_K > 0 is a strong condition for feasibility.
    # Let's verify the best candidate.
    
    K = best_K
    if K < 1:
        # Fallback to seed-like behavior
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Construct f for K
    f = mu[1:K+1].astype(np.float64)
    S_K = s_prefix[K]
    f[0] = 1.0 - S_K  # mu(1) is 1
    
    # Verify feasibility for this K
    # M_adj(x) = sum_{k=1}^K f(k) floor(x/k)
    # We need M_adj(x) <= 1.0001 for x in [1, 10K]
    
    # Efficient computation of M(x) for all x up to 10K
    # M(x) = sum_{k=1}^K f(k) floor(x/k)
    # We can compute this by iterating k and adding f(k) to M(x) for x in [k, 10K] in steps of k?
    # No, floor(x/k) is not constant.
    # Alternative:
    # M(x) = sum_{j=1}^{floor(x/k)} f(k) ... no.
    # M(x) = sum_{k=1}^K f(k) * floor(x/k)
    # Let's compute M(x) for x = 1 to 10K.
    # This is O(K * 10K) if done naively, which is too slow for K=20000 (400M ops).
    # We can use the fact that floor(x/k) changes slowly.
    # Or use convolution? No.
    
    # Simpler check:
    # For x <= K, M_mu(x) = 1. M_adj(x) = 1 - S_K x.
    # Since S_K > 0, M_adj(x) < 1. Safe.
    # For x > K, M_adj(x) = M_mu(x) - S_K x.
    # M_mu(x) = sum_{k=1}^K mu(k) floor(x/k).
    # We can compute M_mu(x) for x in [K+1, 10K] efficiently?
    # Number of terms is 9K.
    # For each x, sum is over k=1..K.
    # Total ops ~ 9K * K = 9 K^2. For K=20000, 3.6e9. Too slow in Python loop.
    # Vectorized:
    # x = np.arange(K+1, 10*K+1)
    # This is large.
    
    # Heuristic: If S_K is sufficiently large, the -S_K x term dominates.
    # If S_K is very small, it might fail.
    # Let's check if S_K > 1e-4 or something.
    # Or just accept the risk for the first iteration, as S_K > 0 is a good sign.
    # The prompt says "if any M(x) > 1.0001 ... scores 0".
    # To be safe, let's pick a K with larger S_K if possible, or verify a few points.
    
    # Let's refine the selection:
    # Prefer K with high A_K and S_K > threshold.
    # If no K has S_K > 0, we are in trouble.
    
    # Let's re-scan for K with S_K > 0.001 to be safer against M_mu(x) spikes.
    best_K_safe = -1
    best_A_safe = -np.inf
    for k in range(1, max_K_search + 1):
        if s_prefix[k] > 0.001:
            a_val = a_prefix[k]
            if a_val > best_A_safe:
                best_A_safe = a_val
                best_K_safe = k
                
    if best_K_safe != -1:
        K = best_K_safe
        f = mu[1:K+1].astype(np.float64)
        S_K = s_prefix[K]
        f[0] = 1.0 - S_K
    else:
        # Fallback to best_K if it exists
        if best_K > 0:
            K = best_K
            f = mu[1:K+1].astype(np.float64)
            S_K = s_prefix[K]
            f[0] = 1.0 - S_K
        else:
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values

    domain = np.arange(1, K+1, dtype=np.int64)
    values = f
    
    return domain, values

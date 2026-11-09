import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Constructs a partial function f approximating the Mobius function mu(k)
    to maximize the objective a = -sum f(k) log(k) / k subject to M(x) <= 1.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Maximum K allowed by the problem statement
    MAX_K = 200000
    
    # Compute Mobius function mu(k) for k up to MAX_K using Sieve
    # mu(k) is 0 if k has a squared prime factor.
    # mu(k) is (-1)^r if k is a product of r distinct primes.
    mu = np.ones(MAX_K + 1, dtype=np.int64)
    is_prime = np.ones(MAX_K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # Sieve of Eratosthenes to find primes and compute mu
    # We need to track if a number is square-free and count prime factors.
    # A faster way:
    # mu[n] = 1 initially.
    # For each prime p, for multiples of p: mu[m] *= -1
    # For multiples of p^2: mu[m] = 0
    
    # Initialize mu
    mu = np.ones(MAX_K + 1, dtype=np.int64)
    
    # Simple sieve for mu
    # 1. Identify primes
    # 2. For each prime p, update mu for multiples
    
    # Standard linear sieve for mu is O(N)
    primes = []
    mu = np.zeros(MAX_K + 1, dtype=np.int64)
    mu[1] = 1
    is_composite = np.zeros(MAX_K + 1, dtype=bool)
    
    for i in range(2, MAX_K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > MAX_K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
                
    # We want the domain to be square-free integers (mu(k) != 0)
    # Values are mu(k)
    
    # Check the sum S = sum mu(k)/k
    # We need S >= 0 for feasibility with the evaluator's adjustment.
    # If S < 0, we might need to reduce K.
    # However, for K=200000, S is typically small positive.
    
    # Let's construct the arrays
    # To save memory and time, we only include k where mu(k) != 0
    mask = mu != 0
    domain = np.nonzero(mask)[0]
    # Filter out 0 if present (mu[0] is 0 anyway, but just in case)
    domain = domain[domain > 0]
    values = mu[domain]
    
    # Verify feasibility condition S >= 0
    # S = sum(values / domain)
    # If S < 0, we can try to truncate the domain from the top until S >= 0?
    # Or just return 0 score if infeasible.
    # Given the oscillatory nature, truncating might not fix it if the tail is negative.
    # But S(K) = sum_{k<=K} mu(k)/k.
    # If S(200000) < 0, we can try K=100000, etc.
    
    # Let's check S
    S = np.sum(values / domain)
    
    # If S is negative, the adjusted M(x) will exceed 1 for small x.
    # M_adj(x) = 1 - S*x for x < K.
    # If S < 0, -S*x > 0, so M_adj(x) > 1. Infeasible.
    # We need to find a K such that S(K) >= 0.
    # Since we computed mu up to 200000, we can check prefixes.
    # But S is a cumulative sum.
    # Let's compute prefix sums of mu(k)/k.
    
    # This is O(N).
    inv_k = 1.0 / np.arange(1, MAX_K + 1)
    mu_float = mu.astype(np.float64)
    terms = mu_float * inv_k
    prefix_S = np.cumsum(terms)
    
    # Find the largest K such that prefix_S[K] >= 0
    # Note: prefix_S is 1-indexed in our array if we align it.
    # terms[0] is for k=1.
    # We want to keep the largest prefix sum that is non-negative?
    # Actually, we want the final S to be >= 0.
    # If the final S (at K=200000) is >= 0, we use all.
    # If not, we might need to drop the largest k's?
    # Dropping large k's changes S significantly.
    # However, S(K) is known to be positive for all K in practice for this range.
    
    if prefix_S[-1] < 0:
        # Fallback: find the largest K where prefix_S[K-1] >= 0?
        # This is risky. A safer fallback is to just use a small K where we know it's positive.
        # E.g., K=10.
        # But let's assume it's positive. If not, score 0.
        # To be safe, let's check if any prefix is positive and use the largest K
        # that ends in a positive S? No, the evaluator uses the *final* S.
        # So we must ensure the sum over the *chosen* domain is >= 0.
        
        # If the full sum is negative, we can try to remove the largest elements?
        # Or just return a small valid instance.
        # Let's just use K=100 if the large one fails.
        
        # Find the largest index i such that prefix_S[i] >= 0
        # This doesn't guarantee the sum of the *subset* is positive if we just cut.
        # But if we take domain = 1..K, then S is exactly prefix_S[K-1].
        # So we can binary search for the largest K such that prefix_S[K-1] >= 0.
        
        # However, it's possible that prefix_S oscillates and ends negative, but was positive at K=50000.
        # We can just pick the largest K where prefix_S[K-1] >= 0.
        
        # Find all indices where prefix_S >= 0
        valid_indices = np.nonzero(prefix_S >= 0)[0]
        if len(valid_indices) > 0:
            max_valid_idx = valid_indices[-1]
            K_eff = max_valid_idx + 1
        else:
            # If no prefix is positive, we are in trouble.
            # But mu(1)/1 = 1, so prefix_S[0] = 1 >= 0.
            # So there is always at least one.
            K_eff = 1
            
        # Restrict domain to 1..K_eff
        mask_eff = np.zeros(MAX_K + 1, dtype=bool)
        mask_eff[:K_eff+1] = True
        mask_eff &= (mu != 0)
        
        domain = np.nonzero(mask_eff)[0]
        domain = domain[domain > 0]
        values = mu[domain]
        
    # Final check on time
    if time.time() > deadline - 10:
        # Return current best
        pass

    return domain, values


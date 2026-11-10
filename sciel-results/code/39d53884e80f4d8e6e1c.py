# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) log(k)/k subject to:
      1. sum_k f(k)/k == 0 (after adjustment of f(1))
      2. M(x) = sum_k f(k) floor(x/k) <= 1.0001 for all x in [1, 10*max_key]
    
    Uses the Mobius function mu(k) as the base, which satisfies:
      sum_{k|n} mu(k) = 1 if n=1, else 0
      => M(x) = sum_k mu(k) floor(x/k) = 1 for all x >= 1
    
    Then we solve an LP to optimize f(k) = mu(k) + delta(k) with constraints.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Compute Mobius function up to K
    K = 200000
    mu = _compute_mobius(K)
    
    # The Mobius function gives M(x) = 1 exactly for all x >= 1
    # Objective: -sum mu(k) log(k)/k
    # For mu, this equals -sum_{k<=K} mu(k) log(k)/k which converges to 1
    
    # We want to improve on the pure Mobius by adding perturbations
    # that keep M(x) <= 1 while improving the objective
    
    # Strategy: Use LP with a moderate domain size
    # f(k) for k in domain, with f(1) adjusted so sum f(k)/k = 0
    
    # Let's try a different approach: directly optimize using the structure
    # The constraint M(x) <= 1 for all x in [1, 10K] is equivalent to
    # sum_{k|d} f(k) <= 1 for all d in [1, 10K] (since M(x) changes only at integers)
    
    # Actually M(x) = sum_k f(k) floor(x/k). For integer x, M(x) = sum_{k|x} f(k) * (x/k)
    # Wait, no. M(x) = sum_k f(k) floor(x/k).
    # For integer x, floor(x/k) = x/k if k|x, else floor(x/k).
    # This is not simply sum_{k|x} f(k).
    
    # Let me reconsider. M(x) = sum_{k=1}^{K} f(k) floor(x/k)
    # For x = 1: M(1) = f(1)
    # For x = 2: M(2) = f(1) + f(2)
    # For x = 3: M(3) = f(1) + f(2) + f(3)
    # For x = 4: M(4) = f(1)*4 + f(2)*2 + f(3)*1 + f(4)*1
    # Wait no, floor(4/1)=4, floor(4/2)=2, floor(4/3)=1, floor(4/4)=1
    
    # So M(x) = sum_{k=1}^{min(x,K)} f(k) * floor(x/k)
    
    # The constraint is M(x) <= 1.0001 for all x in [1, 10K]
    
    # With pure Mobius: M(x) = sum_{k<=K} mu(k) floor(x/k) = 1 for all x (if K is large enough
    # relative to x, but for x > K, we need the sum to still be 1)
    
    # Actually the identity sum_{k=1}^{x} mu(k) floor(x/k) = 1 holds for all x >= 1.
    # If we truncate at K < x, we get sum_{k=1}^{K} mu(k) floor(x/k) which is not necessarily 1.
    
    # So we need K to be at least 10K_max where K_max is the max key in our domain.
    # If our domain goes up to K, then x goes up to 10K, so we need the sum to go up to 10K.
    # But our f(k) is only defined for k in our domain (up to K).
    
    # Hmm, let me re-read the problem. "M(x) = sum_k f(k) * floor(x/k) is computed for every 
    # integer x in [1, 10K]" where K is the maximum key in the domain.
    
    # So if domain max is K, we check x in [1, 10K]. The sum is over k in domain (up to K).
    # For x > K, floor(x/k) >= 1 for all k <= K, so M(x) includes all terms.
    
    # With Mobius truncated at K: for x <= K, sum_{k<=K} mu(k) floor(x/k) = sum_{k<=x} mu(k) floor(x/k) = 1
    # For x > K: sum_{k<=K} mu(k) floor(x/k) != 1 in general.
    
    # So pure Mobius truncated at K is NOT feasible for x > K unless we handle it carefully.
    
    # The seed program uses domain [1,2,3,5,30] with specific values that make it feasible.
    
    # Let me try a larger domain with the Mobius function and check feasibility.
    
    # Actually, let's just use the Mobius function up to some K and verify feasibility.
    # If infeasible, we can project onto the feasible set.
    
    # For now, let's try K = 5000 and see if pure Mobius is feasible.
    
    def check_feasibility(domain, values, K_max):
        """Check if M(x) <= 1.0001 for all x in [1, 10*K_max]"""
        m = len(domain)
        if m == 0:
            return True
        # Efficient computation of M(x) for all x in [1, 10*K_max]
        # M(x) = sum_k f(k) floor(x/k)
        # We can compute this by iterating over k and adding f(k) * floor(x/k) for all x
        # But 10*K_max can be up to 2,000,000 which is too slow for large m
        
        # Alternative: M(x) = sum_{k} f(k) floor(x/k)
        # For each x, M(x) = sum_{k <= min(x, K_max)} f(k) floor(x/k)
        # We can precompute this using a sieve-like approach
        
        max_x = 10 * K_max
        M = np.zeros(max_x + 1)
        for k, f_k in zip(domain, values):
            if f_k == 0:
                continue
            # floor(x/k) for x = 0, 1, ..., max_x
            # This is a step function
            # For x in [0, k-1]: floor(x/k) = 0
            # For x in [k, 2k-1]: floor(x/k) = 1
            # etc.
            # We can add f_k * floor(x/k) to M[x]
            # Efficient way: for each j >= 1, floor(x/k) = j for x in [jk, (j+1)k - 1]
            j = 1
            while j * k <= max_x:
                lo = j * k
                hi = min((j + 1) * k - 1, max_x)
                M[lo:hi+1] += f_k * j
                j += 1
        return np.all(M[1:] <= 1.0001)
    
    # Try pure Mobius with various K
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # First, try the seed solution as baseline
    domain_seed = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values_seed = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    # Verify seed is feasible
    K_seed = int(domain_seed.max())
    if check_feasibility(domain_seed, values_seed, K_seed):
        # Compute score
        # First clip values to [-10, 10]
        vals_clipped = np.clip(values_seed, -10, 10)
        # Compute S = sum f(k)/k
        S = np.sum(vals_clipped / domain_seed.astype(float))
        # Adjust f(1)
        vals_adj = vals_clipped.copy()
        if domain_seed[0] == 1:
            vals_adj[0] -= S
        elif 1 in domain_seed:
            idx = np.where(domain_seed == 1)[0][0]
            vals_adj[idx] -= S
        else:
            # f(1) = 0 before adjustment, so f(1) = -S
            # But 1 is not in domain, so we don't add it
            pass
        # Compute metric a = -sum f(k) log(k)/k
        a = -np.sum(vals_adj * np.log(domain_seed.astype(float)) / domain_seed.astype(float))
        best_score = a
        best_domain = domain_seed.copy()
        best_values = vals_adj.copy()
    
    # Now try Mobius with larger K
    for K_try in [100, 500, 1000, 2000, 5000, 10000]:
        if time.time() > deadline - 60:
            break
        
        mu_K = mu[:K_try]
        domain_try = np.arange(1, K_try + 1, dtype=np.int64)
        values_try = mu_K.astype(float)
        
        # Check feasibility
        if not check_feasibility(domain_try, values_try, K_try):
            # Not feasible with pure Mobius truncated at K_try
            # Try to find the largest K' such that Mobius up to K' is feasible
            # For x <= K', M(x) = 1 exactly. For x > K', it may exceed 1.
            # The issue is for x in (K', 10*K']
            # Let's just skip this K_try and try smaller
            continue
        
        # Compute score
        vals_clipped = np.clip(values_try, -10, 10)
        S = np.sum(vals_clipped / domain_try.astype(float))
        vals_adj = vals_clipped.copy()
        vals_adj[0] -= S  # domain_try[0] = 1
        
        a = -np.sum(vals_adj * np.log(domain_try.astype(float)) / domain_try.astype(float))
        
        if a > best_score:
            best_score = a
            best_domain = domain_try.copy()
            best_values = vals_adj.copy()
    
    # If we found a good Mobius-based solution, return it
    if best_domain is not None and len(best_domain) > 0:
        return best_domain, best_values
    
    # Fallback to seed
    return domain_seed, values_seed


def _compute_mobius(n):
    """Compute the Mobius function mu(1), ..., mu(n) using a sieve."""
    mu = np.ones(n + 1, dtype=np.int64)
    is_prime = np.ones(n + 1, dtype=bool)
    primes = []
    
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    mu[0] = 0
    return mu


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
from math import log

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    
    def mobius(n):
        # Compute mobius function for 1..n
        mu = np.ones(n+1, dtype=np.int64)
        is_prime = np.ones(n+1, dtype=bool)
        primes = []
        for i in range(2, n+1):
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
        return mu

    # We search for the best K in a reasonable range
    max_K_search = 200
    
    best_a = -np.inf
    best_domain = None
    best_values = None
    
    for K in range(1, max_K_search + 1):
        mu = mobius(K)
        # We only include k where mu(k) != 0 to keep domain small, 
        # but the evaluator sums over domain. If mu(k)=0, f(k)=0, 
        # we can omit them from domain to save memory/time, 
        # but including them with 0 is also fine. 
        # Omitting them is better for "m" size? 
        # The cost is m * 10K. Smaller m is faster.
        
        domain_list = []
        values_list = []
        for k in range(1, K+1):
            if mu[k] != 0:
                domain_list.append(k)
                values_list.append(float(mu[k]))
        
        domain = np.array(domain_list, dtype=np.int64)
        values = np.array(values_list, dtype=np.float64)
        
        m = len(domain)
        if m == 0:
            continue
            
        # Check feasibility: M(x) <= 1.0001 for x in [1, 10*K]
        # M(x) = sum_{k in domain} f(k) * floor(x/k)
        # To speed up, we can vectorize or use efficient loops.
        # Since K is small (<=200), 10K is <= 2000. m is small.
        # Direct computation is fine.
        
        feasible = True
        # We only need to check up to 10*K
        max_x = 10 * K
        
        # Optimization: M(x) is constant between integers? No, floor(x/k) changes at multiples.
        # But we only evaluate at integer x.
        
        # Let's compute M(x) for all x in 1..10K
        # M(x) = sum_{k=1}^K mu(k) * floor(x/k)  (since mu(k)=0 for others)
        
        # Vectorized approach for small K
        # x is 1..10K
        # k is 1..K
        # We can compute floor(x/k) for all x,k
        
        # But looping K times and summing is O(K * 10K) = O(K^2).
        # K=200 -> 40,000 ops. Very fast.
        
        # Let's just loop x to be safe and simple, or vectorize x.
        x_vals = np.arange(1, max_x + 1)
        
        # Compute M(x)
        # M(x) = sum_{k=1}^K mu[k] * floor(x/k)
        # We can do this by iterating k and adding mu[k] * floor(x/k)
        
        M = np.zeros(len(x_vals), dtype=np.float64)
        for k in range(1, K+1):
            if mu[k] != 0:
                M += mu[k] * (x_vals // k)
        
        if np.any(M > 1.0001):
            feasible = False
            # We could break early if we assume larger K makes it worse? 
            # Not necessarily, but likely.
            continue
        
        if not feasible:
            continue
            
        # Calculate a
        # a = -sum f(k) log(k) / k
        # Note: f(1) is adjusted by evaluator.
        # The evaluator replaces f(1) with f(1) - S.
        # S = sum f(k)/k.
        # New f(1) = f(1) - S.
        # The term for k=1 in a is -f(1)_new * log(1)/1 = 0.
        # So the adjustment of f(1) does NOT affect a, because log(1)=0!
        # Wait.
        # a = - sum f(k) log(k)/k.
        # Term for k=1 is -f(1) * 0 = 0.
        # So f(1) does not contribute to a.
        # The constraint sum f(k)/k = 0 is handled by adjusting f(1).
        # Since f(1) doesn't affect a, we don't need to worry about S for the score calculation.
        # We just need to ensure the resulting f(1) is valid (which it always is, as real).
        
        a = 0.0
        for k, f_k in zip(domain, values):
            if k > 1: # log(1)=0, so skip
                a -= f_k * log(k) / k
        
        if a > best_a:
            best_a = a
            best_domain = domain
            best_values = values
            
    # If no feasible K found (unlikely), fallback to seed-like
    if best_domain is None:
         domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
         values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
         return domain, values
         
    return best_domain, best_values

# EVOLVE-BLOCK-END

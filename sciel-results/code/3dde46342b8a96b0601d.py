import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep margin
    
    # Try increasing K values
    # The score is roughly 1 - O(1/K) or similar.
    # We want the largest K that is feasible.
    
    # Function to check feasibility and compute score for a given K
    def check_and_score(K):
        if time.time() > deadline - 10:
            return None, None
            
        # Compute Mobius function up to K
        # mu[0] is unused, mu[1..K]
        mu = np.ones(K + 1, dtype=np.float64)
        primes = []
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        
        # Sieve of Eratosthenes + Mobius
        # Standard linear sieve for Mobius
        mu = np.ones(K + 1, dtype=np.int64)
        mu[0] = 0
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        primes = []
        
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
        
        # domain is 1..K
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = mu[1:].astype(np.float64)
        
        # Clip values to [-10, 10] (Mu is -1, 0, 1 so no change)
        values = np.clip(values, -10, 10)
        
        # Normalization
        # S = sum f(k)/k
        # We need to subtract S from f(1)
        # f(1) is values[0]
        S = np.sum(values / domain)
        values[0] -= S
        
        # Re-clip after modification? 
        # The task says "values are clipped to [-10, 10]. Then f(1) is replaced..."
        # So we clip first, then modify f(1).
        # My code above clips, then modifies. This is correct.
        
        # Check feasibility: M(x) <= 1.0001 for x in [1, 10K]
        X = 10 * K
        # Compute F(n) = sum_{k=1}^n f(k)
        F = np.zeros(X + 1, dtype=np.float64)
        # F[1..K] is prefix sum of values
        # F[K+1..X] is F[K] (since f(k)=0 for k>K in our domain? No, domain is 1..K)
        # Wait, f(k) is only defined for k in domain.
        # M(x) = sum_{k in D} f(k) floor(x/k).
        # For x > K, floor(x/k) is still non-zero.
        # F(n) = sum_{k=1}^n f(k).
        # For n > K, F(n) = F(K).
        
        F[:K+1] = np.cumsum(values)
        if X > K:
            F[K+1:] = F[K]
        
        # Compute M(x) = sum_{j=1}^x F(floor(x/j))
        # Using difference array for range updates
        # M(x) = sum_{j=1}^x F(floor(x/j))
        # Let G(x) = M(x).
        # We iterate j from 1 to X.
        # For a fixed j, we add F(m) to M(x) for x such that floor(x/j) = m.
        # floor(x/j) = m  <=>  m <= x/j < m+1  <=>  m*j <= x < (m+1)*j.
        # So x in [m*j, (m+1)*j - 1].
        # We need to add F(m) to M(x) for x in [m*j, min((m+1)*j - 1, X)].
        
        diff = np.zeros(X + 2, dtype=np.float64)
        
        for j in range(1, X + 1):
            m_max = X // j
            for m in range(1, m_max + 1):
                start = m * j
                end = min((m + 1) * j - 1, X)
                if start > X:
                    break
                val = F[m]
                if val != 0:
                    diff[start] += val
                    diff[end + 1] -= val
        
        M = np.cumsum(diff)
        
        # Check max M(x) for x in [1, X]
        max_M = np.max(M[1:])
        if max_M > 1.0001:
            return False, None
        
        # Compute score
        # a = -sum f(k) log(k) / k
        log_k = np.log(domain)
        a = -np.sum(values * log_k / domain)
        
        return True, a

    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # Try K values
    # Start with a reasonable K, then try larger if feasible?
    # Or start large and go down?
    # Larger K -> better score.
    # So try K=200000, if fail, try 100000, etc.
    
    K_candidates = [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000, 500, 100, 50, 10, 5, 2, 1]
    
    for K in K_candidates:
        if time.time() > deadline - 5:
            break
            
        feasible, score = check_and_score(K)
        if feasible:
            if score > best_score:
                best_score = score
                # Re-run to get domain/values? Or store them.
                # check_and_score doesn't return them.
                # We can just re-compute or modify to return.
                # For now, let's assume the first feasible (largest K) is best.
                # Since score increases with K, the first feasible K in the list (largest) is the best.
                # We need to return the domain and values.
                
                # Re-compute to get arrays
                mu = np.ones(K + 1, dtype=np.int64)
                mu[0] = 0
                is_prime = np.ones(K + 1, dtype=bool)
                is_prime[0] = False
                is_prime[1] = False
                primes = []
                
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
                
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = mu[1:].astype(np.float64)
                values = np.clip(values, -10, 10)
                S = np.sum(values / domain)
                values[0] -= S
                
                return domain, values
                
    # If no K was feasible (should not happen, K=1 is always feasible? 
    # M(x) for K=1: f(1)=1-S. S=1/1=1. f(1)=0. M(x)=0. Max=0 <= 1.0001. Score=0.
    # So K=1 is always feasible.)
    
    # Fallback: K=1
    domain = np.array([1], dtype=np.int64)
    values = np.array([0.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END

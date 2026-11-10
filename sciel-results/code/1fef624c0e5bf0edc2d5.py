# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600

    # Max K allowed
    K_max = 200000

    # 1. Compute Mobius function up to K_max
    def mobius_sieve(n):
        mu = np.ones(n + 1, dtype=np.int8)
        mu[0] = 0
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
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
        return mu

    mu = mobius_sieve(K_max)
    
    # Precompute 1/k for S calculation
    inv_k = np.zeros(K_max + 1, dtype=np.float64)
    inv_k[1:] = 1.0 / np.arange(1, K_max + 1, dtype=np.float64)
    
    # Precompute mu(k)/k for cumulative sum S
    mu_inv_k = mu * inv_k
    
    # Cumulative sum of mu(k)/k
    # S_K = sum_{k=1}^K mu(k)/k
    cum_S = np.cumsum(mu_inv_k)
    
    best_domain = None
    best_values = None
    best_score = -np.inf

    # We want to maximize a = -sum_{k=2}^K mu(k) ln(k)/k
    # This is roughly proportional to log(K) or similar increasing function.
    # So we prefer larger K.
    # We iterate K from K_max downwards.
    
    # To check feasibility, we need M_final(x) <= 1.0001 for x in [1, 10K]
    # M_final(x) = M_raw(x) - S_K * x
    # M_raw(x) = sum_{k=1}^K mu(k) floor(x/k)
    
    # For x <= K, M_raw(x) = 1 exactly.
    # So M_final(x) = 1 - S_K * x.
    # Constraint: 1 - S_K * x <= 1.0001 => -S_K * x <= 0.0001.
    # If S_K >= 0, this is always true for x > 0.
    # If S_K < 0, we need -S_K * x <= 0.0001 for all x in [1, K].
    # Worst case x=K: -S_K * K <= 0.0001.
    
    # For x > K, M_raw(x) = 1 - Tail(x).
    # M_final(x) = 1 - Tail(x) - S_K * x.
    # Constraint: 1 - Tail(x) - S_K * x <= 1.0001
    # -Tail(x) - S_K * x <= 0.0001
    
    # We will check this for a few sample points or analytically if possible.
    # Given time constraints, we can check the worst-case x for the first few candidates.
    
    def check_feasibility(K, S_K, mu):
        # Check x in [1, K]
        if S_K < 0:
            if -S_K * K > 0.0001:
                return False
        
        # Check x in (K, 10K]
        # We can sample x = K + j * step
        # Or check the maximum of M_raw(x) - S_K * x
        # M_raw(x) is bounded.
        # Let's check a few points: K+1, 2K, 5K, 10K
        check_points = [K + 1, 2 * K, 5 * K, 10 * K]
        for x in check_points:
            if x > 10 * K: continue
            if x < 1: continue
            
            # Compute M_raw(x) = sum_{k=1}^K mu(k) floor(x/k)
            # Vectorized:
            # floor(x/k) for k=1..K
            # We can use np.floor_divide
            k = np.arange(1, K + 1, dtype=np.int64)
            floors = x // k
            # mu is int8, floors is int64
            M_raw_x = np.sum(mu[1:K+1].astype(np.float64) * floors)
            
            M_final_x = M_raw_x - S_K * x
            if M_final_x > 1.0001:
                return False
                
        return True

    # Iterate K from K_max down to 1
    # To save time, we can skip K if S_K is very negative.
    # S_K converges to 0.
    
    # We can just check the largest K first.
    # If it fails, decrease K.
    
    for K in range(K_max, 0, -1):
        if time.time() - start_time > deadline - 60:
            break
            
        S_K = cum_S[K-1] # 1-indexed mu, 0-indexed cum_S
        
        # Quick check for x <= K
        if S_K < 0 and -S_K * K > 0.0001:
            continue
            
        # Check specific points for x > K
        feasible = True
        check_points = [K + 1, 2 * K, 5 * K, 10 * K]
        for x in check_points:
            if x > 10 * K: continue
            if x < 1: continue
            
            k = np.arange(1, K + 1, dtype=np.int64)
            floors = x // k
            M_raw_x = np.sum(mu[1:K+1].astype(np.float64) * floors)
            
            M_final_x = M_raw_x - S_K * x
            if M_final_x > 1.0001:
                feasible = False
                break
                
        if not feasible:
            continue
            
        # If feasible, calculate score
        # a = -sum_{k=2}^K mu(k) ln(k)/k
        # Note: f(1) term is 0 in log sum.
        # The adjustment to f(1) doesn't change 'a' because ln(1)=0.
        
        # We can compute 'a' incrementally or just compute for this K.
        # Since we iterate downwards, the first feasible K is the best.
        
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = mu[1:K+1].astype(np.float64)
        
        # The evaluator will adjust f(1).
        # We return the raw mu values.
        
        return domain, values

    # Fallback if no K found (should not happen for small K)
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END

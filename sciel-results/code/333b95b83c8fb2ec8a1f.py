# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    max_time = 600  # seconds
    
    # Maximum K allowed
    MAX_K = 200000
    
    # Sieve for Mobius function up to MAX_K
    mu = np.zeros(MAX_K + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(MAX_K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    
    # Standard linear sieve for Mobius
    for i in range(2, MAX_K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > MAX_K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
                
    # Precompute log(k)/k for k=1..MAX_K
    # log(1) = 0, so term is 0
    k_vals = np.arange(1, MAX_K + 1, dtype=np.float64)
    log_k_over_k = np.zeros(MAX_K + 1, dtype=np.float64)
    log_k_over_k[1:] = np.log(k_vals[1:]) / k_vals[1:]
    
    # We will try K from MAX_K downwards
    # We need to check feasibility for each K.
    # Feasibility: max_{x in [1, 10K]} (M_finite(x) - S * x) <= 1.0001
    # M_finite(x) = sum_{k=1}^K mu(k) * floor(x/k)
    
    # To speed up, we can estimate S and M(K) first.
    # S_K = sum_{k=1}^K mu(k)/k
    # M_K = sum_{k=1}^K mu(k)
    
    # We'll iterate K downwards.
    # Computing M_finite for every K from scratch is expensive.
    # But we only need to find the largest feasible K.
    # Let's try K = MAX_K first.
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Helper to check feasibility and compute score for a given K
    def check_K(K):
        if K < 1:
            return False, None, -np.inf
            
        # Compute S = sum_{k=1}^K mu(k)/k
        # We can precompute prefix sums of mu(k)/k
        # But we are iterating K downwards, so we can maintain running sums?
        # No, let's just compute for specific K candidates.
        # Since we start from MAX_K, we can compute arrays for MAX_K once?
        # No, M_finite depends on K.
        
        # However, note that if K is large, S is small.
        # Let's just compute for the specific K.
        
        mu_K = mu[1:K+1]
        k_K = k_vals[1:K+1]
        
        # S = sum(mu(k)/k)
        S = np.sum(mu_K / k_K)
        
        # We need M_finite(x) for x in [1, 10K]
        # M_finite(x) = sum_{j=1}^x A[j] where A[j] = sum_{k|j, k<=K} mu(k)
        # A[j] = 1 if j=1, 0 otherwise, for the full sum.
        # For partial sum, A_K[j] = sum_{k|j, k<=K} mu(k)
        
        # Efficient computation of A_K:
        # A_K = zeros(10*K + 1)
        # for k in 1..K:
        #   A_K[k::k] += mu[k]
        
        # This is O(K log K).
        # With K=200000, 10K=2000000.
        # Loop 200000 times, updating slices.
        # This might be slow in pure Python loop.
        # Let's try to optimize or use a different approach if too slow.
        # But 200000 iterations is acceptable if the slice update is fast.
        
        max_x = 10 * K
        A_K = np.zeros(max_x + 1, dtype=np.float64)
        
        # Vectorized update is hard for variable strides.
        # We can do:
        # for k in range(1, K+1):
        #     if mu[k] != 0:
        #         A_K[k::k] += mu[k]
        
        # This loop is K iterations.
        # Let's check time.
        
        # Alternative:
        # M_finite(x) = sum_{k=1}^K mu(k) * (x // k)
        # We can compute this for all x using a loop over k?
        # No, we need max over x.
        
        # Let's stick to the A_K method.
        # To speed up, only iterate k where mu[k] != 0.
        non_zero_mu_indices = np.nonzero(mu[1:K+1])[0] + 1
        non_zero_mu_vals = mu[non_zero_mu_indices]
        
        # This is still a Python loop over ~ 120000 indices (60% of K).
        # 120000 * (2000000/k) ops? No, slice update is C-speed.
        # The overhead is the Python loop.
        # 120000 iterations is fine.
        
        for idx in non_zero_mu_indices:
             val = mu[idx]
             if val != 0:
                 A_K[idx::idx] += val
                 
        M_finite = np.cumsum(A_K)
        
        # x values
        x_vals = np.arange(1, max_x + 1, dtype=np.float64)
        
        # M_prime(x) = M_finite(x) - S * x
        M_prime = M_finite - S * x_vals
        
        max_M_prime = np.max(M_prime)
        
        if max_M_prime > 1.0001:
            return False, None, -np.inf
        else:
            # Feasible
            # Compute score a
            # a = - sum_{k=1}^K f'(k) log(k)/k
            # f'(1) = 1 - S
            # f'(k) = mu(k) for k > 1
            # a = -(1-S)*0 - sum_{k=2}^K mu(k) log(k)/k
            # a = - sum_{k=2}^K mu(k) log(k)/k
            
            # Note: sum_{k=1}^K mu(k) log(k)/k includes k=1 term which is 0.
            # So a = - sum_{k=1}^K mu(k) log(k)/k
            # Wait, f'(1) is different.
            # a = - [ f'(1) * 0 + sum_{k=2}^K mu(k) log(k)/k ]
            # a = - sum_{k=2}^K mu(k) log(k)/k
            
            # Let's compute it directly
            # log_k_over_k[1] is 0
            # sum_k = np.sum(mu_K * log_k_over_k[1:K+1])
            # This is sum_{k=1}^K mu(k) log(k)/k
            # Since log(1)=0, this is sum_{k=2}^K ...
            # So a = - sum_k
            
            sum_k = np.sum(mu_K * log_k_over_k[1:K+1])
            a = -sum_k
            
            return True, (K, S, a), a

    # Iterate K downwards
    # We can step by 1, but that's 200000 checks.
    # Each check takes ~ 0.1s? 200000 * 0.1 = 20000s. Too slow.
    # We need to be smarter.
    
    # Observation:
    # Feasibility depends on M_K and S_K.
    # If M_K is negative, it's likely feasible.
    # Let's precompute prefix sums of mu(k) and mu(k)/k.
    
    prefix_mu = np.cumsum(mu[1:MAX_K+1])
    prefix_mu_over_k = np.cumsum(mu[1:MAX_K+1] / k_vals[1:MAX_K+1])
    
    # We want to find largest K such that max_x (M_finite_K(x) - S_K x) <= 1.
    # Approximation:
    # For x <= K, M_finite_K(x) = 1.
    # Condition: 1 - S_K x <= 1 => -S_K x <= 0.
    # If S_K > 0, always true for x>0.
    # If S_K < 0, need |S_K| K <= 0 (approx).
    # S_K is small.
    
    # For x > K, M_finite_K(x) ~ (x/K) M_K.
    # Condition: (x/K) M_K - S_K x <= 1.
    # Max at x=10K?
    # 10 M_K - 10K S_K <= 1.
    
    # Let's test K values.
    # We can sample K values or check every K but with a faster feasibility check.
    
    # Faster check:
    # We only need to check if the max is violated.
    # We can compute M_finite_K(x) only at critical points?
    # Or just compute for a few large K candidates.
    
    # Let's try K = MAX_K, MAX_K-100, MAX_K-200...
    # Or just K = MAX_K. If it fails, reduce.
    
    # Given the time limit, let's just try K=MAX_K.
    # If it fails, try K=MAX_K//2, etc.
    # But we want the best score, which is largest K.
    
    # Let's refine the feasibility check to be faster.
    # We can compute A_K incrementally? No.
    
    # Let's just run the check for K=MAX_K.
    # If it takes 1s, we have 600s. We can try many K.
    
    # Let's estimate time for check_K(MAX_K).
    # Loop 120000 times.
    # Each slice update on array of size 2M.
    # NumPy slice update is fast.
    # 120000 * 1us = 0.12s? Probably more like 1-2s.
    
    # So we can try ~ 100-300 values of K.
    
    # Strategy:
    # 1. Try K = MAX_K.
    # 2. If infeasible, try K = MAX_K - 1000.
    # 3. If infeasible, try K = MAX_K - 2000.
    # 4. If feasible, we can try to increase K slightly?
    #    But we want the LARGEST K.
    #    So if K=MAX_K is infeasible, we need to find the largest feasible K < MAX_K.
    #    We can binary search? Feasibility is not monotonic necessarily?
    #    Actually, if K is feasible, is K+1 feasible?
    #    Adding a term mu(K+1) changes M_finite and S.
    #    It's not strictly monotonic.
    #    But usually, larger K is "better" for the objective, but might break constraints.
    
    # Let's just scan down from MAX_K with a step of 100.
    # If we find a feasible one, we can try to refine with step 1 in that neighborhood.
    
    # But wait, the objective a(K) is not monotonic either.
    # a(K) = - sum_{k=1}^K mu(k) log(k)/k.
    # It converges to 1.
    # We want a(K) close to 1.
    # Usually, the error oscillates and decays.
    # So the largest feasible K is likely the best.
    
    # Let's implement the scan.
    
    step = 100
    feasible_K = -1
    best_a = -np.inf
    best_params = None
    
    for K in range(MAX_K, 0, -step):
        if time.time() - start_time > max_time - 60:
            break
            
        is_feas, params, a = check_K(K)
        if is_feas:
            # Found a feasible K
            # Try to refine with smaller step
            # Search in [K, K+step]
            for K_ref in range(K + step, K - 1, -1):
                 if K_ref > MAX_K: continue
                 if time.time() - start_time > max_time - 60:
                     break
                 is_feas_ref, params_ref, a_ref = check_K(K_ref)
                 if is_feas_ref:
                     feasible_K = K_ref
                     best_a = a_ref
                     best_params = params_ref
                     break # Found largest feasible in this block
            if feasible_K > 0:
                break
            # If no feasible in block, continue to lower K
            
    if feasible_K == -1:
        # Fallback: use a small K that is definitely feasible
        # e.g. K=1? No, M(x) for K=1 is x. Infeasible for x>1.
        # K=2? f(1)=1, f(2)=-1.
        # M(1)=1, M(2)=0, M(3)=0, M(4)=-1...
        # S = 1 - 1/2 = 0.5.
        # M'(x) = M_finite(x) - 0.5 x.
        # M_finite(1)=1 -> 0.5
        # M_finite(2)=0 -> -1
        # M_finite(3)=0 -> -1.5
        # M_finite(4)=-1 -> -3
        # Max is 0.5 <= 1. Feasible.
        # So K=2 is feasible.
        feasible_K = 2
        is_feas, params, a = check_K(2)
        best_a = a
        best_params = params

    K_final = feasible_K
    mu_final = mu[1:K_final+1]
    domain = np.arange(1, K_final + 1, dtype=np.int64)
    values = mu_final.astype(np.float64)
    
    return domain, values

# EVOLVE-BLOCK-END

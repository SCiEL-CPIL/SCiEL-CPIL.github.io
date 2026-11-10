# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_mobius(K):
    """Compute the Möbius function μ(n) for n = 1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    if K >= 1:
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
    return mu

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses a scaled Möbius function to satisfy constraints while maximizing the objective.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Candidate K values to test
    # We want large K for better approximation of the integral, 
    # but S_K = sum mu(k)/k might cause infeasibility if scaled by 1.
    # We will determine the optimal scaling factor alpha for each K.
    K_candidates = [10000, 20000, 50000, 100000, 200000]
    
    best_score = -np.inf
    best_domain = None
    best_values = None
    
    for K in K_candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 120:
            break
            
        try:
            mu = _compute_mobius(K)
            
            # Compute S_K = sum_{k=1}^K mu(k)/k
            # mu[0] is unused, mu[1..K]
            ks = np.arange(1, K + 1, dtype=np.float64)
            mu_float = mu[1:K+1].astype(np.float64)
            S_K = np.sum(mu_float / ks)
            
            # The adjusted function is f'(k) = alpha * mu(k) for k > 1, 
            # and f'(1) = alpha * (mu(1) - S_K) = alpha * (1 - S_K).
            # Wait, the evaluator does:
            # 1. Clip values (mu is -1,0,1 so no clip).
            # 2. S = sum f(k)/k. Here f(k) = alpha * mu(k).
            #    S = alpha * S_K.
            # 3. f(1) = f(1) - S = alpha * 1 - alpha * S_K = alpha * (1 - S_K).
            # 4. Check M(x) = sum f'(k) floor(x/k) <= 1.0001.
            #    M(x) = alpha * sum_{k>1} mu(k) floor(x/k) + f'(1) * 1
            #         = alpha * (sum_{k=1}^K mu(k) floor(x/k) - mu(1)floor(x/1) + (1-S_K)*1)
            #         = alpha * (1 - 1 + 1 - S_K)  <-- This logic is for x=1?
            
            # Let's re-evaluate M(x) generally.
            # sum_{k=1}^K mu(k) floor(x/k) = 1 for all x >= 1.
            # f'(k) = alpha * mu(k) for k != 1.
            # f'(1) = alpha * (1 - S_K).
            # M(x) = f'(1)*1 + sum_{k=2}^K alpha * mu(k) floor(x/k)
            #      = alpha * (1 - S_K) + alpha * (sum_{k=1}^K mu(k) floor(x/k) - mu(1)*1)
            #      = alpha * (1 - S_K) + alpha * (1 - 1)
            #      = alpha * (1 - S_K).
            
            # Wait, is this correct?
            # sum_{k=1}^K mu(k) floor(x/k) = 1.
            # Term k=1 is mu(1) floor(x/1) = 1 * x = x.
            # So sum_{k=2}^K mu(k) floor(x/k) = 1 - x.
            # M(x) = alpha(1-S_K) + alpha(1-x) = alpha(2 - S_K - x).
            
            # Let's double check with x=1.
            # M(1) = f'(1) = alpha(1-S_K).
            # Formula: alpha(2 - S_K - 1) = alpha(1 - S_K). Matches.
            
            # Let's check x=2.
            # M(2) = f'(1)*1 + f'(2)*1 = alpha(1-S_K) + alpha(-1) = alpha(-S_K).
            # Formula: alpha(2 - S_K - 2) = alpha(-S_K). Matches.
            
            # So M(x) = alpha * (2 - S_K - x) for x in [1, K].
            # For x > K, floor(x/k) for k<=K is not 1.
            # The identity sum mu(k) floor(x/k) = 1 holds for ALL x >= 1?
            # Yes, sum_{d|n} mu(d) = 1 if n=1 else 0.
            # sum_{k=1}^K mu(k) floor(x/k) = sum_{k=1}^K mu(k) sum_{j=1}^{floor(x/k)} 1
            # = sum_{j=1}^x sum_{k: k*j <= K} mu(k) ... this is not 1 for x > K.
            
            # Correction:
            # The identity is sum_{k=1}^{\infty} mu(k) floor(x/k) = 1.
            # If we truncate at K, sum_{k=1}^K mu(k) floor(x/k) is NOT 1 for x > K.
            # It is 1 for x <= K?
            # Let x=K. sum_{k=1}^K mu(k) floor(K/k).
            # This is sum_{d|n, n<=K} mu(d) ... no.
            # The identity holds for the infinite sum.
            # For finite K, let R(x, K) = sum_{k=1}^K mu(k) floor(x/k).
            # If x <= K, is R(x, K) = 1?
            # No. R(x, K) = sum_{j=1}^x sum_{k=1}^K mu(k) [k*j <= K] ... no.
            # R(x, K) = sum_{n=1}^x (sum_{k|n, k<=K} mu(k)).
            # If x <= K, then for any n <= x, all divisors k of n are <= n <= K.
            # So sum_{k|n} mu(k) = 1 if n=1 else 0.
            # So R(x, K) = 1 for all x <= K.
            
            # For x > K, R(x, K) != 1.
            # We need to check M(x) for x in [1, 10K].
            
            # M(x) = alpha * (f'(1)/alpha * 1 + sum_{k=2}^K mu(k) floor(x/k))
            # f'(1) = alpha(1-S_K).
            # M(x) = alpha(1-S_K) + alpha * sum_{k=2}^K mu(k) floor(x/k)
            #      = alpha * (1 - S_K + sum_{k=2}^K mu(k) floor(x/k))
            #      = alpha * (1 - S_K + (sum_{k=1}^K mu(k) floor(x/k) - 1))
            #      = alpha * (R(x, K) - S_K).
            
            # We need M(x) <= 1.0001 for all x in [1, 10K].
            # So alpha * max_{x in [1, 10K]} (R(x, K) - S_K) <= 1.0001.
            
            # We need to estimate max R(x, K).
            # For x <= K, R(x, K) = 1.
            # For x > K, R(x, K) = sum_{n=1}^x sum_{k|n, k<=K} mu(k).
            # This is related to the Mertens function.
            # R(x, K) = sum_{n=1}^x (1_{n=1} - sum_{k|n, k>K} mu(k)) ... no.
            # R(x, K) = 1 - sum_{n=1}^x sum_{k|n, k>K} mu(k) is not quite right.
            
            # Let's just compute R(x, K) for a few points if K is small, 
            # or use an upper bound.
            # A safe upper bound for R(x, K) is K (since |mu|<=1).
            # But it oscillates.
            
            # If we assume R(x, K) is bounded by some small constant for x up to 10K, 
            # we can find alpha.
            
            # However, for large K, R(x, K) for x > K can be large.
            # Specifically, R(x, K) = sum_{j=1}^{floor(x/K)} sum_{k=1}^K mu(k) ... no.
            
            # Let's compute R(x, K) for x = K, 2K, 5K, 10K numerically for the chosen K.
            # Cost is O(K) per x. 4 * K operations. For K=200,000, this is 800,000 ops, very fast.
            
            max_R = 0
            # Check x = 1, K, 2K, 5K, 10K
            check_xs = [1, K, 2*K, 5*K, 10*K]
            for x in check_xs:
                # R(x, K) = sum_{k=1}^K mu(k) floor(x/k)
                # Vectorized
                ks = np.arange(1, K + 1, dtype=np.int64)
                floors = x // ks
                R_x = np.sum(mu_float * floors)
                if R_x > max_R:
                    max_R = R_x
            
            # Also check x = 10K - 1?
            # The function R(x, K) is step-wise.
            # Max likely at large x.
            
            C = max_R - S_K
            if C <= 0:
                alpha = 1.0
            else:
                alpha = 1.0001 / C
            
            if alpha > 1.0:
                alpha = 1.0
            
            # Calculate objective a
            # a = -sum f(k) log(k) / k
            # f(k) = alpha * mu(k) (for k>1), f(1) = alpha(1-S_K)
            # a = alpha * ( - (1-S_K) log(1)/1 - sum_{k=2}^K mu(k) log(k)/k )
            # log(1) = 0, so first term is 0.
            # a = alpha * ( - sum_{k=2}^K mu(k) log(k)/k )
            # Note: sum_{k=1}^K mu(k) log(k)/k = 0 + sum_{k=2}^K ...
            
            L_K = -np.sum(mu_float[1:] * np.log(ks[1:]) / ks[1:])
            current_score = alpha * L_K
            
            if current_score > best_score:
                best_score = current_score
                best_domain = np.where(mu != 0)[0]
                best_values = mu[best_domain].astype(np.float64) * alpha
                # Store alpha and K for final construction if needed, 
                # but we return the scaled values directly.
                # Wait, the evaluator adjusts f(1) again!
                # We must return the UNADJUSTED f(k) = alpha * mu(k).
                # The evaluator will do:
                # S = sum (alpha * mu(k)) / k = alpha * S_K.
                # f(1) = alpha * 1 - alpha * S_K = alpha(1-S_K).
                # This matches our derivation.
                
        except Exception as e:
            print(f"Error for K={K}: {e}")
            continue

    if best_domain is None:
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_domain, best_values

# EVOLVE-BLOCK-END

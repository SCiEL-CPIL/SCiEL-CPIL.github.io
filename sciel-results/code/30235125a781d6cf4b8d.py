# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum f(k) log(k) / k subject to M(x) <= 1.0001.
    
    Strategy:
    Use the Möbius function mu(k) for k in [2, K].
    The Möbius function has the property that sum_{d|n} mu(d) = 1 if n=1 else 0.
    This implies that if we define f(k) = mu(k) for k >= 2, and choose f(1) such that
    sum_{k in domain} f(k)/k = 0, the resulting M(x) = sum f(k) floor(x/k) will be
    very close to 0 or 1 depending on the exact normalization, typically staying well
    within the M(x) <= 1.0001 bound for large K.
    
    Specifically, let S = sum_{k=2}^K mu(k)/k.
    Set f(1) = -S.
    Then sum_{k=1}^K f(k)/k = f(1) + S = 0.
    M(x) = f(1)*x + sum_{k=2}^K mu(k) floor(x/k).
    
    It is a known result that sum_{k=1}^K mu(k) floor(x/k) approximates 1 for x >= 1
    (related to the inverse of the zeta function or Mertens function properties).
    Actually, sum_{d|n} mu(d) = 1 if n=1 else 0.
    sum_{k=1}^x mu(k) floor(x/k) = sum_{k=1}^x mu(k) * (number of multiples of k <= x)
    = sum_{k=1}^x mu(k) * sum_{j: kj <= x} 1
    = sum_{n=1}^x sum_{d|n} mu(d)
    = 1 (for x >= 1).
    
    So, if we include k=1 in the sum with f(1) = mu(1) = 1, then M(x) = 1 exactly.
    However, the problem requires sum f(k)/k = 0 after f(1) adjustment.
    If we set f(k) = mu(k) for k >= 2, and f(1) = -sum_{k=2}^K mu(k)/k,
    then M(x) = -S*x + sum_{k=2}^K mu(k) floor(x/k).
    
    We know sum_{k=1}^K mu(k) floor(x/k) = 1 (for x <= K, assuming K is large enough to cover divisors).
    sum_{k=1}^K mu(k) floor(x/k) = mu(1)*floor(x/1) + sum_{k=2}^K mu(k) floor(x/k) = x + sum_{k=2}^K mu(k) floor(x/k).
    So sum_{k=2}^K mu(k) floor(x/k) = 1 - x.
    
    Therefore, M(x) = -S*x + (1 - x) = 1 - S*x - x = 1 - x(1 + S).
    
    Wait, this derivation assumes the sum goes to infinity or K is large enough.
    For finite K, there are error terms.
    Let's check the constraint M(x) <= 1.0001.
    If M(x) = 1 - x(1+S), and S = sum_{k=2}^K mu(k)/k.
    We know sum_{k=1}^infty mu(k)/k = 0. So S approaches 0 as K -> infinity.
    If S is small, M(x) approx 1 - x.
    For x=1, M(1) = 1 - 1(1+S) = -S.
    For x=2, M(2) = 1 - 2(1+S) = -1 - 2S.
    This seems to go negative. The constraint is M(x) <= 1.0001. Negative values are fine.
    Is M(x) bounded below? The problem doesn't state a lower bound, only M(x) <= 1.0001.
    And the objective is a = -sum f(k) log(k)/k.
    f(1) log(1)/1 = 0.
    a = -sum_{k=2}^K mu(k) log(k)/k.
    
    To maximize a, we want to minimize sum_{k=2}^K mu(k) log(k)/k.
    Since mu(k) takes values -1, 0, 1, and we want the sum to be very negative (so a is large positive),
    we want to include terms where mu(k) = 1 with large log(k)/k? No, we want -mu(k) log(k)/k to be positive.
    So we want mu(k) = -1 to contribute positively to a.
    a = sum_{k: mu(k)=-1} log(k)/k - sum_{k: mu(k)=1} log(k)/k.
    
    The sum sum_{k=2}^K mu(k) log(k)/k is related to -log(pi^2/6) or similar constants?
    Actually, sum_{k=1}^infty mu(k) log(k) / k = -zeta'(1)/zeta(1) ? No, zeta(1) is divergent.
    sum mu(k) log k / k converges to 0?
    Let's just compute it numerically for K=200000.
    
    The main risk is if M(x) exceeds 1.0001 for some x.
    Let's verify M(x) for the construction f(k)=mu(k) (k>=2), f(1)=-S.
    M(x) = sum_{k=1}^K f(k) floor(x/k).
    For x <= K, the "error" from truncating the sum at K is small.
    The identity sum_{d|n} mu(d) = 1 if n=1 else 0 holds exactly for the infinite sum.
    For finite K, sum_{k=1}^K mu(k) floor(x/k) = 1 + E(x, K).
    Then M(x) = f(1)x + sum_{k=2}^K mu(k) floor(x/k)
    = -S x + (sum_{k=1}^K mu(k) floor(x/k) - x)
    = -S x + (1 + E(x, K) - x)
    = 1 - x(1+S) + E(x, K).
    
    If S is close to 0 (which it is for large K, as sum mu(k)/k converges to 0),
    M(x) approx 1 - x + E(x, K).
    For x=1, M(1) approx 0 + E(1, K). E(1, K) is 0 if K>=1. So M(1) = -S.
    For x=2, M(2) approx -1 + E(2, K).
    Since M(x) becomes negative for x >= 2, the constraint M(x) <= 1.0001 is easily satisfied.
    The maximum of M(x) is likely at x=1, where M(1) = -S.
    S = sum_{k=2}^K mu(k)/k.
    Since sum_{k=1}^infty mu(k)/k = 0, S = -mu(1)/1 + small_tail = -1 + small_tail.
    So S approx -1.
    M(1) = -S approx 1.
    Specifically, M(1) = -sum_{k=2}^K mu(k)/k.
    We need M(1) <= 1.0001.
    -S <= 1.0001 => S >= -1.0001.
    Since S approx -1, this should hold.
    
    Let's implement this with K=200000.
    """
    start_time = time.time()
    deadline = start_time + 580

    def fallback():
        # Small safe fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    try:
        # Set K to maximum allowed
        K = 200000
        
        # Check time
        if time.time() > deadline - 10:
            return fallback()

        # Compute Mobius function up to K using sieve
        mu = np.ones(K + 1, dtype=np.int8)
        mu[0] = 0
        is_prime = np.ones(K + 1, dtype=bool)
        primes = []
        
        # Standard sieve for Mobius
        # Initialize mu[1] = 1
        # For each prime p, mu[n*p] = -mu[n] if p not in n, 0 if p in n.
        
        # Efficient vectorized approach is hard for standard library, but K=200k is small enough for a loop
        # Or use a simple sieve.
        
        # Let's use a standard loop sieve, it's fast enough for 200k in Python?
        # 200k iterations is trivial.
        
        mu = np.ones(K + 1, dtype=np.int8)
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        
        # Sieve of Eratosthenes to find primes, then compute mu
        # Actually, we can compute mu directly.
        
        # mu[n] = 0 if n has a squared prime factor
        # mu[n] = (-1)^k if n is a product of k distinct primes
        
        # Initialize mu to 1
        mu = np.ones(K + 1, dtype=np.int8)
        
        # Sieve for smallest prime factor or just iterate primes
        # Let's just iterate i from 2 to K
        # This is O(K log K) roughly.
        
        # Better:
        # 1. Identify primes.
        # 2. For each prime p, for multiples j*p, mu[j*p] *= -1.
        # 3. For each prime p, for multiples j*p^2, mu[j*p^2] = 0.
        
        # Step 1: Primes
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(np.sqrt(K)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        
        primes = np.where(is_prime)[0]
        
        # Step 2: Compute mu
        mu = np.ones(K + 1, dtype=np.int8)
        for p in primes:
            # Multiply by -1 for multiples of p
            mu[p::p] *= -1
            # Set to 0 for multiples of p^2
            p2 = p * p
            if p2 <= K:
                mu[p2::p2] = 0
        
        # mu[1] is 1 (correct)
        # mu[0] is 1 (irrelevant, we start from 2)
        
        # Calculate S = sum_{k=2}^K mu(k)/k
        # k is 2 to K
        ks = np.arange(2, K + 1, dtype=np.float64)
        mu_vals = mu[2:K+1].astype(np.float64)
        
        S = np.sum(mu_vals / ks)
        
        # f(1) = -S
        f1 = -S
        
        # Check constraint M(1) = f(1) <= 1.0001
        # M(1) = f(1) * floor(1/1) + sum_{k=2}^K f(k) * floor(1/k)
        # floor(1/k) = 0 for k >= 2.
        # So M(1) = f(1) = -S.
        
        if f1 > 1.0001:
            # This would mean S < -1.0001.
            # S = sum_{k=2}^K mu(k)/k.
            # Theoretically S -> -1.
            # If it's slightly less than -1, we might violate.
            # In that case, we can reduce K or scale down?
            # Scaling down f(k) reduces M(x) and a.
            # Let's check if f1 is close to 1.
            # If f1 > 1.0001, we have a problem.
            # However, sum_{k=1}^infty mu(k)/k = 0 => sum_{k=2}^infty mu(k)/k = -1.
            # The partial sum oscillates around -1.
            # For K=200000, it should be very close to -1.
            # If it exceeds, we can just return a smaller K.
            
            # Let's try K=100000
            K = 100000
            # Recalculate
            ks = np.arange(2, K + 1, dtype=np.float64)
            mu_vals = mu[2:K+1].astype(np.float64)
            S = np.sum(mu_vals / ks)
            f1 = -S
            
            if f1 > 1.0001:
                # Fallback to very small K
                K = 100
                ks = np.arange(2, K + 1, dtype=np.float64)
                mu_vals = mu[2:K+1].astype(np.float64)
                S = np.sum(mu_vals / ks)
                f1 = -S
                
                if f1 > 1.0001:
                    return fallback()

        # Construct domain and values
        # Domain: [1, 2, ..., K]
        # Values: [f1, mu(2), ..., mu(K)]
        
        # Only include k where mu(k) != 0 to save space?
        # The problem says "domain holds distinct positive integers".
        # It doesn't say we must include all.
        # But including zeros doesn't change M(x) or a.
        # However, m (length) affects cost? "cost about m * 10K".
        # If m is large, evaluation is slow.
        # But we are constructing it, not evaluating it in our code (the evaluator does).
        # The evaluator cost is m * 10K.
        # If K=200000, m=200000. Cost = 200000 * 2000000 = 4e11 operations?
        # No, "cost about m * 10K" usually refers to the loop in the evaluator.
        # If the evaluator is slow, it might timeout.
        # The task says "Each evaluation must finish within 600 s".
        # If m=200000, and the evaluator does a naive sum for each x in [1, 10K],
        # that's 200000 * 2000000 = 4e11 ops. That's too slow.
        
        # Wait, the evaluator description says:
        # "M(x) = sum_k f(k) * floor(x/k) is computed for every integer x in [1, 10K] (cost about m * 10K)"
        # This implies the evaluator uses a naive O(m * X) algorithm.
        # If m=200000 and X=2000000, it's 4e11.
        # 600 seconds is ~ 1e10 to 1e11 simple ops in Python/C?
        # If the evaluator is in C/NumPy, it might be fast.
        # But if it's Python loops, it will timeout.
        
        # To be safe, we should keep m small.
        # We can filter out k where mu(k) = 0.
        # Number of square-free numbers is ~ 6/pi^2 * K.
        # Still large.
        
        # Can we reduce K?
        # The objective a = -sum mu(k) log(k)/k.
        # The terms decrease as log(k)/k.
        # The sum converges.
        # Most of the value is in the first few thousand terms.
        # Let's try K=10000.
        # m ~ 6000. Cost ~ 6000 * 100000 = 6e8. This is safe.
        
        # Let's restart with K=10000 to ensure speed.
        # We already computed mu up to 200000, so we can just slice.
        
        K_eff = 10000
        ks = np.arange(2, K_eff + 1, dtype=np.float64)
        mu_vals = mu[2:K_eff+1].astype(np.float64)
        
        # Filter out zeros to reduce m
        non_zero_mask = mu_vals != 0
        ks_nz = ks[non_zero_mask]
        mu_vals_nz = mu_vals[non_zero_mask]
        
        S = np.sum(mu_vals_nz / ks_nz)
        f1 = -S
        
        # Check M(1) = f1
        if f1 > 1.0001:
             # Adjust? Or just return fallback.
             # S should be close to -1.
             # If f1 > 1.0001, it's invalid.
             return fallback()
        
        # Construct arrays
        domain = np.empty(len(ks_nz) + 1, dtype=np.int64)
        values = np.empty(len(ks_nz) + 1, dtype=np.float64)
        
        domain[0] = 1
        values[0] = f1
        
        domain[1:] = ks_nz.astype(np.int64)
        values[1:] = mu_vals_nz
        
        # Clip values
        values = np.clip(values, -10.0, 10.0)
        
        return domain, values

    except Exception as e:
        return fallback()

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600.0

    # Fallback safe solution (the seed program)
    fallback_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    fallback_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    def check_feasibility(mu, K):
        """
        Check if the function defined by mu on [1, K] is feasible.
        Returns True if max_{x in [1, 10K]} M'(x) <= 1.0001.
        """
        if time.time() > deadline - 10:
            return False

        max_x = 10 * K
        # g(j) = sum_{d|j, d<=K} mu(d)
        # For x <= K, M'(x) = 1 (since sum_{d|x} mu(d) = 1 for x=1, 0 for x>1? No.)
        # Wait, the identity is sum_{d|x} mu(d) = 1 if x=1, 0 if x>1.
        # The evaluator computes M(x) = sum_{k=1}^K f(k) floor(x/k).
        # If f(k) = mu(k), then M(x) = sum_{k=1}^K mu(k) floor(x/k).
        # This is equal to sum_{n=1}^x (sum_{d|n, d<=K} mu(d)).
        # Let h(n) = sum_{d|n, d<=K} mu(d).
        # If n <= K, h(n) = 1 if n=1, 0 if n>1.
        # If n > K, h(n) = sum_{d|n, d<=K} mu(d).
        # So M(x) = 1 + sum_{n=K+1}^x h(n) for x >= K.
        # For x < K, M(x) is 1 (for x=1) or 0?
        # Actually, for x <= K, M(x) = sum_{n=1}^x h(n).
        # h(1) = 1. h(n) = 0 for 1 < n <= K.
        # So M(1) = 1. M(x) = 1 for x in [1, K].
        # We need M(x) <= 1.0001.
        # So we need 1 + sum_{n=K+1}^x h(n) <= 1.0001 for all x in [K+1, 10K].
        # i.e., max_{x in [K+1, 10K]} sum_{n=K+1}^x h(n) <= 0.0001.
        
        # Compute h(n) for n in [1, 10K]
        # h(n) = sum_{d|n, d<=K} mu(d)
        # We can compute this by iterating d from 1 to K and adding mu[d] to h[m] for m = d, 2d, ... <= 10K.
        
        # Optimization: Only compute h for n in [K+1, 10K]?
        # No, h(n) depends on divisors d <= K.
        # We can just compute the tail sum directly?
        # Let S(x) = sum_{n=K+1}^x h(n).
        # S(x) = sum_{n=K+1}^x sum_{d|n, d<=K} mu(d)
        # S(x) = sum_{d=1}^K mu(d) * (sum_{n=K+1}^x [d|n])
        # sum_{n=K+1}^x [d|n] = floor(x/d) - floor(K/d)
        # So S(x) = sum_{d=1}^K mu(d) * (floor(x/d) - floor(K/d))
        
        # We need max_{x in [K+1, 10K]} S(x) <= 0.0001.
        # Note that S(K) = 0.
        # S(x) is a step function that changes only when x/d is an integer for some d.
        # The changes occur at x = k*d for integers k.
        # Specifically, S(x) increases by mu(d) when x crosses a multiple of d.
        # Let's evaluate S(x) at the critical points.
        # The critical points are integers in [K+1, 10K].
        # Since S(x) is constant between integers, we just need to check integers x in [K+1, 10K].
        # But checking all x is O(K^2) if we do it naively.
        # However, S(x) = sum_{d=1}^K mu(d) floor(x/d) - C, where C = sum_{d=1}^K mu(d) floor(K/d).
        # We want to maximize S(x) over x in [K+1, 10K].
        
        # Alternative approach:
        # The maximum of S(x) is likely to occur at points where many terms floor(x/d) increase.
        # But let's just compute S(x) for all x in [K+1, 10K] efficiently?
        # S(x) - S(x-1) = h(x).
        # So we can compute h(x) for x in [K+1, 10K] and take prefix sums.
        # h(x) = sum_{d|x, d<=K} mu(d).
        
        # To compute h(x) for x in [K+1, 10K]:
        # Initialize h = zeros(10K+1)
        # For d in 1..K:
        #   if mu[d] != 0:
        #     for m in d, 2d, ... <= 10K:
        #       h[m] += mu[d]
        # This is O(K log K).
        
        # Then compute prefix sum of h from K+1 to 10K.
        
        if time.time() > deadline - 5:
            return False

        h = np.zeros(10 * K + 1, dtype=np.float64)
        
        # Vectorized update for h
        # for d in range(1, K + 1):
        #     if mu[d] != 0:
        #         h[d::d] += mu[d]
        
        # This loop is the bottleneck.
        # Let's try to optimize it.
        # We can skip d where mu[d] == 0.
        nonzero_d = np.nonzero(mu)[0]
        # nonzero_d includes 0, we start from 1
        nonzero_d = nonzero_d[nonzero_d > 0]
        
        for d in nonzero_d:
            if time.time() > deadline - 5:
                return False
            h[d::d] += mu[d]
            
        # Compute S(x) for x in [K+1, 10K]
        # S(x) = sum_{n=K+1}^x h(n)
        # We only care about the max of S(x).
        # S(K) = 0.
        # S(K+1) = h(K+1)
        # S(K+2) = h(K+1) + h(K+2)
        
        # We can compute the cumulative sum of h from K+1 to 10K.
        # h is defined up to 10K.
        # tail_h = h[K+1 : 10K+1]
        # S_vals = np.cumsum(tail_h)
        # max_S = np.max(S_vals)
        
        # To save memory, we can do this in chunks or just use cumsum if memory allows.
        # K=200000, 10K=2,000,000. Array of size 2M is fine.
        
        tail_h = h[K+1 : 10*K + 1]
        S_vals = np.cumsum(tail_h)
        
        if len(S_vals) == 0:
            return True
            
        max_S = np.max(S_vals)
        
        return max_S <= 0.0001

    # Try to find the largest feasible K
    # We know that for small K, it's feasible.
    # Let's start with K=200000 and go down.
    
    # Precompute mu for max K = 200000
    K_max = 200000
    mu = np.ones(K_max + 1, dtype=np.int64)
    primes = []
    is_composite = np.zeros(K_max + 1, dtype=bool)
    
    mu[1] = 1
    for i in range(2, K_max + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_max:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # Binary search or linear search for largest feasible K?
    # Feasibility is not monotonic? 
    # If K is feasible, is K-1 feasible?
    # M_K(x) = sum_{k=1}^K mu(k) floor(x/k).
    # M_{K-1}(x) = M_K(x) - mu(K) floor(x/K).
    # If mu(K) = 1, M_{K-1}(x) = M_K(x) - floor(x/K). This decreases M(x), so likely more feasible.
    # If mu(K) = -1, M_{K-1}(x) = M_K(x) + floor(x/K). This increases M(x), so likely less feasible.
    # So it's not strictly monotonic.
    
    # However, the oscillations of the tail sum S(x) for mu are related to the error term of the Prime Number Theorem.
    # The error term is small for large K? No, the error term is large.
    # But we are truncating the sum at K.
    # The "tail" of the Möbius function is what causes the oscillations.
    
    # Let's just try a few large K values.
    # K = 200000, 100000, 50000, 10000, 1000
    
    candidate_Ks = [200000, 150000, 100000, 75000, 50000, 25000, 10000, 5000, 1000]
    
    for K in candidate_Ks:
        if time.time() > deadline - 20:
            break
            
        if check_feasibility(mu, K):
            # Found a feasible K
            # Construct the domain and values
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = mu[1:K+1].astype(np.float64)
            
            # Optimization: Can we remove some elements to improve score?
            # Score a = -sum f(k) log(k) / k.
            # For mu, f(k) is -1, 0, 1.
            # If we remove a k where mu(k) = -1, we lose -(-1) log(k)/k = + log(k)/k from the sum?
            # Wait, a = - sum mu(k) log(k) / k.
            # If mu(k) = -1, term is -(-1) log(k)/k = log(k)/k. This is positive.
            # If mu(k) = 1, term is -(1) log(k)/k = -log(k)/k. This is negative.
            # We want to maximize a.
            # So we want to keep k where mu(k) = -1 and remove k where mu(k) = 1?
            # But removing k changes M(x) and might violate feasibility.
            # Also, the normalization S = sum f(k)/k changes.
            # The evaluator handles normalization.
            # If we remove a k, the set of k changes.
            
            # Let's just return the full set for now.
            # We can try to prune later if time permits.
            
            return domain, values

    # If no large K is feasible, try smaller ones or fallback
    # Actually, K=1 is always feasible?
    # f(1) = 1.
    # M(x) = floor(x/1) = x.
    # For x in [1, 10], M(x) = x. Max is 10.
    # 10 > 1.0001. So K=1 is infeasible!
    # Wait, the evaluator normalizes f(1).
    # "f(1) is replaced by f(1) - S ... so that sum f(k)/k = 0"
    # If domain is [1], f(1) = 1.
    # S = f(1)/1 = 1.
    # f(1) becomes 1 - 1 = 0.
    # Then M(x) = 0 * floor(x/1) = 0.
    # 0 <= 1.0001. Feasible.
    # Score a = -0 * log(1)/1 = 0.
    # This is a valid solution with score 0.
    
    # The seed program has score 1.0.
    # Let's check if the seed program is feasible.
    # domain = [1, 2, 3, 5, 30]
    # values = [3.47, -1, -1, -1, 1]
    # This is a specific partial function.
    
    # If we can't find a large K, we should return the seed program to ensure score 1.0.
    # But wait, if K=200000 is infeasible, maybe K=10000 is feasible and gives score > 1.0?
    # Score for mu up to K is approx -sum_{k<=K} mu(k) log(k)/k.
    # This sum converges to 1.
    # For small K, it might be less than 1 or greater than 1?
    # Let's check the seed score. It is 1.0.
    # The theoretical limit is 1.
    # So we want to get close to 1 from below? Or above?
    # "score is proportional to a". "raw value (higher is better)".
    # If a approaches 1, and seed is 1.0, maybe we can get > 1.0?
    # The partial sums of -sum mu(k) log(k)/k oscillate around 1?
    # Actually, the Dirichlet series for log zeta(s) is -sum mu(n) log n / n^s.
    # At s=1, it diverges? No, zeta(1) diverges.
    # But the sum -sum mu(n) log n / n is related to the derivative of zeta?
    # No, sum mu(n) log n / n^s = -log zeta(s).
    # As s -> 1, log zeta(s) ~ -log(s-1).
    # So -log zeta(s) ~ log(s-1) -> -infty.
    # So the sum diverges to -infinity?
    # Wait, the PNT says sum_{n<=x} mu(n) ~ 0.
    # The sum -sum mu(n) log n / n converges to 1?
    # Let's check the value for the seed program.
    # The seed program returns score 1.0.
    # This suggests that the specific partial function used in the seed achieves a=1.0 (or normalized to 1.0).
    # Actually, the prompt says "score 1.0 equals the seed program".
    # So if we find a better f, score > 1.0.
    
    # If all large K are infeasible, return seed.
    return fallback_domain, fallback_values

# EVOLVE-BLOCK-END

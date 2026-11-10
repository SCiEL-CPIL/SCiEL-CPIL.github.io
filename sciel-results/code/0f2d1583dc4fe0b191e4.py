# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy:
      1. Use the Mobius function mu(k) on square-free integers up to K.
      2. Scale by c such that the constraint M(x) <= 1.0001 is satisfied for all x in [1, 10K].
      3. Normalize f(1) so that sum_k f(k)/k = 0.

    The objective is a = -sum_k f(k) * log(k) / k.
    """
    start_time = time.time()
    deadline = 580.0  # seconds, leave 50s buffer

    # We'll try increasing K values to see which gives the best objective
    # Start with a reasonable K that's fast to compute
    # The cost is roughly O(K * log(K)) for Mobius + O(K * 10K) for M(x) check
    # Actually M(x) for all x in [1, 10K] is expensive. Let's be smart.

    # Key insight: M(x) = sum_k f(k) * floor(x/k)
    # If f(k) = c * mu(k) for k in domain, then M(x) = c * sum_{k in domain} mu(k) * floor(x/k)
    # For the full domain k=1..K, this is c * sum_{k=1}^K mu(k) * floor(x/k)
    # This equals c for x >= 1 if we had the infinite sum, but finite sum has errors.

    # Actually, the identity sum_{d|n} mu(d) = [n=1] gives us that
    # sum_{k=1}^K mu(k) * floor(x/k) = sum_{k=1}^K mu(k) * sum_{j: kj <= x} 1
    # = sum_{j=1}^x sum_{k=1}^{min(K, x/j)} mu(k)
    # = sum_{j=1}^x (1 if x/j < 2 else 0) ... no, that's not right.
    
    # Let me think differently. The function H_K(x) = sum_{k=1}^K mu(k) * floor(x/k)
    # For large x relative to K, floor(x/k) is large.
    # H_K(x) = sum_{k=1}^K mu(k) * floor(x/k)
    
    # We need c * max(0, max_x H_K(x)) <= 1.0001
    # So c <= 1.0001 / max(0, max_x H_K(x))
    
    # Let's compute H_K(x) efficiently. Note that floor(x/k) takes only O(sqrt(x)) distinct values.
    # But we need to check all x in [1, 10K].
    
    # Alternative: just compute for a few strategic x values and verify.
    # Or use the fact that H_K(x) is related to the Mertens function.
    
    # Let's just try K=5000 first, compute everything, and see.
    
    best_domain = None
    best_values = None
    best_objective = -np.inf
    
    for K in [5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() - start_time > deadline - 30:
            break
        
        # Compute mu(k) for k = 1..K
        mu = np.ones(K + 1, dtype=np.int64)
        is_prime = np.ones(K + 1, dtype=bool)
        primes = []
        is_prime[0] = False
        if K >= 1:
            is_prime[1] = False
        
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
        
        # Get square-free indices where mu(k) != 0
        sq_free = np.where(mu != 0)[0]
        sq_free = sq_free[sq_free >= 1]  # k >= 1
        m = len(sq_free)
        
        if m == 0:
            continue
        
        # Compute S = sum_{k in domain} mu(k) / k
        S = np.sum(mu[sq_free] * 1.0 / sq_free)
        
        # We'll set f(k) = c * mu(k) for k in domain, then adjust f(1) for normalization.
        # After clipping (no change since |mu|<=1 and c<=10), S_clipped = c * S
        # f(1) becomes f(1) - S_clipped = c*mu(1) - c*S = c*(1 - S)
        # Other f(k) remain c*mu(k)
        
        # M(x) = f(1)*x + sum_{k in domain, k>=2} f(k) * floor(x/k)
        # = c*(1-S)*x + c * sum_{k in domain, k>=2} mu(k) * floor(x/k)
        # = c * [ (1-S)*x + sum_{k in domain, k>=2} mu(k) * floor(x/k) ]
        # = c * [ x - S*x + sum_{k>=2, k in domain} mu(k) * floor(x/k) ]
        # = c * [ sum_{k in domain} mu(k)*floor(x/k) - S*x ]  since mu(1)=1 and floor(x/1)=x
        # = c * [ H_K(x) - S*x ]  where H_K(x) = sum_{k in domain} mu(k)*floor(x/k)
        
        # Wait, that's only if domain is all k=1..K with mu(k)!=0.
        # H_K(x) = sum_{k=1}^K mu(k)*floor(x/k)
        # M(x)/c = H_K(x) - S*x
        
        # We need max_x (H_K(x) - S*x) * c <= 1.0001
        # And also min can be anything (no lower bound constraint)
        
        # Compute H_K(x) for x in [1, 10K]
        # This is expensive. Let's use the floor sum trick.
        # H_K(x) = sum_{k=1}^K mu(k) * floor(x/k)
        
        # For a given x, we can compute this by grouping k by floor(x/k) values.
        # But we need it for all x in [1, 10K].
        
        # Alternative: iterate over k and add mu(k) to a difference array for ranges where floor(x/k) is constant.
        # floor(x/k) = v for x in [v*k, (v+1)*k - 1]
        # So for each k, for each v = 1, 2, ..., floor(10K/k), 
        #   add mu(k) to the range [v*k, min((v+1)*k-1, 10K)]
        
        # Total work: sum_{k=1}^K floor(10K/k) = 10K * H_K ~ 10K * ln(K)
        # For K=200000, that's ~ 10 * 200000 * 12 = 24 million operations. Might be too slow in Python.
        
        # Let's use numpy. Create an array of size 10K+1, and for each k, add mu(k) to ranges.
        max_x = 10 * K
        H = np.zeros(max_x + 1, dtype=np.float64)
        
        for k in range(1, K + 1):
            if mu[k] == 0:
                continue
            mk = mu[k]
            # floor(x/k) = v for x in [v*k, (v+1)*k - 1]
            # v ranges from 1 to floor(max_x / k)
            v_max = max_x // k
            for v in range(1, v_max + 1):
                start = v * k
                end = min((v + 1) * k - 1, max_x)
                H[start:end+1] += mk
        
        # Now H[x] = H_K(x) for x in [1, 10K]
        # G(x) = H(x) - S*x
        x_vals = np.arange(1, max_x + 1, dtype=np.float64)
        G = H - S * x_vals
        
        max_G = np.max(G)
        
        if max_G <= 0:
            c = 10.0  # Can use full scale
        else:
            c = min(10.0, 1.0001 / max_G)
        
        if c <= 0:
            continue
        
        # Compute objective: a = -sum_k f(k) * log(k) / k
        # f(k) = c * mu(k) for k in domain
        # a = -c * sum_{k in domain} mu(k) * log(k) / k
        log_k = np.log(sq_free.astype(np.float64))
        a = -c * np.sum(mu[sq_free] * log_k / sq_free)
        
        if a > best_objective:
            best_objective = a
            # Build domain and values
            domain = sq_free.copy()
            values = c * mu[sq_free].astype(np.float64)
            # f(1) = c * mu(1) = c * 1 = c, but we need to adjust for normalization
            # The evaluator does: clip, then f(1) -= S where S = sum f(k)/k over clipped values
            # Since we're outputting unclipped values that will be clipped to same values (|c*mu|<=10),
            # the evaluator will compute S_eval = c*S and set f(1) = c - c*S = c*(1-S)
            # We should output the pre-adjustment values; the evaluator handles the adjustment.
            # So we just output f(k) = c * mu(k)
            best_domain = domain
            best_values = values
        
        if time.time() - start_time > deadline - 20:
            break
    
    if best_domain is None:
        # Fallback: minimal valid function
        best_domain = np.array([1], dtype=np.int64)
        best_values = np.array([0.0], dtype=np.float64)
    
    return best_domain, best_values


# EVOLVE-BLOCK-END

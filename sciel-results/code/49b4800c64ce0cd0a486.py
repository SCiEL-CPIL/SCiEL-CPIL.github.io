# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_mobius(K: int) -> np.ndarray:
    """
    Compute the Möbius function mu(n) for n = 1..K using a linear sieve.
    Returns an array of shape (K+1,) where index 0 is unused.
    """
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600.0
    
    # We aim for the maximum K to maximize the score a = -sum mu(k) log(k)/k.
    # As K increases, a approaches 1.
    # The constraint is M(x) = sum_{k<=K} mu(k) floor(x/k) <= 1.0001 for all x in [1, 10K].
    
    # Let's try K = 200000.
    K = 200000
    
    # 1. Compute Mobius function
    mu = _compute_mobius(K)
    
    # 2. Prepare domain and values
    # The evaluator clips values to [-10, 10]. mu is -1, 0, 1, so no clipping needed.
    # f(1) is adjusted by the evaluator to make sum f(k)/k = 0.
    # We return f(1)=0 in our output, the evaluator handles the rest.
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    values[0] = 0.0  # f(1)
    
    # 3. Feasibility Check
    # We need to check M(x) <= 1.0001 for x in [1, 10K].
    # M(x) = sum_{k=1}^K mu[k] * floor(x/k)
    
    # Direct computation of M(x) for all x is O(10K * K) which is too slow.
    # However, we can compute M(x) efficiently by iterating over k and adding to M(x).
    # M(x) = sum_{k=1}^K mu[k] * floor(x/k)
    # This is equivalent to: for each k, add mu[k] to M(x) for x in [j*k, (j+1)*k - 1] for j=1..10K/k.
    # Wait, floor(x/k) = j for x in [j*k, (j+1)*k - 1].
    # So M(x) is a piecewise constant function? No, floor(x/k) is a step function.
    # M(x) is a sum of step functions.
    
    # Let's compute M(x) for all x in [1, 10K] using vectorized operations.
    # x_vals = np.arange(1, 10*K + 1)
    # This array has size 2,000,000.
    # For each k (1..K), we compute x_vals // k.
    # This loop is K iterations, each doing a division on an array of size 10K.
    # Total ops: K * 10K = 200000 * 2000000 = 4 * 10^11. This is too slow in Python/NumPy loop.
    
    # Alternative:
    # M(x) = sum_{k=1}^K mu[k] floor(x/k)
    # We can compute this using the divisor summatory function properties?
    # No, let's try to optimize the verification.
    # Actually, we don't need to check ALL x if we can prove it or check a subset.
    # But the evaluator checks ALL x.
    
    # Let's try a smarter verification:
    # M(x) is maximized at some x. Where?
    # For x <= K, M(x) = 1 (by identity sum_{k|x} mu(k) = 0 for x>1, and 1 for x=1? 
    # Wait, the identity is sum_{d|n} mu(d) = 1 if n=1 else 0.
    # So sum_{k=1}^x mu(k) floor(x/k) = sum_{k=1}^x mu(k) sum_{j=1}^{floor(x/k)} 1 
    # = sum_{j=1}^x sum_{k=1}^{floor(x/j)} mu(k)
    # = sum_{j=1}^x M(floor(x/j)) ... no, that's not right.
    # The identity is: sum_{d|n} mu(d) = [n=1].
    # So sum_{k=1}^x mu(k) floor(x/k) = sum_{k=1}^x mu(k) sum_{m: km <= x} 1
    # = sum_{m=1}^x sum_{k: k <= x/m} mu(k)
    # Let S(n) = sum_{k=1}^n mu(k).
    # Then M(x) = sum_{m=1}^x S(floor(x/m)).
    # For x <= K, S(floor(x/m)) is well-defined and the sum is exactly 1.
    # Proof: sum_{m=1}^x S(floor(x/m)) = sum_{m=1}^x sum_{k=1}^{floor(x/m)} mu(k)
    # = sum_{k=1}^x mu(k) * (number of m such that m <= x/k)
    # = sum_{k=1}^x mu(k) floor(x/k) = 1.
    
    # So for x <= K, M(x) = 1.
    # For x > K, M(x) = sum_{m=1}^x S(floor(x/m)).
    # Since S(n) is bounded (S(n) = O(n exp(-c log n ...))), M(x) should be close to 1.
    # Specifically, M(x) = 1 - sum_{k=K+1}^x mu(k) floor(x/k).
    # The term sum_{k=K+1}^x mu(k) floor(x/k) is the "tail".
    # If the tail is negative, M(x) > 1. If positive, M(x) < 1.
    # We need M(x) <= 1.0001.
    
    # Let's compute M(x) for x in [K+1, 10K] efficiently.
    # We can compute S(n) for n up to 10K? No, S(n) is prefix sum of mu.
    # We have mu up to K. We don't have mu up to 10K.
    # But M(x) only depends on mu(k) for k <= K.
    # M(x) = sum_{k=1}^K mu(k) floor(x/k).
    
    # Let's try to compute M(x) for x in [K+1, 10K] using the formula:
    # M(x) = sum_{m=1}^x S(floor(x/m)) is NOT useful because we don't have S(n) for n > K.
    # We must use M(x) = sum_{k=1}^K mu(k) floor(x/k).
    
    # Can we compute this faster?
    # floor(x/k) takes O(sqrt(x)) distinct values.
    # But we need M(x) for ALL x.
    
    # Let's try to just run the loop with NumPy and see if it fits in time.
    # K = 200000. x_max = 2000000.
    # x_vals = np.arange(1, x_max + 1)
    # M = np.zeros(x_max)
    # for k in range(1, K+1):
    #     M += mu[k] * (x_vals // k)
    # This loop is 200,000 iterations.
    # Each iteration does a division of an array of size 2,000,000.
    # 200,000 * 2,000,000 = 4 * 10^11 operations.
    # NumPy vectorized ops are fast, but 4*10^11 is likely > 600s.
    # (10^8 ops/s is typical for NumPy? Maybe 10^9? 4*10^11 is 400-4000 seconds. Risky.)
    
    # Optimization:
    # Many mu[k] are 0. About 6/pi^2 ~ 60% are non-zero.
    # Still 120,000 iterations.
    
    # Alternative:
    # We only need to check if max M(x) > 1.0001.
    # We can check x in [K+1, 10K].
    # We can compute M(x) in chunks.
    
    # Let's try a smaller K first to ensure speed, then increase if time permits.
    # But the score depends on K.
    # Let's try K = 50000.
    # x_max = 500,000.
    # Ops: 50,000 * 500,000 = 2.5 * 10^10.
    # This is 10x faster. Should be feasible in ~10-50s.
    
    # Let's try K = 100000.
    # x_max = 1,000,000.
    # Ops: 10^11.
    
    # Let's implement a binary search on K or just try K=100000 and verify.
    
    # Revised Plan:
    # 1. Compute mu for K=200000.
    # 2. Try to verify K=200000. If too slow, fall back to K=100000, then 50000.
    
    # To speed up verification:
    # We can compute M(x) using the fact that floor(x/k) is constant for ranges of x?
    # No, we are summing over k.
    
    # Let's try to vectorize the k-loop differently.
    # M[x] = sum_{k=1}^K mu[k] * floor(x/k)
    # This is a convolution-like operation? No.
    
    # Let's just try K=50000 first. It's a safe bet for time.
    # Score for K=50000 is very close to 1.
    
    K = 50000
    mu = _compute_mobius(K)
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    values[0] = 0.0
    
    # Verify K=50000
    x_max = 10 * K
    x_vals = np.arange(1, x_max + 1, dtype=np.int64)
    M = np.zeros(x_max, dtype=np.float64)
    
    # Only iterate over k where mu[k] != 0
    non_zero_k = np.nonzero(mu[1:])[0] + 1
    non_zero_mu = mu[non_zero_k]
    
    for i in range(len(non_zero_k)):
        k = non_zero_k[i]
        M += non_zero_mu[i] * (x_vals // k)
        
        # Time check
        if i % 1000 == 0:
            if time.time() - start_time > deadline - 60:
                # Time out, return current best (which is just the unverified K=50000)
                # But we should return a verified one if possible.
                # If we haven't finished, we can't verify.
                # Let's break and return the unverified one? No, risk of 0 score.
                # Better to return a smaller verified K.
                # But we don't have a verified smaller K yet.
                # Let's just return the seed fallback if we can't verify.
                pass
    
    max_M = np.max(M)
    
    if max_M <= 1.0001:
        return domain, values
    else:
        # Infeasible. Try smaller K.
        # This is unlikely for Mobius, but just in case.
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

# EVOLVE-BLOCK-END

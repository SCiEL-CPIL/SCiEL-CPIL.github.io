# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600.0
    
    # We will compute the Mobius function up to K.
    # A larger K gives a better approximation of the infinite sum, approaching the optimal score of 1.0.
    # However, we must ensure feasibility.
    # Let's try a few K values and pick the best feasible one.
    
    def compute_mobius(K):
        mu = np.ones(K + 1, dtype=np.int8)
        is_prime = np.ones(K + 1, dtype=bool)
        primes = []
        is_prime[0] = False
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
        return mu

    def check_feasibility(domain, values, K):
        # M(x) = sum f(k) floor(x/k)
        # We check for x in 1..10K.
        # Optimization: vectorize?
        # M(x) for all x is a bit expensive if done naively for each x.
        # But K is small (e.g., 10000), so 10K = 100000.
        # m = K.
        # Cost m * 10K = 10^9 operations. Too slow in Python loop.
        # Need faster check.
        
        # M(x) = sum_{k=1}^K f(k) floor(x/k)
        # We can compute this for all x=1..10K using convolution or just loops if K is small enough?
        # 10^9 is too slow.
        # Let's use the property M(x) = A_K(x) - S*x.
        # A_K(x) = sum_{k=1}^K mu(k) floor(x/k) (with f(1) adjusted? No, A_K uses original mu)
        # Wait, M(x) with f(1) = 1-S and f(k)=mu(k) for k>1 is:
        # M(x) = (1-S)x + sum_{k=2}^K mu(k) floor(x/k)
        # Let B_K(x) = sum_{k=1}^K mu(k) floor(x/k).
        # Then sum_{k=2}^K ... = B_K(x) - x.
        # M(x) = (1-S)x + B_K(x) - x = B_K(x) - S*x.
        
        # So we need to check if max_{x in [1, 10K]} (B_K(x) - S*x) <= 1.0001.
        
        # How to compute B_K(x) for all x efficiently?
        # B_K(x) = sum_{k=1}^K mu(k) floor(x/k).
        # This is the Dirichlet convolution of mu and 1, summed? No.
        # It is the summatory function of the multiplicative function g(n) = sum_{d|n} mu(d) = [n==1]? No.
        # sum_{d|n} mu(d) is 1 if n=1, 0 if n>1.
        # So sum_{n=1}^x sum_{d|n} mu(d) = 1.
        # sum_{d=1}^K mu(d) floor(x/d) approximates this sum.
        # The difference is the tail.
        
        # We can compute B_K(x) for all x=1..10K by iterating k and adding mu(k) to ranges?
        # floor(x/k) changes at multiples of k.
        # This is still O(K log K) or similar if done carefully, but Numpy vectorization might help.
        
        # Alternative: Just check a subset of x? No, must be all.
        # But if K is small, maybe we can just do it?
        # If K=1000, 10K=10000. m=1000. 10^7 ops. Fast.
        # If K=10000, 10K=100000. m=10000. 10^9 ops. Slow.
        
        # Let's stick to K=2000 or 5000 for safety in Python loops, or optimize the check.
        
        # Optimization:
        # B_K(x) = sum_{k=1}^K mu(k) floor(x/k).
        # We can compute this array using a loop over k.
        # For a fixed k, floor(x/k) is constant on intervals [j*k, (j+1)*k - 1].
        # We can use np.repeat or similar?
        
        # Actually, let's just try K=1000 first. It should be fast and give a decent score.
        # Score is proportional to a = -sum f(k) log(k) / k.
        # For mu, a approx 1.
        
        pass

    # Let's implement a faster feasibility check using NumPy.
    def is_feasible(K, mu, S):
        # Compute B_K(x) for x=1..10K
        # B_K(x) = sum_{k=1}^K mu(k) floor(x/k)
        # We can compute this by adding mu(k) * floor(x/k) to an array.
        # Vectorized approach:
        # Create array x = 1..10K
        # For each k, add mu(k) * (x // k)
        # This is K iterations of vector ops of size 10K.
        # Total ops K * 10K.
        # If K=1000, 10^7. OK.
        # If K=5000, 2.5 * 10^8. Maybe OK in Numpy.
        
        max_x = 10 * K
        x = np.arange(1, max_x + 1)
        B = np.zeros(max_x, dtype=np.float64)
        
        # mu is 1-indexed 0..K. mu[0] is unused.
        for k in range(1, K + 1):
            B += mu[k] * (x // k)
        
        M_vals = B - S * x
        return np.max(M_vals) <= 1.0001

    best_score = -np.inf
    best_domain = None
    best_values = None
    
    # Try different K
    for K in [1000, 2000, 5000, 10000]:
        if time.time() > deadline - 10:
            break
            
        mu = compute_mobius(K)
        
        # Calculate S = sum mu(k)/k
        k_arr = np.arange(1, K + 1)
        S = np.sum(mu[1:K+1] / k_arr)
        
        # Construct f
        f = mu[1:K+1].astype(np.float64)
        f[0] = 1.0 - S # f(1)
        
        # Clip
        f = np.clip(f, -10, 10)
        
        # Check feasibility
        if is_feasible(K, mu, S):
            # Calculate score a = -sum f(k) log(k) / k
            # Note: f(1) is modified.
            # The evaluator uses the modified f.
            # a = -sum_{k=1}^K f(k) log(k) / k
            # log(1) = 0, so f(1) term is 0.
            a = -np.sum(f * np.log(k_arr) / k_arr)
            
            if a > best_score:
                best_score = a
                best_domain = k_arr
                best_values = f
                # print(f"K={K}, S={S}, Score={a}")
        else:
            # print(f"K={K} infeasible")
            pass

    if best_domain is None:
        # Fallback to original
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_domain, best_values

# EVOLVE-BLOCK-END

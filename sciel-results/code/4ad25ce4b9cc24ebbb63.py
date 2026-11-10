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
    safety_margin = 30.0  # seconds

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

    # Optimized feasibility check using O(N log N) algorithm
    # M(x) = sum_k f(k) floor(x/k)
    # After evaluator adjustment:
    # f_new(1) = f(1) - S, where S = sum_k f(k)/k
    # M_new(x) = f_new(1) * x + sum_{k=2}^K f(k) floor(x/k)
    #          = (f(1) - S) x + sum_{k=2}^K f(k) floor(x/k)
    #          = f(1) x - S x + sum_{k=2}^K f(k) floor(x/k)
    
    # If we choose f(k) = -mu(k) for k >= 2, and f(1) = 0:
    # S = 0 + sum_{k=2}^K -mu(k)/k = - (sum_{k=1}^K mu(k)/k - 1) = 1 - S_mu(K)
    # where S_mu(K) = sum_{k=1}^K mu(k)/k.
    
    # M_new(x) = 0*x - (1 - S_mu(K)) x + sum_{k=2}^K -mu(k) floor(x/k)
    #          = (S_mu(K) - 1) x - sum_{k=2}^K mu(k) floor(x/k)
    
    # We know sum_{k=1}^x mu(k) floor(x/k) = 1 for all x.
    # sum_{k=1}^K mu(k) floor(x/k) = 1 - sum_{k=K+1}^x mu(k) floor(x/k) (for x > K)
    # For x <= K, sum_{k=1}^K mu(k) floor(x/k) = 1.
    
    # Let's define R(x) = sum_{k=K+1}^x mu(k) floor(x/k) for x > K, and 0 otherwise.
    # Then sum_{k=1}^K mu(k) floor(x/k) = 1 - R(x).
    # sum_{k=2}^K mu(k) floor(x/k) = (sum_{k=1}^K mu(k) floor(x/k) - mu(1) floor(x/1))
    #                              = (1 - R(x)) - x
    #                              = 1 - R(x) - x
    
    # So M_new(x) = (S_mu(K) - 1) x - (1 - R(x) - x)
    #             = S_mu(K) x - x - 1 + R(x) + x
    #             = S_mu(K) x - 1 + R(x)
    
    # Constraint: M_new(x) <= 1.0001
    # S_mu(K) x + R(x) - 1 <= 1.0001
    # S_mu(K) x + R(x) <= 2.0001
    
    # For x <= K: R(x) = 0.
    # S_mu(K) x <= 2.0001.
    # Max x is K. So S_mu(K) K <= 2.0001.
    
    # For x > K: R(x) = sum_{k=K+1}^x mu(k) floor(x/k).
    # S_mu(K) x + R(x) <= 2.0001.
    
    # Objective: a = -sum_k f(k) log(k)/k
    # f(1) = 0 => term is 0.
    # a = - sum_{k=2}^K (-mu(k)) log(k)/k = sum_{k=2}^K mu(k) log(k)/k.
    # We know sum_{k=1}^infty mu(k) log(k)/k = -1.
    # So a = -1 - mu(1)log(1)/1 + tail = -1 + sum_{k=K+1}^infty mu(k) log(k)/k.
    # As K -> infinity, a -> -1.
    # Wait, we want to maximize a.
    # Is there a better choice?
    
    # Let's re-evaluate.
    # a = -sum f(k) log(k)/k.
    # If f(k) = mu(k) for k>=2:
    # S = 1 + sum_{k=2}^K mu(k)/k = S_mu(K).
    # f_new(1) = 1 - S_mu(K).
    # M_new(x) = (1 - S_mu(K)) x + sum_{k=2}^K mu(k) floor(x/k)
    #          = x - S_mu(K) x + (1 - R(x) - x)
    #          = 1 - R(x) - S_mu(K) x.
    # Constraint: 1 - R(x) - S_mu(K) x <= 1.0001
    # - R(x) - S_mu(K) x <= 0.0001
    # R(x) + S_mu(K) x >= -0.0001.
    
    # For x <= K, R(x)=0. S_mu(K) x >= -0.0001.
    # S_mu(K) is small positive (approx 0.6 for large K? No, sum mu/k -> 0).
    # Actually sum_{k=1}^K mu(k)/k converges to 0.
    # For K=100, S_mu ~ 0.5? No.
    # Let's check values.
    # If S_mu(K) is positive, then for x <= K, condition is satisfied if S_mu(K) x >= -0.0001.
    # Since S_mu(K) > 0 and x > 0, LHS > 0 > -0.0001. So feasible for x <= K.
    # For x > K: R(x) + S_mu(K) x >= -0.0001.
    # R(x) = sum_{k=K+1}^x mu(k) floor(x/k).
    # This term can be negative.
    
    # Objective for f(k)=mu(k):
    # a = -sum_{k=1}^K mu(k) log(k)/k.
    # sum mu(k) log(k)/k = -1.
    # So a = -(-1) = 1.
    # This is the theoretical maximum!
    
    # So the strategy is:
    # 1. Choose K such that the feasibility condition holds for x in [1, 10K].
    # 2. Set f(k) = mu(k) for k=1..K.
    # 3. Return domain, values.
    
    # The critical check is max_{x in [1, 10K]} (R(x) + S_mu(K) x) >= -0.0001.
    # Or equivalently: min_{x in [1, 10K]} (R(x) + S_mu(K) x) >= -0.0001.
    
    # Let's compute this.
    # We need R(x) for x in K+1 .. 10K.
    # R(x) = sum_{k=K+1}^x mu(k) floor(x/k).
    # This is similar to computing Mertens function but weighted.
    # Note that sum_{d|n} mu(d) = 1 if n=1 else 0.
    # sum_{k=1}^x mu(k) floor(x/k) = 1.
    # sum_{k=1}^K mu(k) floor(x/k) + sum_{k=K+1}^x mu(k) floor(x/k) = 1.
    # R(x) = 1 - sum_{k=1}^K mu(k) floor(x/k).
    
    # So we need to check:
    # min_{x in [1, 10K]} (1 - sum_{k=1}^K mu(k) floor(x/k) + S_mu(K) x) >= -0.0001.
    # Let T(x) = sum_{k=1}^K mu(k) floor(x/k).
    # Check: 1 - T(x) + S_mu(K) x >= -0.0001.
    # T(x) - S_mu(K) x <= 1.0001.
    
    # This is much easier to compute!
    # We just need to compute T(x) for x in [1, 10K] and check the max of T(x) - S_mu(K) x.
    
    def is_feasible_mu(K, mu):
        N = 10 * K
        # T(x) = sum_{k=1}^K mu(k) floor(x/k)
        # We can compute this efficiently.
        # T(x) = sum_{n=1}^x (sum_{d|n, d<=K} mu(d))
        # Let g(n) = sum_{d|n, d<=K} mu(d).
        # T(x) = sum_{n=1}^x g(n).
        
        g = np.zeros(N + 1, dtype=np.int64)
        for d in range(1, K + 1):
            md = mu[d]
            if md != 0:
                g[d:N+1:d] += md
        
        T = np.cumsum(g)
        x = np.arange(1, N + 1)
        
        k_arr = np.arange(1, K + 1)
        S_mu = np.sum(mu[1:K+1] / k_arr)
        
        # Check T(x) - S_mu * x <= 1.0001
        vals = T[1:N+1] - S_mu * x
        return np.max(vals) <= 1.0001

    best_score = -np.inf
    best_domain = None
    best_values = None

    # Try increasing K.
    # Feasibility depends on the error term.
    # For large K, S_mu(K) is small.
    # T(x) for x <= K is 1.
    # T(x) - S_mu x = 1 - S_mu x.
    # Max at x=1: 1 - S_mu. Since S_mu > 0, this is < 1.
    # For x > K, T(x) fluctuates around 1? No, T(x) is the partial sum.
    # The condition is T(x) - S_mu x <= 1.0001.
    
    # Let's test K values.
    for K in [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() > deadline - safety_margin:
            break
        
        mu = compute_mobius(K)
        
        if is_feasible_mu(K, mu):
            # Calculate score
            k_arr = np.arange(1, K + 1)
            f = mu[1:K+1].astype(np.float64)
            
            # Clip
            f = np.clip(f, -10, 10)
            
            # a = -sum f(k) log(k) / k
            a = -np.sum(f * np.log(k_arr) / k_arr)
            
            if a > best_score:
                best_score = a
                best_domain = k_arr
                best_values = f

    if best_domain is None:
        # Fallback: use a very small K or simple function
        # K=1: f(1)=1. S=1. f_new(1)=0. M(x)=0. Feasible. a=0.
        domain = np.array([1], dtype=np.int64)
        values = np.array([1.0], dtype=np.float64)
        return domain, values

    return best_domain, best_values

# EVOLVE-BLOCK-END

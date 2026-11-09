# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    time_limit = 600.0
    
    # Strategy: Use Linear Programming to find the optimal f(k) for k > 1
    # subject to the constraint on M(x) and bounds [-10, 10].
    
    # Choose K. Larger K gives better score but takes more time/memory.
    # K=2000 is a good balance.
    K = 2000
    
    # If we are already close to the limit, fallback
    if time.time() - start_time > time_limit * 0.5:
        return _fallback_mobius()

    try:
        # Variables: f(2), ..., f(K)
        n_vars = K - 1
        
        # Objective: Minimize sum( f(k) * log(k) / k )
        # which is equivalent to Maximize -sum( f(k) * log(k) / k )
        k_vals = np.arange(2, K + 1, dtype=np.float64)
        c = np.log(k_vals) / k_vals
        
        # Constraints:
        # M_eval(x) = sum_{k=2}^K f(k) * (floor(x/k) - x/k) <= 1.0
        # for all x in [1, 10K]
        
        x_vals = np.arange(1, 10 * K + 1, dtype=np.float64)
        n_constraints = len(x_vals)
        
        # Construct A matrix
        # A[i, j] corresponds to x = x_vals[i], k = k_vals[j]
        # A[i, j] = floor(x_vals[i] / k_vals[j]) - x_vals[i] / k_vals[j]
        
        # To save memory and time, we can compute this in chunks or use broadcasting if memory allows.
        # 20000 x 1999 float64 is ~320MB. Should be okay.
        
        # Broadcasting: x_vals[:, None] / k_vals[None, :]
        # This creates a 20000 x 1999 array.
        
        if time.time() - start_time > time_limit * 0.8:
            return _fallback_mobius()

        # Compute the matrix
        # Using np.floor and subtraction
        # Note: x/k can be large, but float64 is precise enough for integers up to 20000.
        
        # We can optimize by noting that floor(x/k) - x/k is the fractional part negated?
        # No, floor(x/k) - x/k = - (x/k - floor(x/k)) = - frac(x/k).
        # So A[i, j] = - (x_vals[i] / k_vals[j]) % 1.0
        
        # Let's compute -frac(x/k)
        # Broadcasting might be heavy. Let's try.
        
        # Alternative: Loop over k to build columns?
        # Or just do it in one go if memory permits.
        
        # Let's try the broadcasting approach.
        # If MemoryError, we fall back.
        
        try:
            # x_vals is shape (N, 1), k_vals is shape (1, M)
            ratios = x_vals[:, None] / k_vals[None, :]
            # frac(r) = r - floor(r)
            # We want -frac(r)
            A = -(ratios - np.floor(ratios))
        except MemoryError:
            return _fallback_mobius()

        b_ub = np.ones(n_constraints) * 1.0
        
        bounds = [(-10.0, 10.0)] * n_vars
        
        # Solve LP
        res = linprog(c, A_ub=A, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            return _fallback_mobius()
        
        f_values = res.x
        
        # Construct domain and values
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = np.zeros(K, dtype=np.float64)
        values[1:] = f_values
        
        # Double check time
        if time.time() - start_time > time_limit * 0.9:
             # If we are cutting it close, maybe just return what we have?
             # The evaluation itself takes time.
             # m * 10K operations.
             # 2000 * 20000 = 40M ops. Should be fast in numpy.
             pass

        return domain, values

    except Exception:
        return _fallback_mobius()

def _fallback_mobius():
    """
    Fallback to the known good Möbius function approximation.
    """
    # The seed program used a small set.
    # A better fallback is the actual Möbius function up to some K.
    # But the seed was score 1.0.
    # Let's use the seed's values or a slightly larger Möbius set.
    
    # Calculate Möbius function up to 1000
    K = 1000
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    # Sieve of Eratosthenes
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, K + 1, i):
                is_prime[j] = False
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                break
                
    # Compute mu
    mu = np.ones(K + 1, dtype=np.float64)
    # mu(1) = 1
    for i in range(2, K + 1):
        if is_prime[i]:
            mu[i] = -1
            for j in range(i * i, K + 1, i):
                mu[j] = -mu[j]
                if (j // i) % i == 0:
                    mu[j] = 0
                    # Actually, standard sieve for mu is easier:
                    # mu(n) = 0 if n has squared prime factor
                    # mu(n) = (-1)^k if n is product of k distinct primes
                    
    # Simpler mu calculation
    mu = np.ones(K + 1, dtype=np.int64)
    for i in range(2, K + 1):
        if is_prime[i]:
            for j in range(i, K + 1, i):
                mu[j] = -mu[j]
            for j in range(i * i, K + 1, i * i):
                mu[j] = 0
                
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    # f(1) is handled by evaluator, but we return 0 or mu(1)?
    # The evaluator adjusts f(1).
    # If we return mu(1)=1, it will be adjusted.
    # Let's return 0 for f(1) to be safe, or mu(1).
    # The seed returned specific values.
    # Let's just return the mu values.
    values[0] = 0.0 # f(1) will be set by evaluator logic anyway, but 0 is a safe start
    
    return domain, values

# EVOLVE-BLOCK-END

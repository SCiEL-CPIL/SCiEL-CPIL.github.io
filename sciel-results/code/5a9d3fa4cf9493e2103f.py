# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def compute_mobius(n):
    """Compute Möbius function mu(k) for k = 0..n using linear sieve."""
    mu = np.zeros(n + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
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

def check_feasibility_fast(f, K, max_x):
    """Check if M(x) <= 1.0001 for all x in [1, max_x] using difference array."""
    diff = np.zeros(max_x + 2, dtype=np.float64)
    for k in range(1, K + 1):
        fk = f[k]
        if fk == 0:
            continue
        j_max = max_x // k
        if j_max == 0:
            continue
        for j in range(1, j_max + 1):
            start = j * k
            end = min((j + 1) * k - 1, max_x)
            diff[start] += fk * j
            diff[end + 1] -= fk * j
    M = np.cumsum(diff)
    return np.max(M[1:max_x + 1]) <= 1.0001

def solve_lp(K, time_limit):
    """Solve LP for given K. Returns (domain, values, score) or None."""
    n_vars = K - 1  # f(2)..f(K)
    if n_vars <= 0:
        return None
    
    ks = np.arange(2, K + 1, dtype=np.float64)
    c = np.log(ks) / ks  # minimize sum c_k * g_k
    
    max_x = 10 * K
    
    # Build constraint matrix: M(x) <= 1 for all x in [1, max_x]
    # M(x) = f(1)*x + sum_{k=2}^{K} f(k)*floor(x/k)
    # With sum f(k)/k = 0 => f(1) = -sum_{k=2}^{K} f(k)/k
    # M(x) = -x*sum_{k=2}^{K} f(k)/k + sum_{k=2}^{K} f(k)*floor(x/k)
    #       = sum_{k=2}^{K} f(k)*(floor(x/k) - x/k)
    #       = -sum_{k=2}^{K} f(k)*{x/k}
    # So constraint: -sum_{k=2}^{K} f(k)*{x/k} <= 1
    
    # For efficiency, build rows incrementally
    A_rows = []
    for x in range(1, max_x + 1):
        row = np.empty(n_vars, dtype=np.float64)
        for i, k in enumerate(ks):
            row[i] = -((x % k) / k)
        A_rows.append(row)
    
    A_ub = np.array(A_rows)
    b_ub = np.ones(max_x, dtype=np.float64)
    
    bounds = [(-10, 10)] * n_vars
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'time_limit': time_limit})
        if res.success:
            g = res.x
            f = np.zeros(K + 1, dtype=np.float64)
            f[2:K + 1] = g
            S = np.sum(f[2:K + 1] / ks)
            f[1] = -S
            
            # Verify feasibility
            if not check_feasibility_fast(f, K, max_x):
                return None
            
            # Compute score
            a = -np.sum(f[2:K + 1] * c)
            
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = f[1:K + 1]
            values = np.clip(values, -10, 10)
            
            return domain, values, a
    except Exception:
        pass
    
    return None

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try LP for increasing K values
    # Memory: K=500 -> 5000x499 ~ 20MB, K=1000 -> 10000x999 ~ 80MB, K=2000 -> 20000x1999 ~ 320MB
    K_list = [100, 200, 500, 1000, 2000]
    
    for K in K_list:
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        if remaining < 60:
            break
        
        # Allocate time for LP solve
        lp_time = min(remaining - 30, 300)
        if lp_time < 10:
            break
        
        result = solve_lp(K, lp_time)
        
        if result is not None:
            domain, values, a = result
            if a > best_score:
                best_score = a
                best_domain = domain
                best_values = values
    
    # If LP didn't give good results, try Möbius as fallback
    if best_domain is None or best_score < 0.5:
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            max_K = 200000
            mu = compute_mobius(max_K)
            
            for K in [200000, 100000, 50000, 20000, 10000, 5000]:
                if time.time() > deadline - 10:
                    break
                max_x = 10 * K
                f = np.zeros(K + 1, dtype=np.float64)
                f[1:K + 1] = mu[1:K + 1].astype(np.float64)
                
                if check_feasibility_fast(f, K, max_x):
                    mask = mu[1:K + 1] != 0
                    domain = np.nonzero(mask)[0] + 1
                    values = mu[1:K + 1][mask].astype(np.float64)
                    
                    log_k = np.log(domain.astype(np.float64))
                    a = -np.sum(values * log_k / domain.astype(np.float64))
                    
                    if a > best_score:
                        best_score = a
                        best_domain = domain
                        best_values = values
                    break
    
    # Fallback
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values

# EVOLVE-BLOCK-END

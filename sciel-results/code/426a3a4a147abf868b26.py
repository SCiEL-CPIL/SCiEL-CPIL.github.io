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

def check_feasibility_diff(f, K, max_x):
    """Check feasibility using difference array - O(K * max_x / K) = O(max_x * H(K))"""
    # M(x) = sum_{k=1}^{K} f(k) * floor(x/k)
    # floor(x/k) = j for x in [j*k, (j+1)*k - 1]
    # So f(k)*j is added to M(x) for x in [j*k, min((j+1)*k-1, max_x)]
    
    diff = np.zeros(max_x + 2, dtype=np.float64)
    for k in range(1, K + 1):
        fk = f[k]
        if fk == 0:
            continue
        j = 1
        while j * k <= max_x:
            start = j * k
            end = min((j + 1) * k - 1, max_x)
            diff[start] += fk * j
            diff[end + 1] -= fk * j
            j += 1
    
    M = np.cumsum(diff)
    return np.max(M[1:max_x + 1]) <= 1.0001

def solve_small_lp(K, time_limit):
    """Solve LP for small K to get optimal values."""
    n_vars = K - 1
    if n_vars <= 0:
        return None
    
    ks = np.arange(2, K + 1, dtype=np.float64)
    c = np.log(ks) / ks
    
    max_x = 10 * K
    
    A_rows = np.zeros((max_x, n_vars), dtype=np.float64)
    for x in range(1, max_x + 1):
        for i, k in enumerate(ks):
            A_rows[x-1, i] = -((x % k) / k)
    
    b_ub = np.ones(max_x, dtype=np.float64)
    bounds = [(-10, 10)] * n_vars
    
    try:
        res = linprog(c, A_ub=A_rows, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'time_limit': time_limit, 'presolve': True})
        if res.success:
            g = res.x
            f = np.zeros(K + 1, dtype=np.float64)
            f[2:K + 1] = g
            S = np.sum(f[2:K + 1] / ks)
            f[1] = -S
            
            if not check_feasibility_diff(f, K, max_x):
                return None
            
            a = -np.sum(f[2:K + 1] * c)
            
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = f[1:K + 1]
            values = np.clip(values, -10, 10)
            
            return domain, values, a
    except Exception:
        pass
    
    return None

def mobius_solution(K):
    """Generate solution using Möbius function."""
    mu = compute_mobius(K)
    
    mask = mu[1:K + 1] != 0
    domain = np.nonzero(mask)[0] + 1
    values = mu[1:K + 1][mask].astype(np.float64)
    
    max_x = 10 * K
    f = np.zeros(K + 1, dtype=np.float64)
    f[1:K + 1] = mu[1:K + 1].astype(np.float64)
    
    if not check_feasibility_diff(f, K, max_x):
        return None
    
    log_k = np.log(domain.astype(np.float64))
    a = -np.sum(values * log_k / domain.astype(np.float64))
    
    return domain, values, a

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 570
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Strategy 1: Small LP
    for K in [50, 100, 200]:
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        if remaining < 30:
            break
        
        lp_time = min(remaining - 20, 120)
        if lp_time < 10:
            break
        
        result = solve_small_lp(K, lp_time)
        
        if result is not None:
            domain, values, a = result
            if a > best_score:
                best_score = a
                best_domain = domain
                best_values = values
    
    # Strategy 2: Möbius
    for K in [200000, 100000, 50000, 20000, 10000, 5000, 2000]:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        result = mobius_solution(K)
        
        if result is not None:
            domain, values, a = result
            if a > best_score:
                best_score = a
                best_domain = domain
                best_values = values
    
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values

# EVOLVE-BLOCK-END

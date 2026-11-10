# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


def _compute_mobius(K):
    """Compute the Mobius function mu(n) for n=1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
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


def _compute_M_fast(domain, values, max_x):
    """
    Compute M(x) = sum_k f(k) * floor(x/k) for x in [1, max_x]
    using an optimized difference-array approach.
    """
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for idx in range(len(domain)):
        k = int(domain[idx])
        f_k = values[idx]
        if f_k == 0.0:
            continue
        max_j = max_x // k
        if max_j == 0:
            continue
        diff[k : (max_j + 1) * k : k] += f_k
    
    M = np.cumsum(diff)
    return M[:max_x + 1]


def _solve_lp_correct(K, time_limit=200):
    """
    Solve LP in g(k) (clipped values) to maximize -sum_{k=2}^K g(k)*log(k)/k
    subject to sum_{k=1}^K g(k)*(floor(x/k) - x/k) <= 1 for x=1..10K,
    and -10 <= g(k) <= 10.
    """
    start_time = time.time()
    n_vars = K
    max_x = 10 * K
    
    # Objective: minimize sum_{k=2}^K g(k)*log(k)/k
    c = np.zeros(n_vars)
    for k in range(2, K + 1):
        c[k - 1] = np.log(k) / k
    
    # Build A_ub: row x has coeff floor(x/k) - x/k for each k
    A_ub = lil_matrix((max_x, n_vars), dtype=np.float64)
    
    for x in range(1, max_x + 1):
        if time.time() - start_time > time_limit - 30:
            return None
        for k in range(1, K + 1):
            coeff = np.floor(x / k) - x / k
            if abs(coeff) > 1e-15:
                A_ub[x - 1, k - 1] = coeff
    
    A_ub = A_ub.tocsr()
    b_ub = np.ones(max_x, dtype=np.float64)
    bounds = [(-10.0, 10.0)] * n_vars
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        return None
    
    g = result.x
    S = np.sum(g / np.arange(1, K + 1, dtype=np.float64))
    h = g.copy()
    h[0] -= S
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    M = _compute_M_fast(domain, h, max_x)
    
    if np.max(M[1:max_x + 1]) > 1.0001:
        return None
    
    k_float = np.arange(1, K + 1, dtype=np.float64)
    a = -np.sum(h * np.log(k_float) / k_float)
    
    return (domain, g, a)


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Möbius baseline with K=200000
    K_mob = 200000
    mu = _compute_mobius(K_mob)
    mask = mu[1:K_mob + 1] != 0
    domain_mob = np.where(mask)[0] + 1
    values_mob = mu[domain_mob].astype(np.float64)
    
    clipped = np.clip(values_mob, -10, 10)
    k_float = domain_mob.astype(np.float64)
    S = np.sum(clipped / k_float)
    adjusted = clipped.copy()
    if domain_mob[0] == 1:
        adjusted[0] -= S
    else:
        adjusted = np.concatenate([[-S], clipped])
        domain_mob = np.concatenate([[1], domain_mob])
    k_float = domain_mob.astype(np.float64)
    a_mob = -np.sum(adjusted * np.log(k_float) / k_float)
    
    best_result = (domain_mob, values_mob)
    best_a = a_mob
    
    # Try LP for small K
    for K in [200, 500]:
        remaining = deadline - (time.time() - start_time)
        if remaining < 60:
            break
        time_limit = min(remaining - 30, 200)
        result = _solve_lp_correct(K, time_limit=time_limit)
        if result is not None:
            domain_lp, values_lp, a_lp = result
            if a_lp > best_a:
                best_a = a_lp
                best_result = (domain_lp, values_lp)
    
    return best_result


# EVOLVE-BLOCK-END

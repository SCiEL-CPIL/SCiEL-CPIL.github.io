# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _compute_mobius(K):
    """Compute the Möbius function mu(n) for n = 1..K using sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    
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


def _compute_M_values(domain, values, x_array):
    """Compute M(x) = sum_k f(k) * floor(x/k) for each x in x_array."""
    m = len(domain)
    M_vals = np.zeros(len(x_array))
    for i in range(m):
        k = int(domain[i])
        if k == 0:
            continue
        M_vals += values[i] * (x_array // k)
    return M_vals


def _compute_a(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k."""
    k_float = domain.astype(np.float64)
    return -np.sum(values * np.log(k_float) / k_float)


def _solve_lp(domain, K, x_max, tolerance=1.0001, time_budget=300):
    """
    Solve the LP to maximize a = -sum f(k)*log(k)/k
    subject to M(x) <= tolerance for all x in [1, x_max].
    Uses cutting-plane method.
    """
    start_time = time.time()
    m = len(domain)
    if m == 0:
        return None
    
    # Objective: maximize -sum f(k)*log(k)/k => minimize sum f(k)*log(k)/k
    c = np.log(domain.astype(np.float64)) / domain.astype(np.float64)
    
    # Bounds: f(k) in [-10, 10] (clipping happens in evaluation, but we can use these bounds)
    bounds = [(-10.0, 10.0)] * m
    
    # Start with a coarse set of x constraints
    # The most constraining x values are likely small ones or multiples
    # Start with x = 1, 2, ..., min(100, x_max) and some larger ones
    n_coarse = min(200, x_max)
    x_coarse = np.arange(1, n_coarse + 1, dtype=np.int64)
    if x_max > n_coarse:
        # Add some larger x values
        x_large = np.linspace(n_coarse + 1, x_max, min(100, x_max - n_coarse), dtype=np.int64)
        x_constraint_set = np.concatenate([x_coarse, x_large])
    else:
        x_constraint_set = x_coarse
    
    # Build A_ub for initial constraints
    A_rows = []
    b_vals = []
    for x in x_constraint_set:
        row = (x // domain).astype(np.float64)
        A_rows.append(row)
        b_vals.append(tolerance)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    x_opt = None
    for iteration in range(50):
        if time.time() - start_time > time_budget:
            break
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            break
        
        if not res.success:
            break
        
        x_opt = res.x
        
        # Check all x in [1, x_max] for violations
        # Sample strategically: check all x up to some limit, then sample
        check_limit = min(x_max, 50000)
        x_check = np.arange(1, check_limit + 1, dtype=np.int64)
        if x_max > check_limit:
            x_sample = np.linspace(check_limit + 1, x_max, min(5000, x_max - check_limit), dtype=np.int64)
            x_check = np.concatenate([x_check, x_sample])
        
        M_vals = _compute_M_values(domain, x_opt, x_check)
        violations = M_vals > tolerance + 1e-9
        
        if not np.any(violations):
            break
        
        # Add violated constraints (up to a batch size)
        viol_idx = np.where(violations)[0]
        max_batch = 500
        if len(viol_idx) > max_batch:
            viol_idx = viol_idx[:max_batch]
        
        for idx in viol_idx:
            x = x_check[idx]
            row = (x // domain).astype(np.float64)
            A_ub = np.vstack([A_ub, row.reshape(1, -1)])
            b_ub = np.append(b_ub, tolerance)
        
        # Limit total constraints to avoid memory issues
        if len(b_ub) > 50000:
            break
    
    if x_opt is None:
        return None
    
    return x_opt


def prime_number_theorem_partial_function():
    """
    Construct a partial function f that maximizes a = -sum_k f(k)*log(k)/k
    subject to M(x) = sum_k f(k)*floor(x/k) <= 1.0001 for all x in [1, 10K].
    """
    start_time = time.time()
    deadline = start_time + 580
    
    best_a = -np.inf
    best_domain = None
    best_values = None
    
    K_candidates = [50000, 20000, 10000]
    
    for K in K_candidates:
        if time.time() > deadline - 60:
            break
        
        mu = _compute_mobius(K)
        
        # Use all k where mu(k) != 0
        mask = mu[1:] != 0
        domain = np.where(mask)[0] + 1
        values_mu = mu[domain].astype(np.float64)
        
        # Try LP solve
        x_max = 10 * K
        time_for_lp = max(30, min(300, int(deadline - time.time() - 30)))
        
        x_opt = _solve_lp(domain, K, x_max, tolerance=1.0001, time_budget=time_for_lp)
        
        if x_opt is not None:
            # Verify feasibility
            M_max_check = _compute_M_values(domain, x_opt, np.arange(1, min(x_max + 1, 200000), dtype=np.int64))
            if np.max(M_max_check) <= 1.0001:
                a_val = _compute_a(domain, x_opt)
                if a_val > best_a:
                    best_a = a_val
                    best_domain = domain.copy()
                    best_values = x_opt.copy()
        
        # Also keep the Möbius solution as fallback
        a_mu = _compute_a(domain, values_mu)
        if a_mu > best_a:
            # Quick feasibility check
            M_check = _compute_M_values(domain, values_mu, np.arange(1, min(x_max + 1, 200000), dtype=np.int64))
            if np.max(M_check) <= 1.0001:
                best_a = a_mu
                best_domain = domain.copy()
                best_values = values_mu.copy()
    
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values


# EVOLVE-BLOCK-END

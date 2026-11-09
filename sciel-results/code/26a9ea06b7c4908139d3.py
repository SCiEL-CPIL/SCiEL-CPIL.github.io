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

def build_lp_matrices(K, x_samples):
    """
    Build the LP matrices for the problem.
    Variables: f(2), f(3), ..., f(K)  (f(1) is determined by the constraint sum f(k)/k = 0)
    
    We maximize a = -sum_{k=2}^{K} f(k) * log(k) / k
    Subject to:
      1) M(x) = f(1) + sum_{k=2}^{min(x,K)} f(k) * floor(x/k) <= 1 for x in x_samples
         where f(1) = -sum_{k=2}^{K} f(k)/k (from the zero-sum constraint)
         So: M(x) = -sum_{k=2}^{K} f(k)/k + sum_{k=2}^{min(x,K)} f(k) * floor(x/k)
              = sum_{k=2}^{K} f(k) * (floor(x/k) - 1/k)  for k <= x
              = sum_{k=2}^{x} f(k) * (floor(x/k) - 1/k)  for k > x (but floor(x/k)=0 for k>x)
              = sum_{k=2}^{min(x,K)} f(k) * (floor(x/k) - 1/k)
    
      2) No explicit bounds on f(k) other than the M(x) constraints
    """
    n_vars = K - 1  # f(2) through f(K)
    
    # Objective: maximize -sum f(k)*log(k)/k => minimize sum f(k)*log(k)/k
    # c = log(k)/k for k=2..K
    c = np.zeros(n_vars)
    for i in range(n_vars):
        k = i + 2
        c[i] = np.log(k) / k
    
    # Constraints: M(x) <= 1 for each x in x_samples
    # M(x) = sum_{k=2}^{min(x,K)} f(k) * (floor(x/k) - 1/k)
    A_ub = np.zeros((len(x_samples), n_vars))
    b_ub = np.ones(len(x_samples))
    
    for idx, x in enumerate(x_samples):
        k_max = min(x, K)
        for i in range(n_vars):
            k = i + 2
            if k <= k_max:
                A_ub[idx, i] = x // k - 1.0 / k
            # else: floor(x/k) = 0, so contribution is 0 - 1/k... wait
            # Actually for k > x: floor(x/k) = 0, so term is f(k)*(0 - 1/k) = -f(k)/k
            # But wait, the constraint is sum_{k=2}^{K} f(k)/k = -f(1), and M(x) includes f(1)*floor(x/1) = f(1)
            # M(x) = f(1)*1 + sum_{k=2}^{min(x,K)} f(k)*floor(x/k)
            #      = -sum_{k=2}^{K} f(k)/k + sum_{k=2}^{min(x,K)} f(k)*floor(x/k)
            #      = sum_{k=2}^{K} f(k) * (floor(x/k) - 1/k) where floor(x/k)=0 if k>x
            # So for k > x: coefficient is 0 - 1/k = -1/k
    
    # Fix the coefficients for k > x
    for idx, x in enumerate(x_samples):
        k_max = min(x, K)
        for i in range(n_vars):
            k = i + 2
            if k > k_max:
                A_ub[idx, i] = -1.0 / k
    
    return c, A_ub, b_ub

def solve_lp(K, x_samples, deadline):
    """Solve the LP for given K and x_samples."""
    c, A_ub, b_ub = build_lp_matrices(K, x_samples)
    
    # No bounds on variables (they can be positive or negative)
    bounds = [(None, None)] * (K - 1)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        return None, None
    
    f_vals = res.x  # f(2), f(3), ..., f(K)
    
    # Compute f(1) = -sum f(k)/k
    f1 = -np.sum(f_vals / np.arange(2, K + 1).astype(np.float64))
    
    # Build domain and values
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = np.zeros(K + 1, dtype=np.float64)
    values[1] = f1
    values[2:K + 1] = f_vals
    
    # Remove zeros (and very small values)
    mask = np.abs(values[1:]) > 1e-10
    domain_out = np.nonzero(mask)[0] + 1
    values_out = values[1:][mask]
    
    if len(domain_out) < 1:
        return None, None
    
    # Compute score
    log_k = np.log(domain_out.astype(np.float64))
    a = -np.sum(values_out * log_k / domain_out.astype(np.float64))
    
    return domain_out, values_out

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 630s limit
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try different K values
    K_list = [50000, 30000, 20000, 10000, 5000]
    
    for K in K_list:
        if time.time() > deadline - 60:
            break
        
        max_x = 10 * K
        
        # Use a subset of x values to make the LP tractable
        # Start with a coarse grid, then refine
        x_samples = list(range(1, max_x + 1, max(1, max_x // 1000)))
        if max_x not in x_samples:
            x_samples.append(max_x)
        
        domain, values = solve_lp(K, x_samples, deadline)
        
        if domain is not None and len(domain) > 0:
            # Verify feasibility on all x in [1, max_x]
            feasible = True
            f_full = np.zeros(K + 1, dtype=np.float64)
            for i, k in enumerate(domain):
                f_full[k] = values[i]
            
            # Check M(x) for a sample of x values
            check_xs = list(range(1, max_x + 1, max(1, max_x // 500)))
            if max_x not in check_xs:
                check_xs.append(max_x)
            
            for x in check_xs:
                M_x = 0.0
                for i, k in enumerate(domain):
                    if k <= x:
                        M_x += values[i] * (x // k)
                if M_x > 1.0001:
                    feasible = False
                    break
            
            if feasible:
                log_k = np.log(domain.astype(np.float64))
                a = -np.sum(values * log_k / domain.astype(np.float64))
                
                if a > best_score:
                    best_score = a
                    best_domain = domain
                    best_values = values
    
    # Fallback
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values

# EVOLVE-BLOCK-END

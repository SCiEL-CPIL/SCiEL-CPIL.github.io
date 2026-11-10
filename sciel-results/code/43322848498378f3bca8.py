# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses Linear Programming to optimize the values f(k) subject to the constraints:
    1. sum_k f(k)/k = 0 (handled by evaluator's adjustment of f(1), but we model the effect)
    2. M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, 10K]
    3. -10 <= f(k) <= 10
    
    The objective is a = -sum_k f(k) * log(k) / k.
    Note that f(1) does not contribute to the objective (log(1)=0).
    """
    start_time = time.time()
    DEADLINE = 620.0  # safety margin before 630s

    def solve_lp(K):
        """
        Solve the LP for a given K.
        Returns (domain, values, score) or (None, None, -inf) if infeasible or timeout.
        """
        # Variables: h[1], h[2], ..., h[K]
        # Index 0 corresponds to k=1, index 1 to k=2, etc.
        n_vars = K
        
        # Objective: Maximize -sum_{k=2}^K h(k) * log(k) / k
        # linprog minimizes, so we minimize sum_{k=2}^K h(k) * log(k) / k
        c = np.zeros(n_vars)
        # c[0] is for k=1, log(1)=0, so c[0]=0
        for k in range(2, K + 1):
            idx = k - 1
            c[idx] = np.log(k) / k
            
        # Bounds: -10 <= h(k) <= 10
        bounds = [(-10.0, 10.0)] * n_vars
        
        # Constraints: M(x) <= 1.0001 for x in [1, 10K]
        # M(x) = h(1)(x-1) + sum_{k=2}^K h(k) (floor(x/k) - x/k) <= 1.0001
        # A_ub * h <= b_ub
        
        max_x = 10 * K
        # We can reduce constraints. M(x) is linear in h.
        # The maximum of M(x) over x occurs at specific points?
        # M(x) is a step function increasing with x generally?
        # Let's just include all x in [1, 10K].
        # If K is small (e.g., 200), 10K = 2000 rows.
        # If K is large (e.g., 2000), 10K = 20000 rows.
        # linprog can handle this.
        
        A_ub = np.zeros((max_x, n_vars))
        b_ub = np.ones(max_x) * 1.0001
        
        # Precompute coefficients
        # col 0 (k=1): coeff is (x-1)
        # col j (k=j+1): coeff is (floor(x/(j+1)) - x/(j+1))
        
        for x in range(1, max_x + 1):
            A_ub[x-1, 0] = x - 1.0
            for k in range(2, K + 1):
                idx = k - 1
                A_ub[x-1, idx] = np.floor(x / k) - x / k
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                h = res.x
                # Calculate score
                # a = -sum_{k=2}^K h(k) * log(k) / k
                # Note: res.fun is min sum h(k) log(k)/k, so a = -res.fun
                score = -res.fun
                
                # Construct domain and values
                # The evaluator expects distinct positive integers in domain.
                # We can return all k in 1..K.
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = h.astype(np.float64)
                
                return domain, values, score
            else:
                return None, None, -np.inf
        except Exception:
            return None, None, -np.inf

    best_domain = None
    best_values = None
    best_score = -np.inf

    # Try progressively larger K
    # Start with small K to get a quick result, then increase.
    K_candidates = [50, 100, 200, 500, 1000, 2000]
    
    for K in K_candidates:
        if time.time() > DEADLINE - 60:
            break
            
        domain, values, score = solve_lp(K)
        
        if domain is not None and score > best_score:
            best_score = score
            best_domain = domain
            best_values = values
            
        # If we found a score significantly better than 1.0, we might stop or continue to improve.
        # Given the LP nature, larger K might allow better optimization.
        
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback: Use Möbius function if LP fails or is too slow
    def compute_mobius(K):
        mu = np.ones(K + 1, dtype=np.float64)
        is_composite = np.zeros(K + 1, dtype=bool)
        primes = []
        for i in range(2, K + 1):
            if not is_composite[i]:
                primes.append(i)
                mu[i] = -1.0
            for p in primes:
                if i * p > K:
                    break
                is_composite[i * p] = True
                if i % p == 0:
                    mu[i * p] = 0.0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    K_fallback = 200000
    mu = compute_mobius(K_fallback)
    nonzero_mask = mu[1:K_fallback + 1] != 0.0
    domain = np.where(nonzero_mask)[0] + 1
    values = mu[1:K_fallback + 1][nonzero_mask].copy()
    
    return domain, values

# EVOLVE-BLOCK-END

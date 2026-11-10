# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time
import math


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k)*log(k)/k subject to M(x) <= 1.
    
    Strategy:
    1. Try solving the LP for increasing K values using HiGHS.
    2. If LP fails or times out, fall back to the Mobius function which is known
       to satisfy the constraints exactly (M(x)=1) and approach the theoretical limit.
    3. Maintain a strict deadline guard to ensure the function returns within 600s.
    """
    start_time = time.time()
    deadline = start_time + 550  # Safety margin before 600s limit

    def _mobius_sieve(n):
        """Compute Mobius function for 1..n using linear sieve."""
        mu = np.ones(n + 1, dtype=np.int64)
        mu[0] = 0
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[:2] = False
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

    def _compute_mobius_solution(K):
        """Return domain and values for the Mobius function up to K."""
        mu = _mobius_sieve(K)
        mask = mu != 0
        domain = np.where(mask)[0].astype(np.int64)
        values = mu[domain].astype(np.float64)
        
        # Ensure 1 is in domain (mu(1)=1)
        if len(domain) == 0 or domain[0] != 1:
            domain = np.concatenate([[1], domain])
            values = np.concatenate([[1.0], values])
            
        return domain, values

    # Fallback solution: Mobius function with K=200000
    # This is the theoretical optimum as K -> infinity
    best_domain, best_values = _compute_mobius_solution(200000)
    best_score = -np.sum(best_values * np.log(best_domain.astype(np.float64)) / best_domain.astype(np.float64))
    
    # Try LP for smaller K values to see if we can beat the Mobius approximation
    # The Mobius function at K=200000 should be very close to the true optimum (1.0)
    # But LP might find a better finite solution for specific small K if the Mobius
    # tail truncation causes issues. However, Mobius is exact for M(x)=1.
    # The objective a = -sum mu(k) log(k)/k converges to 1.
    # For K=200000, it should be very close to 1.
    
    # Let's try a few LP solutions to see if they are better.
    # Note: The LP formulation in inspiration 1 uses constraints M(x) <= 1.
    # The Mobius function satisfies M(x) = 1 exactly, so it is on the boundary.
    # The LP maximizes the objective, so it should find a solution at least as good
    # as the Mobius function if the Mobius function is feasible.
    # Since Mobius is feasible, LP should find a solution with score >= Mobius score.
    
    # However, LP with K=200000 is too large.
    # Let's try K=500, 1000, 2000 if time permits.
    
    K_candidates = [500, 1000, 2000]
    
    for K in K_candidates:
        if time.time() > deadline - 30:
            break
            
        try:
            n_vars = K - 1
            if n_vars == 0:
                continue
                
            # Objective: minimize sum v_k ln(k)/k  (so maximize -sum)
            ks = np.arange(2, K + 1)
            c = np.log(ks) / ks
            
            # Constraints: M(x) <= 1 for x in 1..10K
            # M(x) = sum_{k=2}^K f(k) * floor(x/k) + f(1)
            # Wait, the evaluator adjusts f(1) such that sum f(k)/k = 0.
            # Let S = sum_{k=2}^K f(k)/k. Then f(1) = -S.
            # M(x) = f(1) + sum_{k=2}^K f(k) * floor(x/k)
            #      = -sum_{k=2}^K f(k)/k + sum_{k=2}^K f(k) * floor(x/k)
            #      = sum_{k=2}^K f(k) * (floor(x/k) - 1/k)
            
            # So the constraint is:
            # sum_{k=2}^K f(k) * (floor(x/k) - 1/k) <= 1
            
            # This is a linear constraint in f(k).
            # We can build the matrix A_ub.
            
            # To save memory and time, we might not be able to build the full matrix
            # for large K. Let's stick to small K for LP.
            
            if K > 1000:
                # Building 10K x K matrix for K=2000 is 20000 x 1999 ~ 40M entries.
                # This might be slow to build or solve.
                # Let's skip K=2000 for now and just use K=500, 1000.
                pass
            else:
                x_max = 10 * K
                # A_ub has shape (x_max, n_vars)
                # A_ub[i, j] corresponds to x = i+1, k = j+2
                # A_ub[i, j] = floor((i+1) / (j+2)) - 1/(j+2)
                
                # We can vectorize this.
                x_vals = np.arange(1, x_max + 1)[:, np.newaxis]  # (x_max, 1)
                k_vals = ks[np.newaxis, :]  # (1, n_vars)
                
                A_ub = np.floor(x_vals / k_vals) - 1.0 / k_vals
                b_ub = np.ones(x_max)
                
                bounds = [(-10, 10)] * n_vars
                
                # Solve LP
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                
                if res.success:
                    v = res.x
                    # Calculate score
                    # a = -sum_{k=2}^K f(k) log(k)/k
                    # Note: f(1) term is 0 in the objective because log(1)=0.
                    score = -np.dot(v, c)
                    
                    if score > best_score:
                        best_score = score
                        domain = np.concatenate(([1], ks)).astype(np.int64)
                        values = np.zeros(K, dtype=np.float64)
                        values[1:] = v
                        best_domain = domain
                        best_values = values
                        
        except Exception:
            pass

    return best_domain, best_values


# EVOLVE-BLOCK-END

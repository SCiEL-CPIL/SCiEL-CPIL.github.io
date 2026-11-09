import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math
import sys

def construct():
    start_time = time.time()
    # Hard deadline: leave margin for the 630s kill limit.
    deadline = start_time + 580

    m = 12
    n = 1

    # Precomputation: Transformation matrix from standard basis to Laguerre basis
    # L_k(x) = sum_{j=0}^k A[k,j] x^j
    # P(x) = sum_{k=0}^{12} c_k L_k(x) = sum_{j=0}^{12} a_j x^j
    # a = A^T c  => c = (A^T)^{-1} a
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)

    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    # Weights for the Laguerre norm: <L_k, L_l> = delta_kl * (k+1) for alpha=1
    weights = np.arange(13) + 1

    def compute_cost(roots):
        """
        Compute the cost function (upper bound on C4) given 12 positive roots.
        Cost = 1 / ||P||^2_Laguerre
        """
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e20

        # Check for invalid roots
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e20

        # Build polynomial P(x) = prod(x - r_i)
        # np.poly returns coefficients in descending order [1, ...]
        coeffs_desc = np.poly(roots)
        # Convert to ascending order [a_0, a_1, ...]
        a = coeffs_desc[::-1]

        # Compute Laguerre coefficients c
        c = A_T_inv @ a

        # Compute norm squared in Laguerre basis
        norm_sq = np.sum(c**2 * weights)

        if norm_sq <= 1e-10:
            return 1e20

        return 1.0 / norm_sq

    best_val = 1e20
    best_roots = None

    # Roots bounds: [1e-2, 1000] seems reasonable for Laguerre roots
    # Laguerre roots for L_n^{(1)} are in (0, n+1+2sqrt(n(n+1))) approx (0, 2n+2)
    # But here we have a product of 12 terms, so roots can be larger?
    # The polynomial is degree 12. The "roots" are parameters in the product.
    # If we choose roots to be the zeros of L_{12}^{(1)}, they are in (0, 26).
    # Let's allow a wider range to be safe, e.g., [0.01, 100].
    bounds_roots = [(0.01, 100.0)] * 12

    def update_best(val, x):
        nonlocal best_val, best_roots
        if val < best_val:
            best_val = val
            best_roots = x.copy()

    # --- Phase 1: Global Search (Differential Evolution) ---
    # DE is robust for 12 dimensions.
    # We run multiple seeds or one long run.
    # Given 12 dims, DE is fast.
    
    # Use a Sobol sequence for initialization if possible, else 'latinhypercube'
    # scipy differential_evolution supports 'sobol' in newer versions, 'latinhypercube' is standard.
    
    for seed in [1, 2, 3]:
        if time.time() - start_time > deadline - 60:
            break
        try:
            res = differential_evolution(
                compute_cost,
                bounds_roots,
                seed=seed,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='latinhypercube'
            )
            
            # Polish the result
            if time.time() - start_time < deadline - 10:
                try:
                    res_polish = minimize(
                        compute_cost,
                        res.x,
                        method='L-BFGS-B',
                        bounds=bounds_roots,
                        options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-10}
                    )
                    val = res_polish.fun
                    x = res_polish.x
                except:
                    val = res.fun
                    x = res.x
            else:
                val = res.fun
                x = res.x

            update_best(val, x)
        except Exception:
            pass

    # --- Phase 2: Local Search from Diverse Initial Points ---
    # Start from known good points:
    # 1. Roots of L_12^(1) (approximate)
    # 2. Log-spaced roots
    # 3. Random roots
    
    rng = np.random.RandomState(42)
    
    initial_points = []
    
    # Roots of Laguerre L_12^(1)
    # We can find them by minimizing the norm of P for P = L_12? No, L_12 is a specific polynomial.
    # The roots of L_12^(1) are the optimal roots if we were constrained to be L_12?
    # Actually, the task is to find roots such that the product P has max norm.
    # The Laguerre polynomial itself has the property of being orthogonal, so its norm is fixed.
    # But we are free to choose any 12 roots.
    # It is likely that the optimal roots are close to the roots of L_12^(1) or some variation.
    # Let's get roots of L_12^(1) using numpy.polynomial.laguerre.laggauss?
    # laggauss returns nodes for integration, which are roots of L_n^(0).
    # We need roots of L_n^(1).
    # We can compute them via eigenvalues of the Jacobi matrix or just use a rough approximation.
    # Or simply start with a spread.
    
    # Approximate roots of L_12^(1):
    # They are roughly distributed in (0, 25).
    # Let's just use a log-spaced set and a linear spaced set.
    
    # Log spaced
    r_log = np.logspace(np.log10(0.1), np.log10(50), 12)
    initial_points.append(r_log)
    
    # Linear spaced
    r_lin = np.linspace(0.5, 30, 12)
    initial_points.append(r_lin)
    
    # Random
    for _ in range(10):
        r_rand = rng.uniform(0.1, 50, 12)
        initial_points.append(r_rand)
        
    # Clustered
    for center in [5, 10, 20]:
        r_clust = center + rng.normal(0, 2, 12)
        r_clust = np.maximum(r_clust, 0.1)
        initial_points.append(r_clust)

    for init_x in initial_points:
        if time.time() - start_time > deadline - 20:
            break

        try:
            # L-BFGS-B
            res_lf = minimize(
                compute_cost,
                init_x,
                method='L-BFGS-B',
                bounds=bounds_roots,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-9}
            )
            update_best(res_lf.fun, res_lf.x)

            # Nelder-Mead fine-tuning
            if time.time() - start_time < deadline - 5:
                res_nm = minimize(
                    compute_cost,
                    res_lf.x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                update_best(res_nm.fun, res_nm.x)

        except Exception:
            pass

    # --- Final Fallback ---
    if best_roots is None:
        best_roots = np.logspace(np.log10(0.1), np.log10(50), 12)
        best_val = compute_cost(best_roots)

    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-10)

    return list(best_roots)

# Main entry point
if __name__ == "__main__":
    # For local testing
    roots = construct()
    print(roots)

import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
import time
import math
import sys

def construct():
    start_time = time.time()
    # Hard deadline: leave margin for the 630s kill limit.
    # We target finishing within 580s to be safe.
    deadline = start_time + 580

    m = 12
    n = 1

    # Precomputation: Transformation matrix from standard basis to Laguerre basis
    # We want to represent P(x) = prod(x - r_i) in the Laguerre basis L_k^{(1)}.
    # P(x) = sum_{k=0}^{12} c_k L_k^{(1)}(x)
    # We need to compute c from the coefficients of P(x) in standard basis.
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k,j] = (-1)^j * C(k+1, j+1) / j!
    
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # p (coeffs in standard basis, ascending) and c (coeffs in Laguerre basis).
    # p = A^T c
    # So c = (A^T)^{-1} p.
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    # Weights for the Laguerre norm: <L_k, L_l> = delta_kl * (k+1) for alpha=1
    weights = np.arange(13) + 1

    def compute_cost(log_roots):
        """
        Compute the cost function (upper bound on C4) given log roots.
        Cost = 1 / ||P||^2_Laguerre
        We want to minimize this, which means maximizing ||P||^2.
        """
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e20

        roots = np.exp(log_roots)
        
        # Check validity
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e20

        # Build polynomial P(x) = prod(x - r_i)
        # np.poly(roots) returns coefficients in descending order: [1, -sum(r), sum(r_i r_j), ...]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # Ascending order: p_0, p_1, ..., p_12
        
        # Compute Laguerre coefficients c
        c = A_T_inv @ p
        
        # Compute norm squared in Laguerre basis
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 1e20
        
        return 1.0 / norm_sq

    # Optimization Strategy: Multi-start with diverse algorithms
    # We will generate many random initial points and use local optimizers (BFGS, Powell, Nelder-Mead)
    # as well as some global optimizers (DE, Dual Annealing) in parallel/sequence.

    best_val = 1e20
    best_log_roots = None

    # Define bounds for log roots
    # Roots must be positive.
    # Based on the previous seed [4, 8, 16, ...], log range is roughly [1.4, 9.2].
    # Let's allow a wider range to explore: [0, 12] for log(roots).
    # This corresponds to roots in [1, 1.6e5].
    bounds_log = [(0.0, 12.0)] * 12

    # Helper to update best
    def update_best(val, x):
        nonlocal best_val, best_log_roots
        if val < best_val:
            best_val = val
            best_log_roots = x.copy()

    # --- Phase 1: Global Search (Differential Evolution) ---
    # Run a few DE instances with different seeds and settings.
    # DE is good at finding good basins in high dimensions.
    de_seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    
    for seed in de_seeds:
        if time.time() - start_time > deadline - 60:
            break
        try:
            # Use a moderate population size
            res = differential_evolution(
                compute_cost, 
                bounds_log, 
                seed=seed, 
                maxiter=150, 
                popsize=15, 
                tol=1e-8, 
                polish=False, # We will polish separately to save time if needed, or use built-in
                init='sobol'
            )
            # Polish the result
            if time.time() - start_time < deadline - 10:
                try:
                    res_polish = minimize(
                        compute_cost, 
                        res.x, 
                        method='L-BFGS-B', 
                        bounds=bounds_log, 
                        options={'maxiter': 500, 'ftol': 1e-15}
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
        except Exception as e:
            pass

    # --- Phase 2: Local Search from Diverse Initial Points ---
    # Generate random starting points and run Nelder-Mead or Powell.
    # This helps if DE gets stuck or misses a specific basin.
    
    # Initial points:
    # 1. The original seed: r_i = 4 * 2^i
    # 2. Uniformly spaced roots
    # 3. Random points
    
    initial_points = []
    
    # Seed 1: Geometric
    r_seed = 4.0 * (2.0 ** np.arange(12))
    initial_points.append(np.log(r_seed))
    
    # Seed 2: Uniform in [0.1, 100]
    r_uniform = np.linspace(0.1, 100.0, 12)
    initial_points.append(np.log(r_uniform))
    
    # Seed 3: Uniform in [1, 1000]
    r_uniform2 = np.linspace(1.0, 1000.0, 12)
    initial_points.append(np.log(r_uniform2))
    
    # Random points
    rng = np.random.RandomState(123)
    for _ in range(20):
        # Log-uniform distribution over [0.01, 10000]
        log_r = rng.uniform(np.log(0.01), np.log(10000.0), 12)
        initial_points.append(log_r)

    for i, init_x in enumerate(initial_points):
        if time.time() - start_time > deadline - 20:
            break
        
        # Choose optimizer: Nelder-Mead is robust for non-smooth, L-BFGS-B for smooth.
        # Let's try L-BFGS-B first, then Nelder-Mead if it fails or doesn't improve.
        
        try:
            # L-BFGS-B
            res_lf = minimize(
                compute_cost, 
                init_x, 
                method='L-BFGS-B', 
                bounds=bounds_log, 
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8}
            )
            update_best(res_lf.fun, res_lf.x)
            
            # If L-BFGS-B converged to a local minimum, try Nelder-Mead around it for fine-tuning
            if time.time() - start_time < deadline - 5:
                res_nm = minimize(
                    compute_cost, 
                    res_lf.x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                update_best(res_nm.fun, res_nm.x)
                
        except Exception as e:
            pass

    # --- Phase 3: Dual Annealing (Global) ---
    # Dual Annealing is a powerful global optimizer that can be used as a final check.
    if time.time() - start_time < deadline - 30:
        try:
            # Dual annealing takes a function and bounds.
            # It requires the function to take a single array argument.
            res_da = dual_annealing(
                compute_cost, 
                bounds=bounds_log, 
                seed=101, 
                maxiter=100, 
                restarts=10, 
                verbose=0
            )
            # Polish
            if time.time() - start_time < deadline - 10:
                res_da_polish = minimize(
                    compute_cost, 
                    res_da.x, 
                    bounds=bounds_log, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-15}
                )
                update_best(res_da_polish.fun, res_da_polish.x)
            else:
                update_best(res_da.fun, res_da.x)
        except Exception as e:
            pass

    # --- Final Fallback ---
    if best_log_roots is None:
        # If everything failed, return the original seed
        best_log_roots = np.log(4.0 * (2.0 ** np.arange(12)))
        best_val = compute_cost(best_log_roots)

    # Convert back to roots
    best_roots = np.exp(best_log_roots)
    
    # Ensure positivity and sorting (though order doesn't matter for product)
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-10)
    
    return list(best_roots)

# Main entry point
if __name__ == "__main__":
    # For local testing
    roots = construct()
    print(roots)

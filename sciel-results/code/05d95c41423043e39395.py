import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit

    m = 12
    n = 1

    # Precomputation: Transformation matrix from standard basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k,j] = (-1)^j * C(k+1, j+1) / j!
    
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    weights = np.arange(13) + 1

    def compute_cost(z):
        """
        z: array of 6 log-variables.
        Roots are r_i = exp(2 * z_i).
        Polynomial P(x) = prod_{i=1}^6 (x - r_i)^2.
        """
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e20

        # Calculate roots
        # z is 6D. r_i = exp(2*z_i).
        # We need 12 roots for np.poly.
        # r = [r_1, r_1, r_2, r_2, ...]
        r_half = np.exp(2.0 * z)
        # Ensure validity
        if np.any(r_half <= 0) or np.any(np.isnan(r_half)) or np.any(np.isinf(r_half)):
            return 1e20
        
        # Construct the 12 roots array
        roots = np.repeat(r_half, 2)
        
        # Build polynomial coefficients (descending order)
        # np.poly(roots) returns [1, -sum, ...]
        try:
            coeffs_desc = np.poly(roots)
        except Exception:
            return 1e20
            
        p = coeffs_desc[::-1]  # Ascending order
        
        # Compute Laguerre coefficients
        c = A_T_inv @ p
        
        # Compute norm squared
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 1e20
            
        return 1.0 / norm_sq

    best_val = 1e20
    best_z = None

    # Define bounds for z (log of sqrt(roots))
    # roots = exp(2z). If roots in [0.01, 10000], z in [-2.3, 4.6].
    # Let's use [-3.0, 5.0] to be safe.
    bounds_z = [(-3.0, 5.0)] * 6

    def update_best(val, x):
        nonlocal best_val, best_z
        if val < best_val:
            best_val = val
            best_z = x.copy()

    # Phase 1: Global Search with Differential Evolution
    # 6D space is small enough for DE to be very effective.
    de_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for seed in de_seeds:
        if time.time() - start_time > deadline - 60:
            break
        try:
            res = differential_evolution(
                compute_cost, 
                bounds_z, 
                seed=seed, 
                maxiter=200, 
                popsize=20, 
                tol=1e-10, 
                polish=False,
                init='sobol'
            )
            # Polish the result
            if time.time() - start_time < deadline - 10:
                try:
                    res_polish = minimize(
                        compute_cost, 
                        res.x, 
                        method='L-BFGS-B', 
                        bounds=bounds_z, 
                        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
                    )
                    val = res_polish.fun
                    x = res_polish.x
                except Exception:
                    val = res.fun
                    x = res.x
            else:
                val = res.fun
                x = res.x
            
            update_best(val, x)
        except Exception as e:
            pass

    # Phase 2: Local Search from Diverse Initial Points
    # Generate some structured starting points
    initial_points = []
    
    # 1. Geometric progression in roots
    # r_i = 4 * 2^i for i=0..5? No, 6 unique roots.
    # Let's try r_i = 2 * 2^i = 2^(i+1)
    r_geo = 2.0 * (2.0 ** np.arange(6))
    z_geo = 0.5 * np.log(r_geo)
    initial_points.append(z_geo)
    
    # 2. Uniformly spaced in log domain
    z_uni = np.linspace(-2.0, 4.0, 6)
    initial_points.append(z_uni)
    
    # 3. Random points
    rng = np.random.RandomState(42)
    for _ in range(20):
        z_rand = rng.uniform(-3.0, 5.0, 6)
        initial_points.append(z_rand)
        
    # Also add the best found so far if any
    if best_z is not None:
        initial_points.append(best_z)

    for init_x in initial_points:
        if time.time() - start_time > deadline - 20:
            break
        
        try:
            # L-BFGS-B
            res_lf = minimize(
                compute_cost, 
                init_x, 
                method='L-BFGS-B', 
                bounds=bounds_z, 
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8}
            )
            update_best(res_lf.fun, res_lf.x)
            
            # Nelder-Mead for fine-tuning
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

    # Final Fallback
    if best_z is None:
        # Fallback to geometric seed
        r_geo = 2.0 * (2.0 ** np.arange(6))
        best_z = 0.5 * np.log(r_geo)
        best_val = compute_cost(best_z)

    # Convert back to roots
    # z -> r_i = exp(2 * z_i)
    unique_roots = np.exp(2.0 * best_z)
    
    # The task asks for "list of 12 positive double-root locations".
    # This implies the list should contain the roots with multiplicity.
    # i.e., [r1, r1, r2, r2, ...]
    # However, "double-root locations" might imply just the locations.
    # But the interface says "list of 12 ... locations".
    # So we return the 12 values.
    
    roots_12 = np.repeat(unique_roots, 2)
    
    # Ensure positivity and sorting
    roots_12 = np.sort(roots_12)
    roots_12 = np.maximum(roots_12, 1e-10)
    
    return list(roots_12)

# Main entry point
if __name__ == "__main__":
    # For local testing
    roots = construct()
    print(roots)

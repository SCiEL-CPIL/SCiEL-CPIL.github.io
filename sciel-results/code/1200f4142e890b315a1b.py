import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    m = 12
    alpha = 1

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k + 1, j + 1) / math.factorial(j)

    # Norm vector d_k = (k+1)! for weight e^{-x} on [0, inf)
    # <x^k, x^l> = Gamma(k+l+1)
    # K[i,j] = Gamma(i+j+1)
    K = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            K[i, j] = math.factorial(i + j + 1)

    def compute_norm_sq(roots):
        """
        Compute the squared L2 norm with weight e^{-x} of P(x) = prod(x - r_i).
        """
        # Build polynomial coefficients in ascending order via iterative multiplication
        p = np.array([1.0])  # ascending: p[0] = constant term
        for r in roots:
            new_p = np.zeros(len(p) + 1)
            new_p[0] = -r * p[0]
            for j in range(1, len(p)):
                new_p[j] = p[j-1] - r * p[j]
            new_p[len(p)] = p[len(p)-1]
            p = new_p

        # Compute norm squared: norm_sq = p @ K @ p
        norm_sq = float(p @ K @ p)

        if norm_sq < 0:  # Numerical error check
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Objective: 1.0 / norm_sq
        """
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10

        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Generate diverse seeds.
    # 2. Local polish each seed with L-BFGS-B.
    # 3. If time remains, run a lightweight DE to find a better basin, then polish.
    
    bounds = [(-4.0, 10.0)] * 12
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Seed 1: Geometric sequence a=4, q=2
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Seed 2: Roots of Laguerre Polynomial L_12^(1)
    # L_n^(alpha)(x) = \sum_{k=0}^n (-1)^k \binom{n+alpha}{n-k} \frac{x^k}{k!}
    lag_coeffs_desc = np.zeros(m + 1)
    for k in range(m + 1):
        lag_coeffs_desc[m - k] = ((-1)**k) * math.comb(m + alpha, m - k) / math.factorial(k)
        
    try:
        lag_roots = np.roots(lag_coeffs_desc)
        lag_roots_real = lag_roots[np.abs(lag_roots.imag) < 1e-9].real
        lag_roots_pos = lag_roots_real[lag_roots_real > 0]
        if len(lag_roots_pos) == m:
            seed_log_roots_lag = np.log(lag_roots_pos)
        else:
            seed_log_roots_lag = seed_log_roots
    except Exception:
        seed_log_roots_lag = seed_log_roots

    # Seed 3: Uniform in log space [0.1, 100]
    seed_log_roots_uni = np.linspace(np.log(0.1), np.log(100), m)
    
    # Seed 4: Clustered near 1
    seed_log_roots_clust = np.log(1.0 + 0.1 * np.arange(m))

    seeds = [seed_log_roots, seed_log_roots_lag, seed_log_roots_uni, seed_log_roots_clust]
    
    # Add some random seeds
    rng = np.random.default_rng(42)
    for _ in range(10):
        seeds.append(rng.uniform(-2, 5, size=m))

    # Local polish from each seed
    for seed_log in seeds:
        if time.time() > deadline - 100:
            break
        try:
            res = minimize(
                cost_func, seed_log, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_roots = np.exp(res.x)
                best_log_roots = res.x
        except Exception:
            pass

    # Global Search with Differential Evolution (Lightweight)
    # Only run if we have plenty of time left
    if time.time() < deadline - 200:
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=123, 
                maxiter=100, 
                popsize=15, 
                tol=1e-10, 
                polish=True, 
                init='sobol',
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            if result.fun < best_val:
                best_val = result.fun
                best_roots = np.exp(result.x)
                best_log_roots = result.x
        except Exception:
            pass

    # Final high-precision polish pass
    if best_log_roots is not None and time.time() < deadline - 15:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 200}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_roots = np.exp(polish_result.x)
                best_log_roots = polish_result.x
        except Exception:
            pass
        
        # Try Powell for potential escape from local minima (derivative-free)
        if time.time() < deadline - 15:
            try:
                pw_result = minimize(
                    cost_func, best_log_roots, method='Powell',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'xtol': 1e-12}
                )
                if pw_result.fun < best_val:
                    best_val = pw_result.fun
                    best_roots = np.exp(pw_result.x)
                    best_log_roots = pw_result.x
            except Exception:
                pass

    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]

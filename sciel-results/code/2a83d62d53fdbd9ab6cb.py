"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Deadline guard: 210s limit, use 195s for safety
_DEADLINE = time.time() + 195

# Precompute Hermite polynomial values on a grid
_N_GRID = 10001
_X_GRID = np.linspace(-12, 12, _N_GRID)
_DX = _X_GRID[1] - _X_GRID[0]

# H0(x) = 1
_H0 = np.ones_like(_X_GRID)
# H4(x) = 16x^4 - 48x^2 + 12
_H4 = 16.0 * _X_GRID**4 - 48.0 * _X_GRID**2 + 12.0
# H8(x) = 256x^8 - 4608x^6 + 23040x^4 - 34560x^2 + 10080
_H8 = (256.0 * _X_GRID**8 - 4608.0 * _X_GRID**6 + 23040.0 * _X_GRID**4 -
       34560.0 * _X_GRID**2 + 10080.0)
# H12(x) = 4096x^12 - 147456x^10 + 2090112x^8 - 12902400x^6 + 34406400x^4 - 34406400x^2 + 10321920
_H12 = (4096.0 * _X_GRID**12 - 147456.0 * _X_GRID**10 + 2090112.0 * _X_GRID**8 -
        12902400.0 * _X_GRID**6 + 34406400.0 * _X_GRID**4 - 34406400.0 * _X_GRID**2 +
        10321920.0)

# Gaussian weight
_GAUSS = np.exp(-_X_GRID**2 / 2.0)

# Values at zero
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 3024.0
_H12_0 = 10321920.0

def _l1_product(a0, a4, a8):
    """
    Compute A(g) * A(F[g]) where g(x) = P(x) * exp(-x^2/2).
    
    Since FT[H_n(x) e^{-x^2/2}](xi) = (-i)^n H_n(xi) e^{-xi^2/2},
    we have F[g](xi) = P(xi) e^{-xi^2/2} (up to phase).
    Therefore A(g) = A(F[g]) = integral |P(x)| e^{-x^2/2} dx.
    The product is [integral |P(x)| e^{-x^2/2} dx]^2.
    
    We minimize this subject to P(0) = 0.
    """
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    # P(x) on the grid
    P = a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12
    g = P * _GAUSS
    
    # L1 norm via trapezoidal rule
    abs_g = np.abs(g)
    A_g = np.trapezoid(abs_g, _X_GRID) if hasattr(np, 'trapezoid') else np.trapz(abs_g, _X_GRID)
    
    if not np.isfinite(A_g) or A_g < 0:
        return 1e10
    
    # Product is A_g * A_g = A_g^2 since A(F[g]) = A(g)
    return A_g * A_g

def _l1_product_smooth(a0, a4, a8, eps=1e-6):
    """
    Smooth surrogate for |P(x)| using sqrt(P(x)^2 + eps^2) - eps.
    This is C^1 and helps gradient-based methods.
    """
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    P = a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12
    g = P * _GAUSS
    
    # Smooth absolute value
    abs_g_smooth = np.sqrt(g**2 + eps**2) - eps
    A_g = np.trapezoid(abs_g_smooth, _X_GRID) if hasattr(np, 'trapezoid') else np.trapz(abs_g_smooth, _X_GRID)
    
    if not np.isfinite(A_g) or A_g < 0:
        return 1e10
    
    return A_g * A_g

def construct():
    """
    Optimize coefficients a0, a4, a8 to minimize A(g)*A(F[g]).
    Returns the three coefficients as a numpy array.
    """
    # Seed values from the initial program
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_result = seed_result.copy()
    best_bound = _l1_product(*seed_result)
    
    # Phase 1: Differential evolution for global search
    if time.time() < _DEADLINE - 80:
        try:
            bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.01, 0.01)]
            de_result = differential_evolution(
                _l1_product,
                bounds,
                seed=42,
                maxiter=300,
                tol=1e-12,
                polish=True,
                init='sobol',
                popsize=20
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead from multiple starts
    if time.time() < _DEADLINE - 30:
        starts = [
            best_result,
            seed_result,
            np.array([-1.0, 0.05, 0.0005]),
            np.array([-1.5, 0.06, 0.0006]),
            np.array([-0.8, 0.03, 0.0003]),
            np.array([-2.0, 0.08, 0.0008]),
        ]
        for start in starts:
            if time.time() > _DEADLINE - 25:
                break
            try:
                res = minimize(
                    _l1_product,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                if res.fun < best_bound:
                    best_bound = res.fun
                    best_result = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: High-precision polish with smooth surrogate
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(
                _l1_product_smooth,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': 50000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            # Verify with the true objective
            true_val = _l1_product(*res.x)
            if true_val < best_bound:
                best_bound = true_val
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Final verification
    final_val = _l1_product(*best_result)
    if not np.isfinite(final_val):
        return seed_result
    
    return best_result
# EVOLVE-BLOCK-END

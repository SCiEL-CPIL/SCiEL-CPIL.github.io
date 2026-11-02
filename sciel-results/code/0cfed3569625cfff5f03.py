"""Rewritten program: optimize Fourier-invariant Hermite-Gaussian test function."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _a12_from(a0, a4, a8):
    """Compute a12 so that P(0) = 0."""
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=2032128
    return -(a0 + 12.0*a4 + 1680.0*a8) / 2032128.0


def _poly_coeffs(a0, a4, a8):
    """
    Return coefficient vector for P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12,
    as a numpy array of length 13 (index = power of x).
    """
    a12 = _a12_from(a0, a4, a8)
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12 = 4096x^12 - 73728x^10 + 580608x^8 - 2419200x^6 + 5443200x^4 - 5443200x^2 + 2032128
    c = np.zeros(13)
    c[0] = a0 * 1.0 + a4 * 12.0 + a8 * 1680.0 + a12 * 2032128.0
    c[2] = a4 * (-48.0) + a8 * (-13440.0) + a12 * (-5443200.0)
    c[4] = a4 * 16.0 + a8 * 11520.0 + a12 * 5443200.0
    c[6] = a8 * (-3072.0) + a12 * (-2419200.0)
    c[8] = a8 * 256.0 + a12 * 580608.0
    c[10] = a12 * (-73728.0)
    c[12] = a12 * 4096.0
    return c


def _eval_bound(a0, a4, a8, x_grid):
    """
    Compute the upper bound on C4 = A(f)*A(F).
    f(x) = P(x) * exp(-x^2/2), with P(0)=0.
    The Fourier transform of f is F(k) = P(k) * exp(-k^2/2) (since all n are even).
    A(f) = sup_x |f(x)| / ||f||_2, A(F) = sup_k |F(k)| / ||F||_2.
    Since f and F have the same form, A(f) = A(F), so A(f)*A(F) = (sup|f|/||f||_2)^2.
    """
    c = _poly_coeffs(a0, a4, a8)
    # Evaluate P(x) on grid
    P = np.zeros_like(x_grid)
    for power in range(13):
        if c[power] != 0.0:
            P += c[power] * (x_grid ** power)
    f = P * np.exp(-0.5 * x_grid**2)
    # L2 norm via trapezoidal rule
    norm_sq = np.trapz(f**2, x_grid)
    if norm_sq < 1e-300:
        return 1e300
    norm = np.sqrt(norm_sq)
    sup_abs = np.max(np.abs(f))
    # A(f) * A(F) = (sup|f|/norm)^2 since A(f)=A(F)
    return (sup_abs / norm)**2


def _objective(params, x_grid):
    """Objective function for optimizer."""
    a0, a4, a8 = params
    return _eval_bound(a0, a4, a8, x_grid)


def construct():
    start_time = time.time()
    deadline = 180.0
    safety_margin = 20.0  # seconds
    
    # Use a fine grid for accurate computation
    x_grid = np.linspace(-8, 8, 2001)
    
    # Seed values
    x0 = np.array([-1.13, 0.04, 0.00030864])
    
    best_params = x0.copy()
    best_val = _eval_bound(x0[0], x0[1], x0[2], x_grid)
    
    # Phase 1: Differential evolution for global search
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin - 60:
        try:
            bounds = [(-10, 10), (-10, 10), (-1, 1)]
            de_result = differential_evolution(
                lambda p: _eval_bound(p[0], p[1], p[2], x_grid),
                bounds=bounds,
                seed=42,
                maxiter=200,
                tol=1e-10,
                popsize=15,
                polish=False,
                update_interval=1,
                callback=None
            )
            elapsed = time.time() - start_time
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin:
        try:
            nm_result = minimize(
                lambda p: _eval_bound(p[0], p[1], p[2], x_grid),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            elapsed = time.time() - start_time
            if nm_result.fun < best_val:
                best_val = nm_result.fun
                best_params = nm_result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Further refinement with BFGS (gradient-free since we don't have analytic grad)
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin:
        try:
            # Try multiple local starts around best
            np.random.seed(123)
            for _ in range(5):
                elapsed = time.time() - start_time
                if elapsed > deadline - safety_margin:
                    break
                perturbation = np.random.randn(3) * 0.01
                start_pt = best_params + perturbation
                res = minimize(
                    lambda p: _eval_bound(p[0], p[1], p[2], x_grid),
                    start_pt,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - safety_margin:
        try:
            # Use a finer grid for final evaluation
            x_fine = np.linspace(-8, 8, 5001)
            fine_val = _eval_bound(best_params[0], best_params[1], best_params[2], x_fine)
            
            # One more local step on fine grid
            final_res = minimize(
                lambda p: _eval_bound(p[0], p[1], p[2], x_fine),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if final_res.fun < fine_val:
                best_params = final_res.x.copy()
        except Exception:
            pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - safety_margin:
        # Return best found
        pass
    
    return best_params

# EVOLVE-BLOCK-END

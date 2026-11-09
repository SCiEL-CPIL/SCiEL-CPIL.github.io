"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.special as sp
import scipy.optimize as opt
import scipy.integrate as integrate
import time

# Pre-compute Hermite polynomials
# Physicists' Hermite polynomials
H0 = sp.hermite(0)
H4 = sp.hermite(4)
H8 = sp.hermite(8)
H12 = sp.hermite(12)

# Values at 0 for constraint P(0)=0
# H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
H0_0 = H0(0)
H4_0 = H4(0)
H8_0 = H8(0)
H12_0 = H12(0)

def construct():
    start_time = time.time()
    deadline = 190.0  # Safety margin before 210s kill

    def calculate_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    def integrand_for_quad(x, c0, c4, c8, c12):
        P = c0 * H0(x) + c4 * H4(x) + c8 * H8(x) + c12 * H12(x)
        return np.abs(P) * np.exp(-x**2 / 4.0)

    def objective(coeffs):
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = coeffs
        c12 = calculate_c12(c0, c4, c8)
        
        # Use adaptive quadrature for better accuracy
        result, error = integrate.quad(
            lambda x: integrand_for_quad(x, c0, c4, c8, c12),
            -15, 15, limit=200, epsabs=1e-10, epsrel=1e-10
        )
        
        return result

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 1/3240.0])
    
    best_result = None
    best_val = np.inf
    
    # First, try local optimization from seed
    try:
        res = opt.minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x.copy()
    except Exception:
        pass

    # Try differential evolution for global search
    if time.time() < deadline - 30:
        try:
            bounds = [(-5, 5), (-1, 1), (-0.01, 0.01)]
            de_res = opt.differential_evolution(
                objective, bounds, seed=42, maxiter=50, popsize=15,
                tol=1e-8, mutation=(0.5, 1.5), recombination=0.7
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_result = de_res.x.copy()
        except Exception:
            pass

    # Refine with Nelder-Mead from best found so far
    if best_result is not None and time.time() < deadline - 15:
        try:
            res2 = opt.minimize(objective, best_result, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x.copy()
        except Exception:
            pass

    # Try a few more diverse starting points
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        # Try points near the seed and some random ones
        candidates = [
            x0 + rng.normal(0, 0.1, 3),
            x0 + rng.normal(0, 0.5, 3),
            np.array([-1.0, 0.05, 0.0002]),
            np.array([-1.2, 0.03, 0.0004]),
            np.array([-1.1, 0.045, 0.00028]),
        ]
        for x_start in candidates:
            if time.time() > deadline:
                break
            try:
                res_r = opt.minimize(objective, x_start, method='Nelder-Mead',
                                     options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11})
                if res_r.fun < best_val:
                    best_val = res_r.fun
                    best_result = res_r.x.copy()
            except Exception:
                pass

    if best_result is None:
        return x0
    return best_result
# EVOLVE-BLOCK-END

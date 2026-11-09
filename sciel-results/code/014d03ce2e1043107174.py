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

# Pre-compute Gauss-Hermite quadrature nodes and weights
# For integral of f(x)*exp(-x^2) dx, use change of variables
# Our integral is int |P(x)| * exp(-x^2/4) dx
# Let t = x/sqrt(2), then x = t*sqrt(2), dx = sqrt(2) dt
# Integral = sqrt(2) * int |P(t*sqrt(2))| * exp(-t^2) dt
# Gauss-Hermite quadrature: int f(t)*exp(-t^2) dt ≈ sum w_i * f(t_i)
_N_GH = 100  # Number of Gauss-Hermite nodes
_gh_nodes, _gh_weights = np.polynomial.hermite.hermgauss(_N_GH)
_gh_nodes = _gh_nodes * np.sqrt(2)  # Scale to match exp(-x^2/4)
_gh_weights = _gh_weights * np.sqrt(2)  # Scale weights

# Pre-compute P values at quadrature nodes for each Hermite polynomial
_P0_gh = H0(_gh_nodes)
_P4_gh = H4(_gh_nodes)
_P8_gh = H8(_gh_nodes)
_P12_gh = H12(_gh_nodes)

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill

    def calculate_c12(c0, c4, c8):
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    def objective(coeffs):
        if time.time() > deadline:
            return np.inf
        c0, c4, c8 = coeffs
        c12 = calculate_c12(c0, c4, c8)
        
        # Fast evaluation using pre-computed Gauss-Hermite quadrature
        P_gh = c0 * _P0_gh + c4 * _P4_gh + c8 * _P8_gh + c12 * _P12_gh
        result = np.sum(_gh_weights * np.abs(P_gh))
        
        return result

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 1/3240.0])
    
    best_result = None
    best_val = np.inf
    
    # First, try local optimization from seed with BFGS for gradient-based refinement
    try:
        res = opt.minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x.copy()
    except Exception:
        pass

    # Try differential evolution for global search
    if time.time() < deadline - 60:
        try:
            bounds = [(-5, 5), (-2, 2), (-0.01, 0.01)]
            de_res = opt.differential_evolution(
                objective, bounds, seed=42, maxiter=100, popsize=20,
                tol=1e-10, mutation=(0.5, 1.5), recombination=0.7,
                polish=True, init='latinhypercube'
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_result = de_res.x.copy()
        except Exception:
            pass

    # Try multiple local optimizations from diverse starting points
    if best_result is not None and time.time() < deadline - 30:
        try:
            res2 = opt.minimize(objective, best_result, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x.copy()
        except Exception:
            pass

    # Try a few more diverse starting points
    if time.time() < deadline - 20:
        rng = np.random.default_rng(123)
        # Try points near the seed and some random ones
        candidates = [
            x0 + rng.normal(0, 0.1, 3),
            x0 + rng.normal(0, 0.5, 3),
            x0 + rng.normal(0, 1.0, 3),
            np.array([-1.0, 0.05, 0.0002]),
            np.array([-1.2, 0.03, 0.0004]),
            np.array([-1.1, 0.045, 0.00028]),
            np.array([-1.5, 0.08, 0.0005]),
            np.array([-0.8, 0.02, 0.0001]),
            np.array([-2.0, 0.1, 0.001]),
        ]
        for x_start in candidates:
            if time.time() > deadline:
                break
            try:
                res_r = opt.minimize(objective, x_start, method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13})
                if res_r.fun < best_val:
                    best_val = res_r.fun
                    best_result = res_r.x.copy()
            except Exception:
                pass

    # Final high-precision polish
    if best_result is not None and time.time() < deadline - 5:
        try:
            res_final = opt.minimize(objective, best_result, method='Nelder-Mead',
                                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_result = res_final.x.copy()
        except Exception:
            pass

    if best_result is None:
        return x0
    return best_result
# EVOLVE-BLOCK-END

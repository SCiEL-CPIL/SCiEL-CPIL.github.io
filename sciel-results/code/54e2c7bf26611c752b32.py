"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import minimize

_DEADLINE = time.time() + 190  # 20s margin before 210s kill

# Physicist's Hermite polynomials evaluated at 0
_H0_0 = 1.0
_H4_0 = 3.0
_H8_0 = 105.0
_H12_0 = 945.0

# Seed coefficients (Goncalves et al. 2017)
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])


def _eval_poly(x, a0, a4, a8, a12):
    """Evaluate P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 at scalar x."""
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    h0 = 1.0
    h4 = x4 - 6.0 * x2 + 3.0
    h8 = x8 - 28.0 * x6 + 210.0 * x4 - 420.0 * x2 + 105.0
    h12 = x12 - 66.0 * x10 + 1155.0 * x8 - 6930.0 * x6 + 13860.0 * x4 - 7560.0 * x2 + 945.0
    return a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12


def _compute_delta_x_squared(a0, a4, a8):
    """
    Compute (Δx)² for g(x) = P(x)exp(-x²/2) where
    P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 with g(0)=0.
    For even Fourier-invariant functions, uncertainty product = (Δx)².
    """
    if time.time() > _DEADLINE:
        return None

    # Constraint g(0) = 0
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0

    def g_sq(x):
        p = _eval_poly(x, a0, a4, a8, a12)
        return p * p * np.exp(-x * x)

    def x2_g_sq(x):
        return x * x * g_sq(x)

    try:
        norm_sq, _ = quad(g_sq, -np.inf, np.inf, limit=300, epsabs=1e-13, epsrel=1e-11)
        if norm_sq <= 0 or not np.isfinite(norm_sq):
            return None
        x2_norm, _ = quad(x2_g_sq, -np.inf, np.inf, limit=300, epsabs=1e-13, epsrel=1e-11)
        if not np.isfinite(x2_norm):
            return None
        delta_x_sq = x2_norm / norm_sq
        if not np.isfinite(delta_x_sq) or delta_x_sq < 0:
            return None
        return delta_x_sq
    except Exception:
        return None


def _objective(x):
    """Objective function for optimization."""
    if time.time() > _DEADLINE - 5:
        return np.inf
    v = _compute_delta_x_squared(x[0], x[1], x[2])
    if v is None:
        return np.inf
    return v


def construct():
    """Find optimal coefficients to minimize the uncertainty product bound."""
    start_time = time.time()

    # Always evaluate seed first
    seed_val = _compute_delta_x_squared(*_SEED)
    if seed_val is None:
        seed_val = np.inf
    best_val = seed_val
    best_coeffs = _SEED.copy()

    def time_ok(margin=20):
        return (time.time() - start_time) < (190 - margin)

    # Strategy 1: L-BFGS-B with numerical gradient from seed
    if time_ok(60):
        try:
            result = minimize(_objective, _SEED, method='L-BFGS-B',
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            if np.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Strategy 2: Nelder-Mead refinement from best point
    if time_ok(50):
        try:
            result = minimize(_objective, best_coeffs, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                                       'adaptive': True})
            if np.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Strategy 3: Powell method for additional refinement
    if time_ok(40):
        try:
            result = minimize(_objective, best_coeffs, method='Powell',
                              options={'maxiter': 3000, 'xtol': 1e-14, 'ftol': 1e-16})
            if np.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Strategy 4: Final high-precision Nelder-Mead polish
    if time_ok(25):
        try:
            result = minimize(_objective, best_coeffs, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17,
                                       'adaptive': True})
            if np.isfinite(result.fun) and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Strategy 5: Multi-start from diverse initializations (if time remains)
    if time_ok(30):
        rng = np.random.default_rng(42)
        for i in range(20):
            if not time_ok(15):
                break
            # Diverse starting points
            if i == 0:
                coeffs = np.array([0.0, 0.0, 0.0])
            elif i == 1:
                coeffs = np.array([-1.0, 0.1, 0.0])
            elif i == 2:
                coeffs = np.array([-0.5, 0.05, 0.001])
            else:
                coeffs = _SEED + rng.normal(0, 0.1, 3)
            val = _compute_delta_x_squared(*coeffs)
            if val is not None and val < best_val:
                best_val = val
                best_coeffs = coeffs.copy()
                # Quick local refinement
                if time_ok(10):
                    try:
                        res = minimize(_objective, best_coeffs, method='Nelder-Mead',
                                       options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                                                'adaptive': True})
                        if np.isfinite(res.fun) and res.fun < best_val:
                            best_val = res.fun
                            best_coeffs = res.x.copy()
                    except Exception:
                        pass

    # Ensure we never return worse than seed
    if best_val > seed_val:
        best_coeffs = _SEED.copy()
        best_val = seed_val

    return best_coeffs
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time


def construct():
    start_time = time.time()
    deadline = 175.0  # Safety margin before 210s limit

    # Hermite polynomials (physicists' convention)
    def H0(x):
        return np.ones_like(x, dtype=float)

    def H4(x):
        return x**4 - 6*x**2 + 3

    def H8(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105

    def H12(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395

    def poly(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)

    def poly_deriv(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        dH0 = np.zeros_like(x, dtype=float)
        dH4 = 4*x**3 - 12*x
        dH8 = 8*x**7 - 168*x**5 + 840*x**3 - 840*x
        dH12 = 12*x**11 - 660*x**9 + 10080*x**7 - 55440*x**5 + 92400*x**3 - 55440*x
        return c0*dH0 + c4*dH4 + c8*dH8 + c12*dH12

    def compute_uncertainty_product(coeffs):
        c0, c4, c8 = coeffs

        # Check time
        if time.time() > deadline - 5:
            return np.inf

        # Compute integrals using quad
        try:
            I0, _ = integrate.quad(lambda x: poly(x, c0, c4, c8)**2 * np.exp(-x**2), -np.inf, np.inf, limit=200)
            if I0 <= 0 or not np.isfinite(I0):
                return np.inf

            Ix2, _ = integrate.quad(lambda x: x**2 * poly(x, c0, c4, c8)**2 * np.exp(-x**2), -np.inf, np.inf, limit=200)
            if not np.isfinite(Ix2) or Ix2 < 0:
                return np.inf

            Ideriv, _ = integrate.quad(lambda x: (poly_deriv(x, c0, c4, c8) - x*poly(x, c0, c4, c8))**2 * np.exp(-x**2), -np.inf, np.inf, limit=200)
            if not np.isfinite(Ideriv) or Ideriv < 0:
                return np.inf

            # A(f)^2 = Ix2 / I0
            # A(gf)^2 = (1/(4*pi^2)) * Ideriv / I0
            # Product = sqrt(A_f_sq * A_gf_sq)

            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * np.pi**2)) * Ideriv / I0

            return np.sqrt(A_f_sq * A_gf_sq)
        except Exception:
            return np.inf

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    best_score = compute_uncertainty_product(x0)
    best_coeffs = x0.copy()

    # Nelder-Mead optimization with tighter tolerances
    try:
        res = opt.minimize(compute_uncertainty_product, x0, method='Nelder-Mead',
                          options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000, 'adaptive': True})
        if res.fun < best_score and np.isfinite(res.fun):
            best_score = res.fun
            best_coeffs = res.x.copy()
    except Exception:
        pass

    # Try L-BFGS-B with gradient estimation via finite differences
    try:
        def objective(x):
            return compute_uncertainty_product(x)

        # Use COBYLA or SLSQP if available
        res = opt.minimize(objective, best_coeffs, method='COBYLA',
                          options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
        if res.fun < best_score and np.isfinite(res.fun):
            best_score = res.fun
            best_coeffs = res.x.copy()
    except Exception:
        pass

    # Random restarts with decreasing noise scale
    np.random.seed(42)
    for i in range(30):
        if time.time() > deadline - 15:
            break
        noise_scale = 0.05 / (1 + i * 0.15)
        noise = np.random.normal(0, noise_scale, 3)
        x_try = best_coeffs + noise
        try:
            res = opt.minimize(compute_uncertainty_product, x_try, method='Nelder-Mead',
                              options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 800, 'adaptive': True})
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Final high-precision polish pass
    if time.time() < deadline - 20:
        try:
            res = opt.minimize(compute_uncertainty_product, best_coeffs, method='Nelder-Mead',
                              options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 3000, 'adaptive': True})
            if res.fun < best_score and np.isfinite(res.fun):
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END

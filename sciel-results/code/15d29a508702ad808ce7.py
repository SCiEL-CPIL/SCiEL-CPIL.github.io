import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill

    # Hermite polynomial values at quadrature points (precomputed)
    # Use Gauss-Hermite quadrature for efficient L1 norm computation
    # For even functions: integral_0^inf f(x) dx = 0.5 * integral_-inf^inf f(x) dx
    # Gauss-Hermite: integral_-inf^inf e^{-x^2} f(x) dx ~ sum w_i f(x_i)
    # We need integral_0^inf |e^{-x^2/2} P(x)| dx
    # Let y = x/sqrt(2), x = y*sqrt(2), dx = sqrt(2) dy
    # integral_0^inf |e^{-x^2/2} P(x)| dx = sqrt(2) * integral_0^inf |e^{-y^2} P(y*sqrt(2))| dy
    # = sqrt(2)/2 * integral_-inf^inf |e^{-y^2} P(y*sqrt(2))| dy  (even function)
    # Gauss-Hermite with n=200 points
    n_quad = 200
    y_nodes, y_weights = np.polynomial.hermite.hermgauss(n_quad)
    x_nodes = y_nodes * np.sqrt(2.0)
    x_weights = y_weights * np.sqrt(2.0) * 0.5  # factor for 0 to inf and even

    # Precompute Hermite polynomial values at x_nodes
    x2 = x_nodes ** 2
    x4 = x2 ** 2
    x6 = x4 * x2
    x8 = x4 ** 2
    x10 = x8 * x2
    x12 = x8 * x4

    H0_vals = np.ones_like(x_nodes)
    H4_vals = 16.0 * x4 - 48.0 * x2 + 12.0
    H8_vals = 256.0 * x8 - 4480.0 * x6 + 26880.0 * x4 - 48384.0 * x2 + 1680.0
    H12_vals = (4096.0 * x12 - 129024.0 * x10 + 1548288.0 * x8 - 7654656.0 * x6 +
                16636928.0 * x4 - 15974400.0 * x2 + 665280.0)

    # Precompute weighted values for L1 norm
    w_H0 = x_weights * H0_vals
    w_H4 = x_weights * H4_vals
    w_H8 = x_weights * H8_vals
    w_H12 = x_weights * H12_vals

    # L2 norm coefficients
    sqrt_pi = np.sqrt(np.pi)
    l2_coeff = np.array([1.0, 384.0, 10321920.0, 1962000550400.0])

    def coeffs_from_log(log_c):
        """Convert log-magnitude + sign parameters to actual coefficients."""
        # log_c has shape (6,): [log|c0|, sign0, log|c4|, sign4, log|c8|, sign8]
        # Actually simpler: use [log|c0|, log|c4|, log|c8|] and derive signs from constraint
        # Better: parameterize as [log|c0|, log|c4|, log|c8|] with c12 from constraint
        log_c0, log_c4, log_c8 = log_c
        c0 = math.exp(log_c0)
        c4 = math.exp(log_c4)
        c8 = math.exp(log_c8)
        # c12 from constraint: c0 + 12*c4 + 1680*c8 + 665280*c12 = 0
        c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
        return c0, c4, c8, c12

    def objective_log(log_c):
        """Objective in log-magnitude space. Returns (bound, coeffs)."""
        c0, c4, c8, c12 = coeffs_from_log(log_c)

        # L1 norm via Gauss-Hermite quadrature
        poly_vals = c0 * H0_vals + c4 * H4_vals + c8 * H8_vals + c12 * H12_vals
        l1_g = np.sum(x_weights * np.abs(poly_vals))

        # L2 norm squared
        norm_sq = sqrt_pi * (
            c0**2 * l2_coeff[0] +
            c4**2 * l2_coeff[1] +
            c8**2 * l2_coeff[2] +
            c12**2 * l2_coeff[3]
        )

        if norm_sq < 1e-15:
            return np.inf, (c0, c4, c8)

        bound = (l1_g * l1_g) / norm_sq
        return bound, (c0, c4, c8)

    def objective_scalar(log_c):
        """Scalar objective for optimizers."""
        bound, _ = objective_log(log_c)
        return bound

    # Initial guess in log-space
    c0_init = -1.13
    c4_init = 0.04
    c8_init = 1.0 / 3240.0
    log_init = np.array([
        math.log(abs(c0_init)),
        math.log(abs(c4_init)),
        math.log(abs(c8_init))
    ])

    best_c = np.array([c0_init, c4_init, c8_init])
    best_val, _ = objective_log(log_init)

    # Phase 1: Nelder-Mead in log-space from initial guess
    try:
        res = minimize(objective_scalar, log_init, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_c = np.array(coeffs_from_log(res.x)[:3])
    except Exception:
        pass

    # Phase 2: Differential evolution in log-space for global search
    if time.time() - start_time < deadline - 60:
        try:
            # Bounds in log-space: c can range from ~1e-8 to ~1e2
            # log(1e-8) ~ -18.4, log(1e2) ~ 4.6
            bounds = [(-20.0, 5.0), (-20.0, 5.0), (-20.0, 5.0)]
            pop_size = 40
            de_res = differential_evolution(
                objective_scalar, bounds, seed=42, maxiter=80,
                tol=1e-12, polish=True, popsize=pop_size,
                mutation=(0.3, 1.0), recombination=0.7,
                callback=lambda x, convergence: (time.time() - start_time) < deadline - 20
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_c = np.array(coeffs_from_log(de_res.x)[:3])
        except Exception:
            pass

    # Phase 3: Fine-tune with Nelder-Mead from best found
    if time.time() - start_time < deadline - 15:
        try:
            # Get current best in log-space
            c0_b, c4_b, c8_b = best_c
            if abs(c0_b) < 1e-30:
                c0_b = 1e-10
            if abs(c4_b) < 1e-30:
                c4_b = 1e-10
            if abs(c8_b) < 1e-30:
                c8_b = 1e-10
            log_best = np.array([
                math.log(abs(c0_b)),
                math.log(abs(c4_b)),
                math.log(abs(c8_b))
            ])
            res2 = minimize(objective_scalar, log_best, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = np.array(coeffs_from_log(res2.x)[:3])
        except Exception:
            pass

    # Phase 4: Multi-start Nelder-Mead with random perturbations in log-space
    if time.time() - start_time < deadline - 10:
        rng = np.random.RandomState(123)
        for trial in range(10):
            if time.time() - start_time > deadline - 10:
                break
            # Perturb the best log-coefficients
            perturbation = rng.randn(3) * 0.5
            log_trial = log_best + perturbation
            try:
                res3 = minimize(objective_scalar, log_trial, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                if res3.fun < best_val:
                    best_val = res3.fun
                    best_c = np.array(coeffs_from_log(res3.x)[:3])
            except Exception:
                pass

    return best_c

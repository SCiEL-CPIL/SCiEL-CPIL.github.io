"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _time_left(deadline):
    return max(0.0, deadline - time.time())


def _E_cpl(a, w0, wa):
    """Hubble parameter H(a)/H0 for CPL model in flat universe."""
    z = 1.0 / a - 1.0
    base = 1.0 + w0 + wa * z
    base = np.maximum(base, 1e-10)
    exponent = wa / (1.0 + w0)
    return a**(-3.0) * base**exponent


def _integrand(z, w0, wa):
    """1/E(z) for CPL model."""
    a = 1.0 / (1.0 + z)
    return 1.0 / _E_cpl(a, w0, wa)


def _exact_integral(z, w0, wa):
    """Compute exact integral_0^z dz'/E(z') using adaptive quadrature."""
    from scipy.integrate import quad
    result, _ = quad(lambda zp: _integrand(zp, w0, wa), 0, z, limit=300, epsabs=1e-12, epsrel=1e-12)
    return result


def _quad_integral(z, nodes, weights, w0, wa):
    """Quadrature approximation of integral_0^z dz'/E(z')."""
    z_nodes = z * nodes
    vals = _integrand(z_nodes, w0, wa)
    return z * np.dot(weights, vals)


def _relative_error(z, nodes, weights, w0, wa):
    exact = _exact_integral(z, w0, wa)
    approx = _quad_integral(z, nodes, weights, w0, wa)
    if abs(exact) < 1e-15:
        return 0.0
    return abs((approx - exact) / exact)


def _worst_error(nodes, weights, models, z_vals):
    max_err = 0.0
    for w0, wa in models:
        for z in z_vals:
            err = _relative_error(z, nodes, weights, w0, wa)
            if err > max_err:
                max_err = err
    return max_err


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin

    # Representative CPL models
    models = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (-0.5, -0.5),
        (-1.0, 0.5),
        (-0.8, -0.5),
        (-1.2, 0.3),
        (-0.3, -0.3),
        (-1.0, -0.5),
    ]

    # Test redshifts
    z_test = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])

    # Initial guess: Gauss-Legendre nodes and weights on [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes0 = 0.5 * (gl_nodes + 1.0)
    weights0 = 0.5 * gl_weights
    nodes0 = np.clip(nodes0, 1e-10, 1.0 - 1e-10)
    nodes0 = np.sort(nodes0)

    best_nodes = nodes0.copy()
    best_weights = weights0.copy()
    best_err = _worst_error(best_nodes, best_weights, models, z_test)

    # Strategy: Use a different approach - optimize with scipy's minimize
    # but with a better initial guess and more robust handling
    from scipy.optimize import minimize

    def objective(params):
        if _time_left(deadline) < 5.0:
            return 1e10
        nodes = params[:8]
        log_weights = params[8:]

        # Enforce nodes in (0,1) and strictly increasing
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(np.diff(nodes) <= 0):
            return 1e10

        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)

        return _worst_error(nodes, weights, models, z_test)

    params0 = np.concatenate([nodes0, np.log(weights0)])

    # Try multiple optimization methods and starting points
    starting_points = [
        params0,
        # Perturbed Gauss-Legendre
        params0 + 0.01 * np.random.randn(16),
        # Equally spaced nodes
        np.concatenate([np.linspace(0.05, 0.95, 8), np.log(np.ones(8) / 8)]),
        # Chebyshev nodes
        np.concatenate([0.5 - 0.5 * np.cos(np.pi * (np.arange(8) + 0.5) / 8),
                        np.log(np.ones(8) / 8)]),
    ]

    for sp in starting_points:
        if _time_left(deadline) < 10.0:
            break

        try:
            result = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )

            if result.fun < 1e10:
                nodes_opt = result.x[:8]
                log_weights_opt = result.x[8:]
                weights_opt = np.exp(log_weights_opt)
                weights_opt = weights_opt / np.sum(weights_opt)

                if np.all(nodes_opt > 0) and np.all(nodes_opt < 1) and np.all(np.diff(nodes_opt) > 0):
                    err = result.fun
                    if err < best_err:
                        best_err = err
                        best_nodes = nodes_opt.copy()
                        best_weights = weights_opt.copy()
        except Exception:
            pass

    # Final check
    if np.all(best_nodes > 0) and np.all(best_nodes < 1) and np.all(np.diff(best_nodes) > 0):
        return {"nodes": best_nodes, "weights": best_weights}

    # Fallback: return Gauss-Legendre
    return {"nodes": nodes0, "weights": weights0}
# EVOLVE-BLOCK-END

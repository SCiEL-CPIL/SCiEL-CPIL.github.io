<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 85.0
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE


def _cpl_E(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model, flat geometry."""
    zp1 = 1.0 + z
    exp1 = 3.0 * (1.0 + w0 + wa)
    return np.sqrt(np.power(zp1, exp1) * np.power(1.0 + wa * z, -3.0 * wa))


# Pre-compute Gauss-Legendre nodes for exact integration
_GX64, _GW64 = np.polynomial.legendre.leggauss(64)


def _exact_comoving_fast(z, w0, wa):
    """Exact comoving distance using pre-computed 64-point Gauss-Legendre."""
    if z <= 0:
        return 0.0
    z_vals = 0.5 * z * (_GX64 + 1.0)
    E_vals = _cpl_E(z_vals, w0, wa)
    return 0.5 * z * np.sum(_GW64 / E_vals)


def _quad_comoving(z, w0, wa, nodes, weights):
    """Quadrature comoving distance."""
    if z <= 0:
        return 0.0
    z_vals = z * nodes
    E_vals = _cpl_E(z_vals, w0, wa)
    return z * np.sum(weights / E_vals)


def _worst_error_smooth(nodes, weights, w0_vals, wa_vals, z_vals, beta=50.0):
    """Compute smooth max (log-sum-exp) of relative errors."""
    errors = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                exact = _exact_comoving_fast(z, w0, wa)
                quad = _quad_comoving(z, w0, wa, nodes, weights)
                rel_err = abs(quad - exact) / max(abs(exact), 1e-30)
                errors.append(rel_err)
    errors = np.array(errors)
    # Log-sum-exp for smooth max
    m = np.max(errors)
    if m < 1e-15:
        return m
    log_sum_exp = m + np.log(np.sum(np.exp(beta * (errors - m)))) / beta
    return log_sum_exp


def _worst_error_exact(nodes, weights, w0_vals, wa_vals, z_vals):
    """Compute exact worst relative error."""
    worst = 0.0
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                exact = _exact_comoving_fast(z, w0, wa)
                quad = _quad_comoving(z, w0, wa, nodes, weights)
                rel_err = abs(quad - exact) / max(abs(exact), 1e-30)
                worst = max(worst, rel_err)
    return worst
=======
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 85.0
_T0 = time.time()


def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE


def _cpl_E(z, w0, wa):
    """E(z) = H(z)/H0 for CPL model, flat geometry."""
    zp1 = 1.0 + z
    exp1 = 3.0 * (1.0 + w0 + wa)
    return np.sqrt(np.power(zp1, exp1) * np.power(1.0 + wa * z, -3.0 * wa))


# Pre-compute Gauss-Legendre nodes for exact integration
_GX64, _GW64 = np.polynomial.legendre.leggauss(64)

# Evaluation grid
_W0_GRID = np.array([-1.2, -1.0, -0.8, -0.5, 0.0, 0.3, 0.5])
_WA_GRID = np.array([-0.5, -0.3, 0.0, 0.3, 0.5, 0.8])
_Z_GRID = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])

_W0_M, _WA_M, _Z_M = np.meshgrid(_W0_GRID, _WA_GRID, _Z_GRID, indexing='ij')
_W0_FLAT = _W0_M.ravel()
_WA_FLAT = _WA_M.ravel()
_Z_FLAT = _Z_M.ravel()
_N_EVALS = len(_Z_FLAT)

# Pre-compute exact comoving distances for the full grid
_EXACT_COMOVING = np.empty(_N_EVALS)
for _i in range(_N_EVALS):
    _z = _Z_FLAT[_i]
    _w0 = _W0_FLAT[_i]
    _wa = _WA_FLAT[_i]
    if _z <= 0:
        _EXACT_COMOVING[_i] = 0.0
    else:
        _z_vals = 0.5 * _z * (_GX64 + 1.0)
        _E_vals = _cpl_E(_z_vals, _w0, _wa)
        _EXACT_COMOVING[_i] = 0.5 * _z * np.sum(_GW64 / _E_vals)


def _quad_comoving_vec(z, w0, wa, nodes, weights):
    """Vectorized quadrature comoving distance over grid."""
    # z: shape (N,), w0: shape (N,), wa: shape (N,), nodes: shape (8,), weights: shape (8,)
    # Compute E(z * u_i) for all i and all grid points
    # z_vals shape: (N, 8)
    z_vals = z[:, None] * nodes[None, :]  # (N, 8)
    # E_vals shape: (N, 8)
    zp1 = 1.0 + z_vals  # (N, 8)
    exp1 = 3.0 * (1.0 + w0[:, None] + wa[:, None])  # (N, 1)
    E_vals = np.sqrt(np.power(zp1, exp1) * np.power(1.0 + wa[:, None] * z_vals, -3.0 * wa[:, None]))
    # result: sum_i weights[i] / E(z*u_i) * z for each grid point
    return z * np.sum(weights[None, :] / E_vals, axis=1)


def _rel_errors_vec(nodes, weights):
    """Compute relative errors for all grid points vectorized."""
    quad_vals = _quad_comoving_vec(_Z_FLAT, _W0_FLAT, _WA_FLAT, nodes, weights)
    return np.abs(quad_vals - _EXACT_COMOVING) / np.maximum(np.abs(_EXACT_COMOVING), 1e-30)


def _worst_error_smooth(nodes, weights, beta=50.0):
    """Compute smooth max (log-sum-exp) of relative errors, vectorized."""
    errors = _rel_errors_vec(nodes, weights)
    m = np.max(errors)
    if m < 1e-15:
        return m
    log_sum_exp = m + np.log(np.sum(np.exp(beta * (errors - m)))) / beta
    return log_sum_exp


def _worst_error_exact(nodes, weights):
    """Compute exact worst relative error, vectorized."""
    errors = _rel_errors_vec(nodes, weights)
    return np.max(errors)
>>>>>>> REPLACE

Now let me update the `_params_to_nw` and `_optimize` functions to use the vectorized versions:


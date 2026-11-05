"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _evaluate(h):
    """Compute the task value for a given height vector."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s * s))


def _build_h(params, n_harm=8):
    """Build h from DFT coefficients: h[k] = sum_j a_j * cos(2*pi*j*k/N + phi_j)."""
    k = np.arange(N)
    h = np.zeros(N)
    for j in range(n_harm):
        a = params[2 * j]
        phi = params[2 * j + 1]
        h += a * np.cos(2 * np.pi * j * k / N + phi)
    return h


def _objective(params, n_harm=8):
    h = _build_h(params, n_harm)
    val = _evaluate(h)
    if not np.isfinite(val):
        return 1e10
    return val


def _chirp_objective(params, n_harm=8):
    """Quadratic phase chirp: h[k] = A * cos(2*pi*f0*k/N + B*k^2/N + phi)."""
    A, f0, B, phi = params
    k = np.arange(N, dtype=float)
    h = A * np.cos(2 * np.pi * f0 * k / N + B * k**2 / N + phi)
    val = _evaluate(h)
    if not np.isfinite(val):
        return 1e10
    return val


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_h = np.ones(N)
    best_val = 2.0

    n_harm = 8
    n_params = 2 * n_harm

    # Global search with multiple random starts
    np.random.seed(42)
    n_starts = 12
    max_iter = 200

    for i in range(n_starts):
        if time.time() > deadline - 10.0:
            break

        # Random initial parameters
        x0 = np.random.randn(n_params) * 0.5
        res = minimize(_objective, x0, args=(n_harm,), method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < best_val:
            h_candidate = _build_h(res.x, n_harm)
            v = _evaluate(h_candidate)
            if v < best_val and np.isfinite(v):
                best_val = v
                best_h = h_candidate.copy()

    # Chirp-based search
    for i in range(6):
        if time.time() > deadline - 5.0:
            break
        x0 = np.array([1.0, np.random.uniform(0.5, 5.0),
                       np.random.uniform(-10, 10), np.random.uniform(0, 2 * np.pi)])
        res = minimize(_chirp_objective, x0, method='L-BFGS-B',
                       options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < best_val:
            A, f0, B, phi = res.x
            k = np.arange(N, dtype=float)
            h_candidate = A * np.cos(2 * np.pi * f0 * k / N + B * k**2 / N + phi)
            v = _evaluate(h_candidate)
            if v < best_val and np.isfinite(v):
                best_val = v
                best_h = h_candidate.copy()

    # Final local polish on best_h using direct gradient-free optimization
    if time.time() < deadline - 5.0:
        def _direct_obj(h_flat):
            h = h_flat.reshape(N)
            v = _evaluate(h)
            return v if np.isfinite(v) else 1e10

        res = minimize(_direct_obj, best_h, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_val and np.isfinite(res.fun):
            best_val = res.fun
            best_h = res.x.reshape(N)

    # Ensure finite and non-zero sum
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END

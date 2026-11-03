"""Logarithmic energy minimization for N points on the unit sphere."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0          # hard kill from evaluator
_SAFETY = 15.0             # safety margin -> work until ~135s
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _log_energy(X):
    """Sum_{i<j} log(1/|x_i - x_j|) for unit points X (N,3)."""
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-18)
    d = np.sqrt(d2)
    m = np.triu_indices(N, k=1)
    return float(np.sum(np.log(1.0 / d[m])))


def _log_energy_grad(X):
    """Value and gradient of the logarithmic energy."""
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-18)
    d = np.sqrt(d2)
    # dE/dX_i = - sum_{j!=i} (X_i - X_j) / |X_i - X_j|^2
    # (X_i - X_j)/d^2 with the 1/d^2 weight; note gradient of log(1/d) w.r.t. X_i is -(X_i-X_j)/d^2
    inv_d2 = np.zeros_like(d2)
    mask = ~np.eye(N, dtype=bool)
    inv_d2[mask] = 1.0 / d2[mask]
    G = -1.0 * (X * inv_d2.sum(axis=1, keepdims=True) - (X * inv_d2) @ np.ones((N, N)).T)
    # Simpler correct form: G_i = -sum_j (X_i - X_j)/d2_ij
    G = np.zeros_like(X)
    for i in range(N):
        diff = X[i] - X  # (N,3)
        G[i] = -np.sum(diff * inv_d2[i, None], axis=0)
    m = np.triu_indices(N, k=1)
    val = float(np.sum(np.log(1.0 / d[m])))
    return val, G


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _fibonacci(jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        X = X + jitter * rng.standard_normal(X.shape)
    return _project(X)


def _random_sphere(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, 3))
    return _project(v)


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    # min pairwise distance
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def construct():
    best = None
    best_val = np.inf

    # Start with a solid deterministic configuration
    start0 = _fibonacci(jitter=0.0, seed=0)
    try:
        v0, _ = _log_energy_grad(start0)
    except Exception:
        v0 = _log_energy(start0)
    if v0 < best_val:
        best_val, best = v0, start0.copy()

    # Candidate starts: pure spiral, jittered spirals, random
    starts = []
    starts.append(start0)
    for s in range(6):
        starts.append(_fibonacci(jitter=0.05, seed=100 + s))
    for s in range(4):
        starts.append(_random_sphere(seed=200 + s))

    # Time-budgeted refinement with L-BFGS-B
    for idx, X0 in enumerate(starts):
        if _remaining() < 3.0:
            break
        try:
            res = minimize(_log_energy_grad, X0, method='L-BFGS-B',
                           jac=True, options={'maxiter': 4000, 'gtol': 1e-12, 'ftol': 1e-15})
            X = _project(np.asarray(res.x, dtype=float))
            if not _valid(X):
                continue
            val = _log_energy(X)
            if val < best_val:
                best_val, best = val, X.copy()
        except Exception:
            continue
        # A second, finer polish pass on the current best if time permits
        if _remaining() > 25.0 and best is not None:
            try:
                res2 = minimize(_log_energy_grad, best, method='L-BFGS-B',
                                jac=True, options={'maxiter': 4000, 'gtol': 1e-13, 'ftol': 1e-15})
                X2 = _project(np.asarray(res2.x, dtype=float))
                if _valid(X2):
                    val2 = _log_energy(X2)
                    if val2 < best_val:
                        best_val, best = val2, X2.copy()
            except Exception:
                pass

    # Final polish on the global best
    if _remaining() > 10.0 and best is not None:
        try:
            res3 = minimize(_log_energy_grad, best, method='L-BFGS-B',
                            jac=True, options={'maxiter': 4000, 'gtol': 1e-14, 'ftol': 1e-15})
            X3 = _project(np.asarray(res3.x, dtype=float))
            if _valid(X3):
                val3 = _log_energy(X3)
                if val3 < best_val:
                    best_val, best = val3, X3.copy()
        except Exception:
            pass

    if best is None:
        best = start0
    best = _project(best)
    return best

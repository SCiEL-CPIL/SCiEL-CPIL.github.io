import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 100
NPAIRS = N * (N - 1) // 2
_IU = np.triu_indices(N, 1)


def _energy(X):
    """Accurate logarithmic energy using math.fsum."""
    XTX = X @ X.T
    D2 = 2.0 - 2.0 * XTX
    np.fill_diagonal(D2, 1.0)
    D = np.sqrt(np.clip(D2, 1e-30, None))
    vals = np.log(1.0 / D[_IU])
    return math.fsum(vals)


def _energy_fast(X):
    """Fast vectorized energy."""
    XTX = X @ X.T
    D2 = 2.0 - 2.0 * XTX
    np.fill_diagonal(D2, 1.0)
    D = np.sqrt(np.clip(D2, 1e-30, None))
    return np.sum(np.log(1.0 / D[_IU]))


def _gradient(X):
    """Gradient of E = sum_{i<j} -log|d_ij| w.r.t. X.

    dE/dx_i = sum_{j != i} (x_j - x_i)/D2_ij = (M@X)_i - S_i x_i.
    """
    XTX = X @ X.T
    D2 = 2.0 - 2.0 * XTX
    np.fill_diagonal(D2, 1.0)
    M = 1.0 / D2
    np.fill_diagonal(M, 0.0)
    S = np.sum(M, axis=1)
    W = M @ X
    G = W - S[:, np.newaxis] * X
    return G


def _project_tangent(G, X):
    G_dot_X = np.sum(G * X, axis=1, keepdims=True)
    return G - G_dot_X * X


def _normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return X / norms


def _retract(X, step):
    """Retraction onto the sphere (project X + step)."""
    return _normalize(X + step)


def _fibonacci_start():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random
    X = rng.randn(N, 3)
    return _normalize(X)


def _jittered_fibonacci(jitter=0.1, seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random
    base = _fibonacci_start()
    X = base + jitter * rng.randn(N, 3)
    return _normalize(X)


def _cg_optimize(X0, max_iter=20000, time_limit=60.0, tol=1e-10):
    """Conjugate gradient on the tangent space with retraction line search."""
    start = time.time()
    X = _normalize(X0.copy())
    E = _energy_fast(X)
    best_X = X.copy()
    best_E = E

    G = _project_tangent(_gradient(X), X)
    d = -G
    gnorm = np.linalg.norm(G)

    for it in range(max_iter):
        if time.time() - start > time_limit - 3.0:
            break
        if gnorm < tol:
            break

        # Backtracking line search with retraction
        alpha = 0.01
        for _ in range(30):
            X_new = _retract(X, alpha * d)
            E_new = _energy_fast(X_new)
            if E_new < E:
                break
            alpha *= 0.5
        else:
            break

        X = X_new
        E = E_new
        if E < best_E:
            best_E = E
            best_X = X.copy()

        G_new = _project_tangent(_gradient(X), X)
        gnorm = np.linalg.norm(G_new)

        ratio = np.sum(G_new * G_new) / max(np.sum(G * G), 1e-15)
        d = -G_new + ratio * d
        G = G_new

    return best_X, best_E


def _lbfgs_optimize(X0, max_iter=3000, time_limit=40.0):
    """L-BFGS-B on the 300-dim ambient space with retraction inside."""
    start = time.time()

    def objective_and_grad(params):
        Xp = _normalize(params.reshape(N, 3))
        E = _energy_fast(Xp)
        G = _project_tangent(_gradient(Xp), Xp)
        return E, G.reshape(-1)

    best_X = _normalize(X0.copy())
    best_E = _energy_fast(best_X)
    params = best_X.reshape(-1).copy()

    for restart in range(4):
        if time.time() - start > time_limit - 5.0:
            break
        try:
            result = minimize(
                lambda p: objective_and_grad(p)[0],
                params,
                jac=lambda p: objective_and_grad(p)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter // 4, 'gtol': 1e-12, 'factr': 1e12},
            )
            X_new = _normalize(result.x.reshape(N, 3))
            E_new = _energy_fast(X_new)
            if E_new < best_E:
                best_E = E_new
                best_X = X_new.copy()
            if time.time() - start > time_limit - 8.0:
                break
            params = _normalize(X_new + 0.005 * np.random.randn(N, 3)).reshape(-1)
        except Exception:
            break

    return best_X, best_E


def _basin_hopping(X0, time_limit=60.0, max_iter=400):
    """Basin hopping with a smooth soft-min surrogate for the acceptance test."""
    start = time.time()
    X = _normalize(X0.copy())
    E = _energy_fast(X)
    best_X = X.copy()
    best_E = E
    temp = 1.0

    for it in range(max_iter):
        if time.time() - start > time_limit - 3.0:
            break

        # Perturb: move each point by a small random tangent displacement
        rng = np.random
        disp = rng.randn(N, 3)
        disp -= np.sum(disp * X, axis=1, keepdims=True) * X
        scale = 0.05 * (1.0 + 0.5 * math.sin(0.05 * it))
        X_try = _normalize(X + scale * disp)

        # Local polish of the perturbed point
        X_pol, E_pol = _cg_optimize(X_try, max_iter=1500, time_limit=6.0, tol=1e-9)

        # Smooth surrogate: soft-min of the two energies (log-sum-exp based)
        # Use a temperature-scaled soft-min to make acceptance smooth.
        beta = 1.0 / max(temp, 1e-6)
        a, b = E, E_pol
        m = min(a, b)
        softmin = m - (1.0 / beta) * math.log(math.exp(-beta * a - m * 0) + math.exp(-beta * b - m * 0))
        # Simpler: accept if E_pol < E or with Metropolis using soft-min gap
        gap = E_pol - E
        if gap < 0 or (temp > 1e-4 and rng.rand() < math.exp(-gap / max(temp, 1e-4))):
            X = X_pol
            E = E_pol
            if E < best_E:
                best_E = E
                best_X = X.copy()
        temp *= 0.985

    return best_X, best_E


def construct():
    start_time = time.time()
    deadline = 150.0
    margin = 15.0

    def remaining():
        return deadline - time.time() - margin

    best_X = _fibonacci_start()
    best_E = _energy_fast(best_X)

    # Diverse starting points
    starts = [_fibonacci_start()]
    for s in range(6):
        starts.append(_random_start(seed=100 + s))
    for s in range(3):
        starts.append(_jittered_fibonacci(jitter=0.05 + 0.05 * s, seed=200 + s))

    # Phase 1: L-BFGS from each start
    for i, X0 in enumerate(starts):
        if remaining() < 15.0:
            break
        X_opt, E_opt = _lbfgs_optimize(X0, max_iter=3000, time_limit=min(35.0, remaining() - 5.0))
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()

    # Phase 2: CG polish on the best
    if remaining() > 25.0:
        X_pol, E_pol = _cg_optimize(best_X, max_iter=20000, time_limit=min(60.0, remaining() - 5.0), tol=1e-11)
        if E_pol < best_E:
            best_E = E_pol
            best_X = X_pol.copy()

    # Phase 3: Basin hopping for global exploration
    if remaining() > 30.0:
        X_bh, E_bh = _basin_hopping(best_X, time_limit=min(50.0, remaining() - 10.0))
        if E_bh < best_E:
            best_E = E_bh
            best_X = X_bh.copy()

    # Phase 4: Final high-precision polish
    if remaining() > 15.0:
        X_fin, E_fin = _cg_optimize(best_X, max_iter=30000, time_limit=min(40.0, remaining() - 3.0), tol=1e-12)
        if E_fin < best_E:
            best_E = E_fin
            best_X = X_fin.copy()

    return _normalize(best_X)

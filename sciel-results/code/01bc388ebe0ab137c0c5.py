"""Logarithmic energy minimizer for 100 points on the unit sphere."""
import math
import time
import numpy as np

N = 100
_TIME_LIMIT = 150.0
_MARGIN = 8.0


def _log_energy(X):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = X.shape[0]
    terms = []
    for i in range(n):
        d = np.linalg.norm(X - X[i], axis=1)
        for j in range(i + 1, n):
            terms.append(math.log(1.0 / d[j]))
    return math.fsum(terms)


def _fibonacci_points():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _energy_and_grad(X):
    """Energy and gradient (in ambient R^3) of the logarithmic energy."""
    n = X.shape[0]
    X = np.asarray(X, dtype=float)
    # Pairwise squared distances
    sq = np.sum(X * X, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(D2, 0.0)
    D2 = np.maximum(D2, 1e-300)
    D = np.sqrt(D2)
    # log(1/D) = -log(D)
    E = 0.0
    for i in range(n):
        row = D[i]
        for j in range(i + 1, n):
            E += math.log(1.0 / row[j])
    # Gradient: d/dX_i sum_{j != i} log(1/|X_i - X_j|) = -sum_{j != i} (X_i - X_j)/|X_i - X_j|^2
    G = np.zeros_like(X)
    invD2 = np.zeros_like(D2)
    np.fill_diagonal(D2, 1.0)
    invD2 = 1.0 / D2
    np.fill_diagonal(invD2, 0.0)
    # G_i = -sum_j (X_i - X_j) / D2_ij = sum_j (X_j - X_i)/D2_ij
    for i in range(n):
        w = invD2[i]
        G[i] = np.sum(X * w[:, None], axis=0) - w.sum() * X[i]
    return E, G


def _project(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-300)
    return X / norms


def _line_search(X, G, max_iter=60):
    """Backtracking line search with sphere projection."""
    best_X = _project(X)
    best_E, _ = _energy_and_grad(best_X)
    Gproj = G - (G * best_X).sum(axis=1, keepdims=True) * best_X
    step = 0.5
    for _ in range(max_iter):
        cand = _project(best_X - step * Gproj)
        E, Gc = _energy_and_grad(cand)
        if E < best_E - 1e-12:
            return cand, E, Gc, step
        step *= 0.5
    return best_X, best_E, Gproj, step


def _optimize(X0, time_start, max_iter=3000):
    """Gradient descent with line search, projected to sphere."""
    X = _project(X0.copy())
    E, G = _energy_and_grad(X)
    best_X, best_E = X.copy(), E
    for it in range(max_iter):
        if time.time() - time_start > _TIME_LIMIT - _MARGIN:
            break
        X, E, G, _ = _line_search(X, G)
        if E < best_E:
            best_E = E
            best_X = X.copy()
    return best_X, best_E


def _polish(X0, time_start, max_iter=2000):
    """High-precision polish: tighter tolerance, more iterations."""
    X = _project(X0.copy())
    E, G = _energy_and_grad(X)
    best_X, best_E = X.copy(), E
    step = 0.5
    for it in range(max_iter):
        if time.time() - time_start > _TIME_LIMIT - _MARGIN:
            break
        Gproj = G - (G * X).sum(axis=1, keepdims=True) * X
        cand = _project(X - step * Gproj)
        E, G = _energy_and_grad(cand)
        if E < best_E - 1e-14:
            best_E = E
            best_X = cand.copy()
            X = cand
            step = min(step * 1.1, 5.0)
        else:
            step *= 0.7
            if step < 1e-9:
                break
    return best_X, best_E


def construct():
    t0 = time.time()
    # Start from Fibonacci spiral (good global structure)
    X0 = _fibonacci_points()
    X, E = _optimize(X0, t0, max_iter=2000)

    # Try a few random restarts if time permits
    for seed in range(5):
        if time.time() - t0 > _TIME_LIMIT - _MARGIN:
            break
        Xr = _random_points(seed)
        Xr, Er = _optimize(Xr, t0, max_iter=800)
        if Er < E:
            X, E = Xr, Er

    # Final high-precision polish
    if time.time() - t0 < _TIME_LIMIT - _MARGIN:
        X, E = _polish(X, t0, max_iter=2000)

    # Final validation
    X = _project(X)
    if not np.all(np.isfinite(X)):
        return _fibonacci_points()
    return X

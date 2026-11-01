"""Minimal logarithmic energy of 100 points on the unit sphere."""
import time
import math
import numpy as np

N = 100

_DEADLINE = 120.0          # hard wall-clock limit (s)
_SAFETY = 10.0             # safety margin (s)
_FLOOR = 1e-12             # floor on pair distances for smooth log
_START = time.time()


def _remaining():
    """Seconds left before the safety margin is hit."""
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _energy(X):
    """Logarithmic energy sum_{i<j} log(1/d_ij) with a floor on distances."""
    X = np.asarray(X, dtype=float)
    d = np.sqrt(np.maximum(1.0 - X @ X.T, 0.0))
    np.fill_diagonal(d, 1.0)
    d = np.maximum(d, _FLOOR)
    E = -math.fsum(np.log(d[np.triu_indices(N, k=1)]).ravel().tolist())
    return float(E)


def _energy_and_grad(X):
    """Return (energy, gradient) of the logarithmic energy on the sphere."""
    X = np.asarray(X, dtype=float)
    d = np.sqrt(np.maximum(1.0 - X @ X.T, 0.0))
    np.fill_diagonal(d, 1.0)
    d = np.maximum(d, _FLOOR)
    logd = np.log(d)
    np.fill_diagonal(logd, 0.0)
    E = -math.fsum(logd[np.triu_indices(N, k=1)].ravel().tolist())
    G = -2.0 * (logd[:, None] * X)      # d/dX_ij log(1/d_ij) summed over j
    return float(E), G


def _project(X):
    """Re-normalize each point to the unit sphere."""
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-15)
    return X / nrm


def _fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedral():
    t = (1.0 + math.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    X = np.repeat(base, 8, axis=0)
    # small deterministic jitter to break the exact symmetry
    rng = np.random.default_rng(12345)
    X = X + 0.01 * rng.standard_normal(X.shape)
    return _project(X)


def _random(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _riemannian_polish(X, max_iter=40000):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    X = _project(X)
    E, G = _energy_and_grad(X)
    lr = 1.0
    best_E, best_X = E, X.copy()
    for it in range(max_iter):
        if _remaining() < 0.1:
            break
        if np.all(np.abs(G) < 1e-10):
            break
        # project gradient onto the tangent space
        Gt = G - (G * X).sum(axis=1, keepdims=True) * X
        step = -lr * Gt
        # backtracking line search
        for _ in range(40):
            Xn = _project(X + step)
            En, _ = _energy_and_grad(Xn)
            if En < E - 1e-9:
                break
            step *= 0.5
        else:
            break
        if En < E - 1e-9:
            E, G = En, _energy_and_grad(Xn)[1]
            X = Xn
            if E < best_E - 1e-12:
                best_E, best_X = E, X.copy()
            lr = min(lr * 1.02, 3.0)
        else:
            lr *= 0.5
            if lr < 1e-9:
                break
    return best_X, best_E


def construct():
    global _START
    _START = time.time()
    starts = [_fibonacci(), _icosahedral()]
    for s in (1, 7, 42, 99, 2024, 55555, 31337, 88888):
        starts.append(_random(s))

    best_X, best_E = None, math.inf
    for X0 in starts:
        if _remaining() < 0.5:
            break
        Xp, Ep = _riemannian_polish(X0)
        if Ep < best_E:
            best_E, best_X = Ep, Xp

    # final re-normalization and validity guard
    out = _project(best_X)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-15)
    out = out / nrm
    return out


if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("energy:", _energy(pts), "time:", time.time() - t0)
    print("shape:", pts.shape, "finite:", np.all(np.isfinite(pts)))

"""Sphere covering with 130 equal caps -- gradient descent on the facet circumradius."""
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 130
_T0 = time.time()
_DEADLINE = _T0 + 130.0          # safety margin well under the 150 s kill limit


def _golden(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    p = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def _facets(p):
    """Return (circumradii, per-vertex unit gradient of circumradius, face->vertex map)."""
    try:
        tri = Delaunay(p).simplices
    except Exception:
        return None
    P = p[tri]                                  # (F,3,3)
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    v1 = b - a
    v2 = c - a
    cross = np.cross(v1, v2)
    nrm = np.linalg.norm(cross, axis=1, keepdims=True)
    eps = 1e-14
    nrm = np.where(nrm < eps, eps, nrm)
    n = cross / nrm                             # outward normal (CCW winding)
    d = np.einsum('ij,ij->i', a, n)             # plane offset
    # circumcenter in plane: O = n*d + alpha*v1 + beta*v2
    det = np.einsum('ij,ij->i', cross, cross)
    det = np.where(np.abs(det) < eps, eps, det)
    alpha = np.einsum('ij,ij->i', v1, v2) / det
    beta = np.einsum('ij,ij->i', v2, v1) / det
    O = d * n + alpha * v1 + beta * v2
    R2 = np.einsum('ij,ij->i', O, O)            # = 1 + |n|*d^2
    R = np.sqrt(np.clip(R2, 0.0, None))
    # dR/dv1 = (O-v1)/R, dR/dv2 = (O-v2)/R, dR/dv3 = 0
    g = np.zeros_like(P)
    g[:, 0] = (O - a) / R[:, None]
    g[:, 1] = (O - b) / R[:, None]
    return R, g, tri


def _score(p):
    f = _facets(p)
    if f is None:
        return np.nan
    R, _, _ = f
    return float(np.max(R))


def _objective(p, w, lam):
    f = _facets(p)
    if f is None:
        return 1e6, np.zeros_like(p)
    R, g, tri = f
    w = w / w.sum()
    val = float(np.sum(w * R))
    grad = np.zeros_like(p)
    for k in range(3):
        grad[tri[:, k]] += w[:, None] * g[:, k]
    return val, lam * grad


def _step(p, grad, lr):
    """Gradient step on the sphere via normalized retraction."""
    p2 = p - lr * grad
    n = np.linalg.norm(p2, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    p2 = p2 / n
    # keep points distinct
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(p2[i] - p2[j]) < 1e-9:
                p2[j] = p2[j] + 1e-9 * (np.random.rand(3) - 0.5)
                p2[j] /= np.linalg.norm(p2[j])
    return p2


def _polish(p, budget):
    """High-precision local polish (tighter tolerance, more iterations)."""
    best = p.copy()
    bestval = _score(p)
    for _ in range(60):
        if time.time() > budget:
            break
        f = _facets(p)
        if f is None:
            break
        R, g, tri = f
        # exact max-facet weighting (hardest facets get the update)
        i = int(np.argmax(R))
        grad = np.zeros_like(p)
        for k in range(3):
            grad[tri[i, k]] += g[i, k]
        lr = 1e-5
        p = _step(p, grad, lr)
        v = _score(p)
        if v < bestval - 1e-12:
            bestval = v
            best = p.copy()
        else:
            break
    return best, bestval


def construct():
    t0 = time.time()
    p = _golden(N)
    best = p.copy()
    bestval = _score(p)

    w = np.ones(N)
    lam = 0.5
    lr = 0.02

    while time.time() < _DEADLINE - 8.0:
        val, grad = _objective(p, w, lam)
        if not np.isfinite(val):
            p = _golden(N)
            continue
        p = _step(p, grad, lr)
        f = _facets(p)
        if f is not None:
            R, _, _ = f
            # re-weight toward the worst facets (exponential focus)
            w = np.exp(6.0 * R)
            v = float(np.max(R))
            if v < bestval - 1e-12:
                bestval = v
                best = p.copy()
        # decay learning rate slowly
        lr = max(0.0005, lr * 0.995)
        if time.time() > _DEADLINE - 45.0:
            break

    # final high-precision polish pass
    if time.time() < _DEADLINE - 12.0:
        best, bestval = _polish(best, _DEADLINE - 8.0)

    return best / np.linalg.norm(best, axis=1, keepdims=True)

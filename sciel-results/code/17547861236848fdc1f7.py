"""Minimal logarithmic energy of 100 points on the sphere."""
import math
import time
import numpy as np

N = 100


def _energy(pts):
    """Logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(d[iu]))


def _grad(pts):
    """Tangent-space gradient of the energy on the sphere (vectorized)."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    d[~np.eye(N, dtype=bool)] += 1e-300
    # g[i] = sum_j (x_i - x_j)/d_ij^2
    g = np.einsum('ij,ijk->ik', 1.0 / (d * d),
                  pts[:, None, :] - pts[None, :, :])
    # project off the radial (normal) direction
    g -= np.einsum('i,ij->ij', g, pts) * pts
    return g


def _fibonacci(k=0):
    """Golden-angle spiral start with a small rotational offset."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + k * 0.37
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random(nseeds=8):
    rng = np.random.default_rng(12345)
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _momentum(pts, t0, budget, lr0=2e-3, lr1=5e-5, gtol=1e-10):
    """Momentum gradient descent with adaptive step, staying on the sphere."""
    pts = pts.copy()
    vel = np.zeros_like(pts)
    best = pts.copy()
    best_e = _energy(pts)
    e = best_e
    beta = 0.9
    it = 0
    while time.time() - t0 < budget:
        g = _grad(pts)
        gnorm = np.linalg.norm(g)
        # adaptive step: shrink near singular pairs, else decay over iterations
        step = min(lr0, lr1 + (lr0 - lr1) * np.exp(-it / 400.0))
        vel = beta * vel - step * g
        cand = pts + vel
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        new_e = _energy(cand)
        if new_e < e - 1e-14:
            pts = cand
            e = new_e
            if e < best_e:
                best_e = e
                best = pts.copy()
        else:
            vel *= 0.5
            step *= 0.5
        if gnorm < gtol:
            break
        it += 1
    return best, best_e


def construct():
    t0 = time.time()
    deadline = 148.0
    best_pts = _fibonacci(0)
    best_e = _energy(best_pts)

    starts = [best_pts] + [(_random() if i % 2 else _fibonacci(i * 7))
                           for i in range(1, 9)]

    budget = 12.0
    for s in starts:
        if time.time() - t0 > deadline - 4.0:
            break
        pts, e = _momentum(s, t0, budget)
        if e < best_e:
            best_e = e
            best_pts = pts

    # final polish on the global best with a generous remaining budget
    if time.time() - t0 < deadline - 2.0:
        pts, e = _momentum(best_pts, t0, deadline - time.time() - 2.0,
                           lr0=5e-4, lr1=1e-6, gtol=1e-11)
        if e < best_e:
            best_e = e
            best_pts = pts

    # guarantee valid unit vectors and a finite result
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts

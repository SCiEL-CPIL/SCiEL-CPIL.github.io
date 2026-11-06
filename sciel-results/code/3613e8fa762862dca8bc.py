"""Covering the sphere with 50 equal caps: direct smooth optimization of the covering radius."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 50
_DEADLINE = 150.0


def _fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _covering_radius_deg(X):
    try:
        hull = ConvexHull(X)
        best = 0.0
        for s in hull.simplices:
            a, b, c = X[s[0]], X[s[1]], X[s[2]]
            ab = np.linalg.norm(b - a)
            bc = np.linalg.norm(c - b)
            ca = np.linalg.norm(a - c)
            ssum = ab + bc + ca
            if ssum <= 0:
                continue
            area = np.linalg.norm(np.cross(b - a, c - a)) * 0.5
            if area <= 1e-12:
                continue
            R = (ab * bc * ca) / (4.0 * area)
            if R > 1e-12:
                ang = np.arcsin(np.clip(1.0 / (2.0 * R), -1.0, 1.0))
                if ang > best:
                    best = ang
        return np.degrees(best)
    except Exception:
        return np.pi


def construct():
    t0 = time.time()
    time_limit = _DEADLINE - 2.0
    rng = np.random.default_rng(12345)

    def remaining():
        return time_limit - (time.time() - t0)

    best = None
    best_val = np.inf

    # ---- Stage 1: smooth epigraph optimization of the covering radius ----
    X = _fibonacci(N).astype(np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(nrm, 1e-12)

    t = -1.0
    beta = 25.0
    lr_t = 0.02
    lr_x = 0.02
    beta_max = 120.0
    min_rem = 0.25

    while remaining() > min_rem:
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        C = X @ X.T
        np.fill_diagonal(C, 1.0)
        C = np.clip(C, -1.0, 1.0)
        C_no_diag = C.copy()
        np.fill_diagonal(C_no_diag, -1.0)
        t = -np.max(np.min(C_no_diag, axis=1))
        t = np.clip(t, -1.0, 1.0)

        S = beta * (C - t)
        S = np.clip(S, -30.0, 30.0)
        S_max = np.max(S)
        expS = np.exp(S - S_max)
        expS[np.eye(N, dtype=bool)] = 0.0
        Z = np.sum(expS, axis=1)
        Z = np.maximum(Z, 1e-12)
        P = expS / Z[:, None]

        # logsumexp objective and gradient
        LSE = S_max + np.log(Z)
        gS = P
        grad_t = np.sum(gS, axis=1)
        grad_t = np.clip(grad_t, -1e12, 1e12)
        grad_X = (2.0 * (gS @ X)) / beta
        grad_X = np.clip(grad_X, -1e12, 1e12)

        # Riemannian update: descent on X, ascent on t
        X = X - lr_x * grad_X
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        t = t + lr_t * grad_t
        t = np.clip(t, -1.0, 1.0)

        # anneal the surrogate
        beta = min(beta * 1.01, beta_max)
        lr_t *= 0.995
        lr_x *= 0.998

        # track best by facet covering radius (the true metric)
        if remaining() > 0.5:
            cr = _covering_radius_deg(X)
            if cr < best_val:
                best_val = cr
                best = X.copy()

    if best is None:
        best = X
        best_val = _covering_radius_deg(best)

    # ---- Stage 2: facet-based polish (maximize the minimum facet circumradius) ----
    X = best.copy()
    time_limit2 = time_limit - 1.0
    lr = 0.002
    beta_p = 40.0
    while (time.time() - t0) < time_limit2:
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        try:
            hull = ConvexHull(X)
        except Exception:
            break
        radii = np.zeros(N)
        for s in hull.simplices:
            a, b, c = X[s[0]], X[s[1]], X[s[2]]
            ab = np.linalg.norm(b - a)
            bc = np.linalg.norm(c - b)
            ca = np.linalg.norm(a - c)
            area = np.linalg.norm(np.cross(b - a, c - a)) * 0.5
            if area <= 1e-12:
                continue
            R = (ab * bc * ca) / (4.0 * area)
            if R <= 1e-12:
                ang = 0.0
            else:
                ang = np.arcsin(np.clip(1.0 / (2.0 * R), -1.0, 1.0))
            for idx in s:
                if ang > radii[idx]:
                    radii[idx] = ang
        # soft-min of the facet circumradii
        S = -beta_p * radii
        S_max = np.max(S)
        expS = np.exp(S - S_max)
        Z = np.sum(expS)
        Z = max(Z, 1e-12)
        P = expS / Z
        # move each point away from its facet circumcenter (tangent plane)
        centers = np.zeros((N, 3))
        for s in hull.simplices:
            a, b, c = X[s[0]], X[s[1]], X[s[2]]
            center = a + b + c
            center = center / np.maximum(np.linalg.norm(center), 1e-12)
            for idx in s:
                centers[idx] += center
        centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        grad = (P[:, None] * centers)
        grad = grad - (grad * X[:, None]).sum(axis=1, keepdims=True) * X
        grad = np.clip(grad, -1e6, 1e6)
        X = X + lr * grad
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        lr *= 0.995
        beta_p = min(beta_p * 1.05, 200.0)

    # final best selection
    cr = _covering_radius_deg(X)
    if cr < best_val:
        best = X
        best_val = cr

    # ensure valid: unit sphere, nonzero, not in a closed hemisphere
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    if np.any(~np.isfinite(best)):
        best = _fibonacci(N)
    # hemisphere guard
    for _ in range(8):
        c = best.mean(axis=0)
        cn = np.linalg.norm(c)
        if cn < 1e-3:
            break
        c = c / cn
        if np.all(np.dot(best, c) <= 1e-6):
            best = -best
            break
    return best

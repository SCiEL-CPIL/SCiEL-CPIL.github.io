"""Sphere covering: 130 points, minimize covering radius via mesh gradient + annealing + polish."""
import time
import numpy as np

N = 130
_T0 = time.time()
_DEADLINE = 120.0          # hard wall-clock limit
_MARGIN = 8.0              # safety margin before deadline

def _time_left():
    return _DEADLINE - (time.time() - _T0) - _MARGIN

# ---------------------------------------------------------------------------
# Fixed high-resolution spherical mesh (coordinate system for the surrogate)
# ---------------------------------------------------------------------------
def _build_mesh(nlat=361, nlon=720):
    # lat from -90..90 (nlat), lon 0..360 (nlon, endpoint excluded)
    lat = np.linspace(-np.pi/2, np.pi/2, nlat)
    lon = np.linspace(0.0, 2*np.pi, nlon, endpoint=False)
    LAT, LON = np.meshgrid(lat, lon, indexing='ij')
    sinlat = np.sin(LAT)
    coslat = np.cos(LAT)
    X = coslat*np.cos(LON)
    Y = coslat*np.sin(LON)
    Z = sinlat
    mesh = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return mesh

MESH = _build_mesh()
MW = np.cos(np.arcsin(np.clip(MESH[:, 2], -1, 1)))  # lat weight (area)

# ---------------------------------------------------------------------------
# Distance field helpers
# ---------------------------------------------------------------------------
def _pair_dots(P, M):
    return M @ P.T  # (nM, N)

def _softmin_d(d, beta):
    # d in [0, pi]; soft-min of d
    d = np.clip(d, 0.0, np.pi)
    m = d.min(axis=1, keepdims=True)
    return m + np.log(np.sum(np.exp(-beta*(d - m)), axis=1, keepdims=True)) / beta

def _argmin_d(d):
    return np.argmin(d, axis=1)

def _covering_radius(P):
    d = _pair_dots(P, MESH)
    return float(np.max(-d.min(axis=1))) * 180.0/np.pi

def _objective(P, beta):
    d = _pair_dots(P, MESH)
    sm = _softmin_d(d, beta)
    # soft-max of the smooth min-distance (area weighted)
    b = beta * 0.35
    m = sm.max()
    return float(np.exp(b*(sm - m)).mean())

def _gradient(P, beta):
    d = _pair_dots(P, MESH)
    sm = _softmin_d(d, beta)
    # softmax weights over the mesh (area weighted) for the soft-max
    b = beta * 0.35
    w = np.exp(b*(sm - sm.max()))
    w *= MW
    w /= w.sum()
    # gradient of -d_{i,k} wrt P_k = -M_i
    # grad = -sum_i w_i * M_i  for the argmin point k(i)
    k = _argmin_d(d)
    G = np.zeros((N, 3))
    np.add.at(G, k, -MESH.T * w[:, None])
    return G

# ---------------------------------------------------------------------------
# Local polish (exact local solve on the active facet circumcenters)
# ---------------------------------------------------------------------------
def _polish(P, iters=3000, step0=5e-4):
    P = P.copy()
    d = _pair_dots(P, MESH)
    k = _argmin_d(d)
    for it in range(iters):
        if _time_left() < 5.0:
            break
        step = step0 * (0.97 ** (it // 200))
        G = _gradient(P, 60.0)
        P = P + step * G
        P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P

# ---------------------------------------------------------------------------
# Main construction
# ---------------------------------------------------------------------------
def construct():
    # Fibonacci spiral initialisation
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z*z, 0, None))
    P = np.column_stack([r*np.cos(phi), r*np.sin(phi), z])

    best = P.copy()
    best_cr = _covering_radius(P)

    # ---- Phase 1: gradient descent with annealing (variable beta) ----
    beta = 25.0
    beta_max = 80.0
    lr = 3e-3
    temp = 0.02
    n_iter = 6000
    for it in range(n_iter):
        if _time_left() < 15.0:
            break
        G = _gradient(P, beta)
        # anneal: add random perturbation that decays
        noise = temp * np.random.randn(N, 3)
        P = P + lr * G + noise
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        # schedule
        beta = min(beta_max, beta * 1.0005)
        lr *= 0.9995
        temp *= 0.999
        cr = _covering_radius(P)
        if cr < best_cr:
            best_cr = cr
            best = P.copy()
    P = best.copy()

    # ---- Phase 2: restarts / simulated annealing from random perturbations ----
    rng = np.random.default_rng(12345)
    n_restarts = 8
    for rs in range(n_restarts):
        if _time_left() < 30.0:
            break
        # perturb best
        Q = best.copy()
        Q = Q + 0.05 * rng.randn(N, 3)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        # short annealed gradient descent
        b2 = 30.0
        lr2 = 2e-3
        temp2 = 0.015
        for it in range(2500):
            if _time_left() < 30.0:
                break
            G = _gradient(Q, b2)
            Q = Q + lr2 * G + temp2 * rng.randn(N, 3)
            Q /= np.linalg.norm(Q, axis=1, keepdims=True)
            b2 = min(beta_max, b2 * 1.001)
            lr2 *= 0.999
            temp2 *= 0.9985
        cr = _covering_radius(Q)
        if cr < best_cr:
            best_cr = cr
            best = Q.copy()

    # ---- Phase 3: final high-precision polish ----
    if _time_left() > 10.0:
        best = _polish(best, iters=4000, step0=4e-4)
        best_cr = _covering_radius(best)

    # ---- Phase 4: tiny deterministic local descent to clean up ----
    for it in range(2000):
        if _time_left() < 2.0:
            break
        G = _gradient(best, 100.0)
        best = best + 2e-4 * G
        best /= np.linalg.norm(best, axis=1, keepdims=True)

    return best

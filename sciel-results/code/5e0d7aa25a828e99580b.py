"""Packing 12 unit hexagons in a regular hexagon via support-function optimization."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
_TIME_LIMIT = 150.0
_DEADLINE = time.time() + _TIME_LIMIT - 12.0  # safety margin

# 6 support directions (60 deg apart)
_DIRS = np.array([[math.cos(i * math.pi / 3.0), math.sin(i * math.pi / 3.0)]
                  for i in range(6)])
_COS = _DIRS[:, 0]
_SIN = _DIRS[:, 1]

# Support of a unit hexagon rotated by a in each direction: h(a)=|cos a|+|sin a|/sqrt(3)
_HBASE = np.abs(_COS) + np.abs(_SIN) / math.sqrt(3.0)

def _h(a):
    """Support of unit hexagon rotated by a (radians) in the 6 directions."""
    return np.abs(np.cos(a)) + np.abs(np.sin(a)) / math.sqrt(3.0)

def _objective_and_grad(z):
    """z: [x0,y0,...,x11,y11, a0,...,a11]. Returns (f, grad)."""
    P = z[:2 * N].reshape(N, 2)
    A = z[2 * N:]
    h = _h(A)  # (N,6)
    # outer span per direction
    U = P @ _DIRS.T  # (N,6)
    hi = U.max(axis=0)
    lo = U.min(axis=0)
    span = hi - lo
    f = np.max(span + 2.0 * _HBASE)
    k = int(np.argmax(span + 2.0 * _HBASE))
    # which index attains hi/lo in direction k
    hi_i = int(np.argmax(U[:, k]))
    lo_i = int(np.argmin(U[:, k]))
    # gradient of objective w.r.t. P in direction k
    gP = np.zeros_like(P)
    gP[hi_i, 0] += _COS[k]
    gP[hi_i, 1] += _SIN[k]
    gP[lo_i, 0] -= _COS[k]
    gP[lo_i, 1] -= _SIN[k]
    # gradient w.r.t. angles: d/dA of span = d/dA(hi - lo) = (h[hi_i]-h[lo_i])'
    dH = _h(A)  # reuse; but need derivative; compute analytically
    def _dh(a):
        # d/da |cos a| = -sin a * sign(cos a); d/da |sin a| = cos a * sign(sin a)
        c = np.cos(a); s = np.sin(a)
        dc = -s * np.sign(c)
        ds = c * np.sign(s)
        return dc + ds / math.sqrt(3.0)
    dA = _dh(A)  # (N,6)
    gA = np.zeros(N)
    gA[hi_i] += dA[hi_i, k]
    gA[lo_i] -= dA[lo_i, k]
    grad = np.concatenate([gP.ravel(), gA])
    return f, grad

def _solve_from(z0):
    res = minimize(_objective_and_grad, z0, jac=True, method='L-BFGS-B',
                   options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8})
    return res

def construct():
    best_z = None
    best_f = float('inf')

    # deterministic ring + center start
    def make_ring(R):
        z = np.zeros(3 * N)
        # center
        z[0:2] = [0.0, 0.0]
        z[2 * N + 0] = 0.0
        for i in range(1, N):
            ang = 2.0 * math.pi * (i - 1) / (N - 1)
            z[2 * i] = R * math.cos(ang)
            z[2 * i + 1] = R * math.sin(ang)
            z[2 * N + i] = 0.0
        return z

    starts = [make_ring(3.0), make_ring(2.5), make_ring(3.5)]
    # a couple of randomized starts
    rng = np.random.default_rng(12345)
    for _ in range(3):
        z = np.zeros(3 * N)
        z[:2 * N] = rng.normal(0.0, 2.0, size=(N, 2)).ravel()
        z[2 * N:] = rng.uniform(-math.pi / 6.0, math.pi / 6.0, size=N)
        starts.append(z)

    for z0 in starts:
        if time.time() > _DEADLINE:
            break
        try:
            res = _solve_from(z0)
            f, _ = _objective_and_grad(res.x)
            if f < best_f:
                best_f = f
                best_z = res.x
        except Exception:
            pass

    if best_z is None:
        # fallback: trivially valid large hexagon
        z = make_ring(3.0)
        best_z = z
        best_f = _objective_and_grad(z)[0]

    P = best_z[:2 * N].reshape(N, 2)
    A = best_z[2 * N:]
    inner = []
    for i in range(N):
        inner.append([float(P[i, 0]), float(P[i, 1]), float(math.degrees(A[i]))])
    side = float(best_f)
    return inner, (0.0, 0.0), side, 0.0

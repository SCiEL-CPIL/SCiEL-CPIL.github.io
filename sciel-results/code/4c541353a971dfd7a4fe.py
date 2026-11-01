"""Force-directed packing of 11 unit hexagons inside a minimal regular hexagon."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11
_T0 = time.time()
_DEADLINE = 140.0  # safety margin before 150 s kill

# Unit hexagon circumradius = 1. Minkowski sum of two same-orientation unit
# hexagons has circumradius 2 -> centers must be >= 2 apart.
_SEP = 2.0

# 6 outward normal directions (unit vectors) for a pointy-top hexagon.
_DIRS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))]
                  for a in range(0, 360, 60)])
_DIRS = _DIRS.astype(float)


def _hex_radius(p, beta):
    """Circumscribed radius (max |p . d|) of the hexagon centered at p, side s."""
    # s = beta / 2, circumradius = s.
    return np.max(np.abs(p @ _DIRS.T)) + beta / 2.0


def _softmax(x, beta):
    """Soft max of x with temperature 1/beta (log-sum-exp)."""
    m = np.max(x)
    return m + (1.0 / beta) * np.log(np.sum(np.exp(beta * (x - m))))


def _energy(pos, beta, lam, soft_beta):
    """Return (value, grad) of repulsion + soft outer-bound objective."""
    n = len(pos)
    val = 0.0
    grad = np.zeros_like(pos)
    # Pairwise repulsion (smooth): penalize d < _SEP
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[i] - pos[j]
            dist = math.hypot(d[0], d[1])
            if dist < _SEP - 1e-12:
                # quadratic penalty in (SEP - dist), force magnitude ~ (SEP-dist)
                f = lam * (_SEP - dist)
                if dist > 1e-12:
                    g = f / dist
                    grad[i] += g * d
                    grad[j] -= g * d
                val += lam * 0.5 * (_SEP - dist) ** 2
    # Soft outer bound: soft-max over directions of (|p.d| + s)
    s = beta / 2.0
    for i in range(n):
        r = np.abs(pos[i] @ _DIRS.T) + s
        # gradient of soft-max wrt each coordinate
        exps = np.exp(soft_beta * (r - np.max(r)))
        w = exps / np.sum(exps)
        # d r_k / d p_i = sign(p_i . d_k) * d_k
        signs = np.sign(pos[i] @ _DIRS.T)
        drdp = (signs * _DIRS).T  # shape (2,6)
        g = soft_beta * (drdp @ w)
        grad[i] += g
        val += _softmax(r, soft_beta)
    return val, grad


def _fit_outer(pos):
    """Exact outer side length for positions pos (side s = 1)."""
    radii = np.max(np.abs(pos @ _DIRS.T), axis=1) + 1.0
    return float(np.max(radii))


def _valid(pos, outer, tol=1e-7):
    """Check non-overlap (centers >= 2) and containment."""
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]) < _SEP - tol:
                return False
    # containment: max |p.d| + 1 <= outer + tol
    if np.max(np.max(np.abs(pos @ _DIRS.T), axis=1) + 1.0) > outer + tol:
        return False
    return True


def construct():
    global _T0
    # Best valid solution found so far.
    best_pos = None
    best_outer = None
    best_angle = 0.0

    # Initial seed: a compact ring + center.
    def seed(ring_r, center_r):
        p = np.zeros((N, 2))
        # center
        p[0] = [0.0, 0.0]
        # 6 around center at distance 2
        for k in range(6):
            a = math.radians(60 * k)
            p[1 + k] = [2.0 * math.cos(a), 2.0 * math.sin(a)]
        # 4 more in a second shell
        for k in range(4):
            a = math.radians(60 * k + 30)
            p[7 + k] = [3.6 * math.cos(a), 3.6 * math.sin(a)]
        return p

    pos = seed(2.0, 0.0)
    outer = _fit_outer(pos)
    if _valid(pos, outer):
        best_pos = pos.copy()
        best_outer = outer

    # Force-directed gradient descent with annealing.
    lam = 1.0
    soft_beta = 6.0
    lr = 0.05
    it = 0
    while True:
        if time.time() > _DEADLINE - 25.0:
            break
        val, grad = _energy(pos, beta=2.0, lam=lam, soft_beta=soft_beta)
        # adaptive step: limit movement
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            step = lr * grad / gnorm
            # clip step size
            maxstep = 0.15
            s = np.linalg.norm(step)
            if s > maxstep:
                step = step * (maxstep / s)
            pos = pos - step
        it += 1
        # anneal
        if it % 40 == 0:
            lr *= 0.97
            soft_beta = min(soft_beta * 1.02, 40.0)
        # record valid
        o = _fit_outer(pos)
        if _valid(pos, o):
            if best_outer is None or o < best_outer - 1e-9:
                best_outer = o
                best_pos = pos.copy()
        if it % 200 == 0 and lr < 1e-4:
            break

    # Polish with hard max via L-BFGS-B if time remains.
    if time.time() < _DEADLINE - 20.0:
        p0 = best_pos if best_pos is not None else pos

        def hard_obj_and_grad(x):
            p = x.reshape(-1, 2)
            # objective = max over i of (max_k |p_i.d_k| + 1)
            radii = np.max(np.abs(p @ _DIRS.T), axis=1) + 1.0
            obj = float(np.max(radii))
            g = np.zeros_like(x)
            # subgradient: pick active i, active k
            i = int(np.argmax(radii))
            vals = np.abs(p[i] @ _DIRS.T)
            k = int(np.argmax(vals))
            sign = 1.0 if (p[i] @ _DIRS.T)[k] >= 0 else -1.0
            g[2 * i:2 * i + 2] = sign * _DIRS[k]
            # add small repulsion for feasibility
            for a in range(N):
                for b in range(a + 1, N):
                    d = p[a] - p[b]
                    dist = math.hypot(d[0], d[1])
                    if dist < _SEP:
                        f = 5.0 * (_SEP - dist)
                        if dist > 1e-12:
                            gg = f / dist
                            g[2 * a:2 * a + 2] += gg * d
                            g[2 * b:2 * b + 2] -= gg * d
            return obj, g

        res = minimize(hard_obj_and_grad, p0.ravel(), method='L-BFGS-B',
                       options={'maxiter': 2000, 'gtol': 1e-10, 'maxls': 50})
        pf = res.x.reshape(-1, 2)
        of = _fit_outer(pf)
        if _valid(pf, of):
            if best_outer is None or of < best_outer - 1e-9:
                best_outer = of
                best_pos = pf.copy()

    # Final: use best valid positions.
    if best_pos is None:
        best_pos = pos
        best_outer = _fit_outer(pos)

    # Center the cluster at origin (outer hexagon centered at 0).
    # The outer hexagon orientation is fixed (pointy-top); inner hexagons
    # are all pointy-top (angle 0) which is optimal for this packing.
    inner = [[float(best_pos[i][0]), float(best_pos[i][1]), 0.0] for i in range(N)]
    return inner, (0.0, 0.0), float(best_outer), 0.0

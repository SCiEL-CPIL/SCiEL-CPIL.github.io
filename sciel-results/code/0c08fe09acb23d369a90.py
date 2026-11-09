"""Hexagon packing: place 11 unit regular hexagons to minimize enclosing hexagon side."""
import math
import time
import numpy as np

N = 11
_START = time.time()
_DEADLINE = _START + 110.0   # safety margin before the 120 s kill

# Unit regular hexagon vertices (flat-top), circumradius = 1.
_BASE = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
_EDGES = np.vstack([_BASE[1:] - _BASE[:-1], _BASE[0] - _BASE[-1]])  # 6 directed edges, |e|=1
_EDGE2 = _EDGES[1:] - _EDGES[:-1]
_EDGE2 = _EDGE2 / np.linalg.norm(_EDGE2, axis=1)[:, None]           # 6 edge-to-edge directions

# Three edge-normal directions (30, 90, 150 deg) for the enclosing-hexagon width.
_NORM = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in (30.0, 90.0, 150.0)])


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _edge_min_dist(V, W):
    """Minimum distance between two convex hexagons' boundaries (0 if overlapping)."""
    best = 1e18
    for i in range(6):
        e = V[i + 1] - V[i]
        e = e / (math.hypot(e[0], e[1]) + 1e-12)
        d = np.dot(W, e)
        j = int(np.argmin(d))
        w1, w2 = W[j], W[j + 1]
        a0, a1 = V[i], V[i + 1]
        for p, q in ((a0, a1), (a1, a0), (w1, w2), (w2, w1)):
            ap, aq = p - a0, q - a0
            t = float(np.dot(ap, e))
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            cp = a0 + t * e
            dx, dy = q[0] - cp[0], q[1] - cp[1]
            d2 = dx * dx + dy * dy
            if d2 < best:
                best = d2
    return math.sqrt(best)


def _width(v):
    """Width of a point set along the 3 enclosing-hexagon normal directions."""
    w = []
    for n in _NORM:
        p = np.dot(v, n)
        w.append(p.max() - p.min())
    return w


def _enclosing_side(v):
    return max(_width(v)) / math.sqrt(3.0)


def _eval(params):
    n = N
    verts = np.empty((n, 6, 2))
    for i in range(n):
        verts[i] = params[3 * i] + _BASE @ _rot(params[3 * i + 2])
    # enclosing side (global rotation already baked into params)
    side = _enclosing_side(verts.reshape(-1, 2))
    # smooth collision penalty: sum over pairs of (margin - dist)
    pen = 0.0
    margin = 0.001
    for i in range(n):
        vi = verts[i]
        for j in range(i + 1, n):
            d = _edge_min_dist(vi, verts[j])
            if d < margin:
                pen += (margin - d) ** 2
    return side, pen


def _pack():
    """Optimize 11 hexagons (x,y,theta) + global rotation to minimize enclosing side."""
    best_side = 1e18
    best_pen = 1e18
    best_params = None
    rng = np.random.default_rng(7)
    it = 0
    while time.time() < _DEADLINE:
        it += 1
        # ---- initial guess: a hexagonal cluster of 7 (radius 2) + 4 satellites ----
        theta_g = float(rng.uniform(0.0, math.pi / 3.0))
        R = _rot(theta_g)
        centers = []
        # central hexagon
        centers.append(np.array([0.0, 0.0]))
        # 6 neighbours at distance 2 (edge-to-edge touching)
        for k in range(6):
            ang = math.radians(60.0 * k + 30.0)
            centers.append(2.0 * np.array([math.cos(ang), math.sin(ang)]))
        # 4 satellites at the 4 "corner" gaps (radius ~3.46, angles 0/90/180/270)
        for k in range(4):
            ang = math.radians(90.0 * k)
            centers.append(3.464 * np.array([math.cos(ang), math.sin(ang)]))
        centers = np.array(centers)
        centers = centers @ R.T  # apply global rotation
        # local rotations: small random
        thetas = rng.uniform(-0.05, 0.05, size=N)
        params = np.empty(3 * N + 1)
        for i in range(N):
            params[3 * i] = centers[i][0]
            params[3 * i + 1] = centers[i][1]
            params[3 * i + 2] = thetas[i]
        params[3 * N] = theta_g
        params += rng.normal(0.0, 0.02, size=params.shape)

        # ---- coordinate-descent / Nelder-Mead-style local search ----
        step = 0.15
        cur_side, cur_pen = _eval(params)
        cur_cost = cur_side + 50.0 * cur_pen
        no_improve = 0
        while time.time() < _DEADLINE and no_improve < 400:
            # pick a random coordinate to nudge
            idx = int(rng.integers(0, params.shape[0]))
            old = params[idx]
            # adaptive step: shrink when stuck, grow when improving
            params[idx] = old + float(rng.normal(0.0, step))
            if idx == 3 * N:  # keep global rotation in [0, pi/3)
                params[idx] = params[idx] % (math.pi / 3.0)
            ns, npen = _eval(params)
            ncost = ns + 50.0 * npen
            if ncost < cur_cost:
                cur_cost, cur_side, cur_pen = ncost, ns, npen
                no_improve = 0
                step = min(0.5, step * 1.05)
            else:
                params[idx] = old
                no_improve += 1
                step = max(0.01, step * 0.9)
            # accept into global best if valid (no collision)
            if npen <= 1e-7 and ns < best_side:
                best_side = ns
                best_pen = npen
                best_params = params.copy()

        # ---- final polish with smaller steps on the best params ----
        if best_params is not None and time.time() < _DEADLINE:
            params = best_params.copy()
            step = 0.05
            cur_side, cur_pen = _eval(params)
            cur_cost = cur_side + 50.0 * cur_pen
            no_improve = 0
            while time.time() < _DEADLINE and no_improve < 300:
                idx = int(rng.integers(0, params.shape[0]))
                old = params[idx]
                params[idx] = old + float(rng.normal(0.0, step))
                if idx == 3 * N:
                    params[idx] = params[idx] % (math.pi / 3.0)
                ns, npen = _eval(params)
                ncost = ns + 50.0 * npen
                if ncost < cur_cost:
                    cur_cost, cur_side, cur_pen = ncost, ns, npen
                    no_improve = 0
                    step = min(0.1, step * 1.05)
                else:
                    params[idx] = old
                    no_improve += 1
                    step = max(0.005, step * 0.9)
                if npen <= 1e-7 and ns < best_side:
                    best_side = ns
                    best_pen = npen
                    best_params = params.copy()
    return best_params, best_side, best_pen


def construct():
    try:
        params, side, pen = _pack()
        if params is None:
            # fallback: the seed-like line (still valid, large side)
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
        # apply global rotation to centers and angles
        theta_g = float(params[3 * N]) % (math.pi / 3.0)
        R = _rot(theta_g)
        inner = []
        for i in range(N):
            cx = float(params[3 * i])
            cy = float(params[3 * i + 1])
            ca = float(params[3 * i + 2])
            # local center already includes global rotation in x,y; add global to angle
            ang = (ca + theta_g) % 360.0
            inner.append([cx, cy, ang])
        return inner, (0.0, 0.0), float(side), float(theta_g)
    except Exception:
        # safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0


if __name__ == "__main__":
    res = construct()
    print("outer side:", res[2], "rotation:", res[3], "n inner:", len(res[0]))
    print("elapsed:", time.time() - _START)

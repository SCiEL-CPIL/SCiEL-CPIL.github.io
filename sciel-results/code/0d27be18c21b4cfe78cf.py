"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n < 1e-14] = 1.0
    return pts / n


def _per_point_radius(pts):
    """For each point, radius of the largest spherical cap centered at it
    that stays within its Voronoi cell (i.e. circumradius of the facet it
    borders). This is the local contribution to the covering radius."""
    p = _normalize(pts)
    n = len(p)
    g = np.arccos(np.clip(p @ p.T, -1.0, 1.0))
    # for each i, sort neighbors by distance
    order = np.argsort(g, axis=1)
    radii = np.zeros(n)
    for i in range(n):
        d = g[i, order]
        # find the smallest j such that d[j] >= d[j-1] + d[j-2] (triangle closes)
        best = d[-1]
        # scan
        for j in range(2, n):
            if d[j] >= d[j - 1] + d[j - 2] - 1e-12:
                best = d[j - 1]
                break
        else:
            best = d[-1]
        radii[i] = best
    return radii


def _covering_radius(pts):
    return float(np.max(_per_point_radius(pts)))


def _force(pts, radii):
    """Repulsion force: each point pushed away from neighbors weighted by
    how much that neighbor's cell dominates."""
    p = _normalize(pts)
    n = len(p)
    g = np.arccos(np.clip(p @ p.T, -1.0, 1.0))
    f = np.zeros_like(p)
    for i in range(n):
        wi = radii[i]
        for j in range(n):
            if i == j:
                continue
            d = g[i, j]
            if d < 1e-12:
                continue
            # weight: larger when neighbor j has large radius
            w = radii[j]
            # repulsion magnitude ~ 1/d^2, capped
            mag = min(w / (d * d + 1e-6), 10.0)
            # direction: away from j, tangent to sphere
            v = -p[j]
            v = v - np.dot(v, p[i]) * p[i]
            nv = np.linalg.norm(v)
            if nv < 1e-14:
                continue
            f[i] += (mag / nv) * v
    return f


def _fib_sample():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150s

    rng = np.random.default_rng(12345)
    best = _fib_sample()
    best_r = _covering_radius(best)

    # multiple restarts with random + perturbations
    n_restarts = 8
    for rs in range(n_restarts):
        if time.time() > deadline:
            break
        if rs == 0:
            cur = _fib_sample()
        else:
            # random or perturbed best
            if rs % 2 == 0:
                cur = rng.normal(size=(N, 3))
            else:
                cur = best + rng.normal(scale=0.3, size=(N, 3))
        cur = _normalize(cur)
        cur_r = _covering_radius(cur)
        if cur_r < best_r:
            best, best_r = cur.copy(), cur_r

        # gradient-descent-like repulsion iterations
        step = 0.05
        iters = 600
        for it in range(iters):
            if time.time() > deadline:
                break
            radii = _per_point_radius(cur)
            f = _force(cur, radii)
            cur = _normalize(cur + step * f)
            # occasional annealing of step
            if it % 100 == 99:
                step *= 0.8
            r = _covering_radius(cur)
            if r < cur_r:
                cur_r = r
                if r < best_r:
                    best, best_r = cur.copy(), r
            else:
                # if stuck, nudge
                if it % 50 == 49:
                    cur = _normalize(cur + rng.normal(scale=0.05, size=(N, 3)))
                    cur_r = _covering_radius(cur)

    # final polish: a few more small steps on the best
    if time.time() < deadline - 5.0:
        cur = best.copy()
        step = 0.01
        for it in range(200):
            if time.time() > deadline:
                break
            radii = _per_point_radius(cur)
            f = _force(cur, radii)
            cur = _normalize(cur + step * f)
            r = _covering_radius(cur)
            if r < best_r:
                best, best_r = cur.copy(), r

    return _normalize(best)
# EVOLVE-BLOCK-END

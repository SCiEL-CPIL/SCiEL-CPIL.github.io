# EVOLVE-BLOCK-START
import time
import numpy as np


def _hex_verts(cx, cy, d, ang_deg):
    a = np.deg2rad(ang_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + d * np.cos(th), cy + d * np.sin(th)], axis=1)


def _separation(a, b):
    """Minimum signed gap between two convex polygons (positive = separated)."""
    best = -np.inf
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            e = p2 - p1
            nrm = np.array([-e[1], e[0]])
            nl = np.linalg.norm(nrm)
            if nl < 1e-15:
                continue
            nrm = nrm / nl
            da = np.dot(a - p1, nrm)
            db = np.dot(b - p1, nrm)
            min_a = da.min()
            max_a = da.max()
            min_b = db.min()
            max_b = db.max()
            gap = min(min_a - max_b, min_b - max_a)
            if gap > best:
                best = gap
    return best


def _separation_grads(a, b, a_centers, b_centers):
    """Return (gap, grad_a, grad_b) for the separating axis giving the max gap."""
    best = -np.inf
    best_ga = None
    best_gb = None
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            e = p2 - p1
            nrm = np.array([-e[1], e[0]])
            nl = np.linalg.norm(nrm)
            if nl < 1e-15:
                continue
            nrm = nrm / nl
            da = np.dot(a - p1, nrm)
            db = np.dot(b - p1, nrm)
            min_a = da.min()
            max_a = da.max()
            min_b = db.min()
            max_b = db.max()
            gap = min(min_a - max_b, min_b - max_a)
            if gap > best:
                best = gap
                if min_a - max_b <= min_b - max_a:
                    # a is the "left" set, b the "right" set on this axis
                    ga = -nrm
                    gb = nrm
                else:
                    ga = nrm
                    gb = -nrm
                best_ga = ga
                best_gb = gb
    if best_ga is None:
        return best, np.zeros(2), np.zeros(2)
    # Each vertex of a hexagon depends on exactly one center, so the center
    # gradient equals the axis normal (with sign) for all 6 vertices summed
    # over the min/max selection. For a regular hexagon projected on its own
    # edge normal, min is one vertex, max is the opposite; moving the center
    # by nrm*1 moves both by the same amount, so the gap gradient w.r.t. the
    # center is exactly nrm (or -nrm).
    return best, best_ga, best_gb


def _max_radius_and_grad(centers, verts_list):
    """Max distance from origin to any vertex, plus gradient w.r.t. centers."""
    maxr = 0.0
    maxv = None
    for v in verts_list:
        r = np.linalg.norm(v)
        if r > maxr:
            maxr = r
            maxv = v
    grad = np.zeros((12, 2))
    if maxv is not None:
        # find which hexagon owns maxv
        for k, v in enumerate(verts_list):
            if np.array_equal(v, maxv):
                grad[k] = maxv / maxr if maxr > 1e-15 else np.zeros(2)
                break
    return maxr, grad


def hexagon_packing_12():
    t0 = time.time()
    deadline = t0 + 600.0

    # Seed layout: center + ring of 6 + ring of 5 (pointy-top, angle 0)
    centers = np.array([
        [0.0, 0.0],
        [2.5, 0.0],
        [1.25, 2.1650635094610964],
        [-1.25, 2.1650635094610964],
        [-2.5, 0.0],
        [-1.25, -2.1650635094610964],
        [1.25, -2.1650635094610964],
        [5.0, 0.0],
        [3.75, 3.247595264191645],
        [0.0, 4.3301270189221935],
        [-3.75, 3.247595264191645],
        [-5.0, 0.0],
    ], dtype=float)

    def compute_objective(c):
        verts_list = [_hex_verts(c[k, 0], c[k, 1], 1.0, 0.0) for k in range(12)]
        # center the cluster at origin
        centroid = c.mean(axis=0)
        verts_list = [v - centroid for v in verts_list]
        maxr, grad = _max_radius_and_grad(c - centroid, verts_list)
        # containment: maxr <= s, but s = maxr for a pointy-top hexagon, so
        # containment is automatically satisfied. We minimize maxr.
        return maxr, grad

    def separation_penalty(c, mu):
        verts_list = [_hex_verts(c[k, 0], c[k, 1], 1.0, 0.0) for k in range(12)]
        penalty = 0.0
        grad = np.zeros((12, 2))
        for i in range(12):
            for j in range(i + 1, 12):
                gap = _separation(verts_list[i], verts_list[j])
                if gap < 0:
                    penalty += mu * gap * gap
                    g, ga, gb = _separation_grads(verts_list[i], verts_list[j], c[i], c[j])
                    grad[i] += mu * 2.0 * gap * ga
                    grad[j] += mu * 2.0 * gap * gb
        return penalty, grad

    # Projected gradient descent
    c = centers.copy()
    best_c = c.copy()
    best_s = np.inf

    def evaluate(c):
        verts_list = [_hex_verts(c[k, 0], c[k, 1], 1.0, 0.0) for k in range(12)]
        centroid = c.mean(axis=0)
        verts_list = [v - centroid for v in verts_list]
        s = 0.0
        for v in verts_list:
            r = np.linalg.norm(v)
            if r > s:
                s = r
        return s

    best_s = evaluate(c)

    mu = 50.0
    lr = 0.05
    for it in range(20000):
        if time.time() > deadline - 30.0:
            break
        # objective gradient
        maxr, g_obj = compute_objective(c)
        # penalty gradient
        pen, g_pen = separation_penalty(c, mu)
        grad = g_obj + g_pen
        # recenter to keep centroid at origin (reduce effective dimension)
        grad -= grad.mean(axis=0, keepdims=True)
        c = c - lr * grad
        s = evaluate(c)
        if s < best_s:
            best_s = s
            best_c = c.copy()
        # adapt lr
        if it % 500 == 0:
            lr *= 0.95
            mu *= 1.02

    c = best_c

    # Build return arrays
    inner = np.zeros((12, 3))
    for k in range(12):
        inner[k, 0] = c[k, 0]
        inner[k, 1] = c[k, 1]
        inner[k, 2] = 0.0

    centroid = c.mean(axis=0)
    outer = np.array([centroid[0], centroid[1], 0.0])
    s = best_s

    # Ensure s >= sqrt(12)
    if s < np.sqrt(12.0):
        s = np.sqrt(12.0)

    return inner, outer, s


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = 630.0
_MARGIN = 3.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(th), cy + d * np.sin(th)))


def _hexagon_packing_12():
    n = 12
    # 12 hexagons in a "flower" of radius 1 around a center, all axis-aligned.
    base = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, np.sqrt(3.0)],
        [-1.0, np.sqrt(3.0)],
        [-2.0, 0.0],
        [-1.0, -np.sqrt(3.0)],
        [1.0, -np.sqrt(3.0)],
        [3.0, np.sqrt(3.0)],
        [3.0, -np.sqrt(3.0)],
        [0.0, 2 * np.sqrt(3.0)],
        [0.0, -2 * np.sqrt(3.0)],
        [-3.0, np.sqrt(3.0)],
    ])
    # Parameters: 24 positions (x,y) + 12 angles (deg) + 1 outer side s = 37 vars
    x0 = np.concatenate([base.ravel(), np.zeros(n), [3.0]])

    def unpack(p):
        xy = p[:2 * n].reshape(n, 2)
        ang = p[2 * n:3 * n]
        s = p[3 * n]
        return xy, ang, s

    def score(p):
        xy, ang, s = unpack(p)
        # outer hexagon centered at origin, angle 0, side s
        outer = _hex_vertices(0.0, 0.0, s, 0.0)
        # outer edge normals (inward): edge from v[i] to v[i+1], normal = rotate(edge, +90)
        edges = np.roll(outer, -1, axis=0) - outer
        normals = np.column_stack((-edges[:, 1], edges[:, 0]))
        # ensure normals point inward (toward origin): dot(normal, -outer[i]) > 0
        for i in range(6):
            if np.dot(normals[i], -outer[i]) < 0:
                normals[i] = -normals[i]
        # containment: for each inner vertex, min over outer edges of dot(normal, v - outer[i]) >= 0
        penalty = 0.0
        for k in range(n):
            verts = _hex_vertices(xy[k, 0], xy[k, 1], 1.0, ang[k])
            # dot products: (6,6)
            d = normals @ (verts - outer).T  # shape (6,6)
            m = d.min()
            if m < 0:
                penalty += m * m
        # separation: for each pair, SAT on the 12 edge normals
        for i in range(n):
            vi = _hex_vertices(xy[i, 0], xy[i, 1], 1.0, ang[i])
            ei = np.roll(vi, -1, axis=0) - vi
            ni = np.column_stack((-ei[:, 1], ei[:, 0]))
            for j in range(i + 1, n):
                vj = _hex_vertices(xy[j, 0], xy[j, 1], 1.0, ang[j])
                ej = np.roll(vj, -1, axis=0) - vj
                nj = np.column_stack((-ej[:, 1], ej[:, 0]))
                alln = np.vstack([ni, nj])
                # project both hexagons onto each normal
                # for normal n: min/max of dot(n, verts)
                pi = alln @ vi.T  # (12,6)
                pj = alln @ vj.T
                min_i = pi.min(axis=1)
                max_i = pi.max(axis=1)
                min_j = pj.min(axis=1)
                max_j = pj.max(axis=1)
                # gap = min(max_i, max_j) - max(min_i, min_j); if < 0 overlap
                gap = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
                gmin = gap.min()
                if gmin < 0:
                    penalty += gmin * gmin
        return s + 50.0 * penalty

    # Nelder-Mead with deadline guard
    from scipy.optimize import minimize

    best = x0.copy()
    best_val = score(x0)
    n_iter = 0
    try:
        res = minimize(
            score, x0, method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9, "adaptive": True},
        )
        n_iter = res.nit
        if score(res.x) < best_val:
            best = res.x
            best_val = score(res.x)
    except Exception:
        pass

    xy, ang, s = unpack(best)
    inner = np.column_stack([xy, ang])
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(s)


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger
    regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3) rows (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    inner, outer, s = _hexagon_packing_12()
    return inner, outer, s


# EVOLVE-BLOCK-END

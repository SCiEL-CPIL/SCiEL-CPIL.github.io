# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])

def _pair_overlap(v1, v2):
    # v1, v2: (6,2). True if intersect (gap <= 0) on any of 12 edge normals.
    n1 = 12
    n2 = 12
    overlap = np.zeros((n1, n2))
    for i in range(6):
        a = v1[i]; b = v1[(i + 1) % 6]
        e = b - a
        n = np.array([-e[1], e[0]])
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        n = n / ln
        p1 = v1 @ n
        p2 = v2 @ n
        overlap[i, :] = np.maximum(p1.min() - p2.max(), p2.min() - p1.max())
        overlap[i + 6, :] = np.maximum(p2.min() - p1.max(), p1.min() - p2.max())
    # gap on each normal = min over the two orientations
    gap = np.min(overlap, axis=0)
    return gap.min()

def _required_side(inner_shifted, a_deg):
    # inner_shifted: (11,2) vertices centers already shifted; compute vertices
    verts = np.array([_hex_vertices(x, y, 1.0, 0.0) for x, y in inner_shifted])
    # outer hexagon centered at origin, side s, angle a
    # containment: each inner vertex inside outer hexagon
    # outer edge normals point inward; distance from center along inward normal must be <= s*cos(30)
    a = np.deg2rad(a_deg)
    # outward normals of outer hexagon edges
    th = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
    # outward normal for edge i
    normals = np.column_stack([np.cos(th), np.sin(th)])
    # For a point p, it's inside iff for all outward normals n: dot(p, n) <= s*cos(30)
    # So s >= max over all inner vertices, all normals of dot(p,n) / cos(30)
    cos30 = np.cos(np.pi / 6.0)
    allv = verts.reshape(-1, 2)
    dots = allv @ normals.T  # (66, 6)
    return (dots.max() / cos30)

def hexagon_packing_11():
    t0 = time.time()
    DEADLINE = 600.0

    # Honeycomb arrangement: center + 6 ring + 4 more in a compact pattern
    # Use axial coordinates for hex grid. Ring of 6 around center, plus 4 additional.
    # A compact 11-hexagon shape: center, 6 neighbors, and 4 of the 12 second-ring.
    # We'll build a set of unit hexagons in a honeycomb and optimize orientation.

    # Honeycomb neighbor offsets for a hex grid (pointy/flat). Use flat-top unit hexagons.
    # For flat-top hexagons of side 1, center spacing: horizontal 1.5, vertical sqrt(3).
    # Axial coords (q, r) -> x = 1.5*q, y = sqrt(3)*(r + q/2)
    sqrt3 = np.sqrt(3.0)
    axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, -2), (1, -2), (0, -2),
    ]
    centers = np.array([[1.5 * q, sqrt3 * (r + q / 2.0)] for q, r in axial])

    def objective(params):
        # params: [a_deg, s]
        a_deg, s = params
        # rotate centers
        a = np.deg2rad(a_deg)
        c = np.cos(a); sn = np.sin(a)
        R = np.array([[c, -sn], [sn, c]])
        shifted = centers @ R.T  # (11,2)
        # overlap penalty
        verts = np.array([_hex_vertices(x, y, 1.0, 0.0) for x, y in shifted])
        pen = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                g = _pair_overlap(verts[i], verts[j])
                if g < 1e-6:
                    pen += (1e-6 - g) ** 2
        # containment: required side vs s
        req = _required_side(shifted, a_deg)
        if req > s:
            pen += (req - s) ** 2
        # objective: minimize s + penalty
        return s + 100.0 * pen

    best = None
    best_val = np.inf

    # Multi-start over outer hexagon orientation
    for a0 in np.linspace(0.0, 60.0, 13, endpoint=False):
        if time.time() > DEADLINE - 60:
            break
        # initial guess for s: required side at this orientation
        a = np.deg2rad(a0)
        c = np.cos(a); sn = np.sin(a)
        R = np.array([[c, -sn], [sn, c]])
        shifted = centers @ R.T
        s0 = _required_side(shifted, a0) + 0.1
        try:
            res = minimize(objective, np.array([a0, s0]), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8,
                                    'adaptive': True})
            a_opt, s_opt = res.x
            # normalize orientation into [0,60)
            a_opt = a_opt % 60.0
            if s_opt < best_val:
                best_val = s_opt
                best = (a_opt, s_opt)
        except Exception:
            pass

    if best is None:
        # fallback: seed-like
        inner = np.array([[0,0,0],[-2.5,0,0],[2.5,0,0],[-1.25,2.17,0],[1.25,2.17,0],
                          [-1.25,-2.17,0],[1.25,-2.17,0],[-3.75,2.17,0],[3.75,2.17,0],
                          [-3.75,-2.17,0],[3.75,-2.17,0]])
        return inner, np.array([0,0,0]), 8.0

    a_opt, s_opt = best
    a = np.deg2rad(a_opt)
    c = np.cos(a); sn = np.sin(a)
    R = np.array([[c, -sn], [sn, c]])
    shifted = centers @ R.T

    # fine-tune s to just the required value
    s_opt = _required_side(shifted, a_opt)

    inner = np.column_stack([shifted, np.zeros(11)])
    inner[:, 2] = 0.0
    outer = np.array([0.0, 0.0, a_opt])
    return inner, outer, float(s_opt)

# EVOLVE-BLOCK-END

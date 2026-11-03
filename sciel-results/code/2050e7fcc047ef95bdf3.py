"""Hexagon packing: place N unit hexagons in minimal outer hexagon."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# ---- Geometry helpers ----
def _hex_vertices(center, side, angle_deg):
    """6 vertices of a regular hexagon (flat-top when angle=0)."""
    cx, cy = center
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    base = np.array([[side * np.cos(np.radians(60 * i)),
                      side * np.sin(np.radians(60 * i))] for i in range(6)])
    rot = np.array([[ca, -sa], [sa, ca]])
    return base @ rot.T + np.array([cx, cy])

def _poly_overlap_area(v1, v2):
    """Sutherland-Hodgman polygon intersection area."""
    def clip(subject, clip_edge):
        if not subject:
            return []
        A, B = clip_edge
        out = []
        def inside(p):
            return np.cross(B - A, p - A) >= -1e-12
        for i in range(len(subject)):
            S = np.array(subject[i])
            E = np.array(subject[(i + 1) % len(subject)])
            s_in = inside(S)
            e_in = inside(E)
            if s_in:
                out.append(S)
                if not e_in:
                    d1 = np.cross(B - A, S - A)
                    d2 = np.cross(B - A, E - A)
                    out.append(S + (E - S) * d1 / (d1 - d2))
            elif e_in:
                d1 = np.cross(B - A, S - A)
                d2 = np.cross(B - A, E - A)
                out.append(S + (E - S) * d1 / (d1 - d2))
        return out

    subj = [np.array(p) for p in v1]
    for i in range(6):
        subj = clip(subj, [v2[i], v2[(i + 1) % 6]])
    if len(subj) < 3:
        return 0.0
    a = 0.0
    for i in range(len(subj)):
        x1, y1 = subj[i]
        x2, y2 = subj[(i + 1) % len(subj)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0

def _soft_overlap(v1, v2, k=12.0):
    """Smooth surrogate: 0 when disjoint, ~overlap area when intersecting."""
    area = _poly_overlap_area(v1, v2)
    if area <= 1e-9:
        return 0.0
    return 1.0 - np.exp(-k * area)

def _hex_in_hex(v_in, v_out):
    """1 if hex_in is fully inside hex_out (all vertices inside), else soft penalty."""
    # Point-in-convex-polygon via cross products
    n = len(v_out)
    for i in range(n):
        A = v_out[i]
        B = v_out[(i + 1) % n]
        for p in v_in:
            c = np.cross(B - A, p - A)
            if c < 0:
                return c  # negative => outside
    return 0.0

def _outer_side(points, center, angle_deg):
    """Minimum side of regular hexagon (center, rotation) containing all points."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    rot = np.array([[ca, -sa], [sa, ca]])
    rel = (np.array(points) - np.array(center)) @ rot.T
    # 6 support directions (normals) of a flat-top hexagon
    dirs = np.array([[np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)])
    proj = rel @ dirs.T  # (n, 6)
    widths = proj.max(axis=0) - proj.min(axis=0)
    # side = max(width) / sqrt(3)
    return widths.max() / np.sqrt(3.0)

# ---- Objective ----
def _objective(x, angle_deg, center):
    """x = flattened positions (2N). Returns scalar cost."""
    pos = x.reshape(N, 2)
    vs = [_hex_vertices(pos[i], 1.0, 0.0) for i in range(N)]
    cost = 0.0
    # pairwise soft overlaps
    for i in range(N):
        for j in range(i + 1, N):
            cost += _soft_overlap(vs[i], vs[j])
    # containment in outer hexagon
    v_out = _hex_vertices(center, 1.0, angle_deg)  # unit outer, will scale
    # scale outer to contain: compute required scale
    # Actually compute side directly and add containment penalty via support
    # Use support-function containment: each inner vertex must be within outer
    # We'll just use _outer_side as the objective and rely on overlap penalty for disjointness
    side = _outer_side(pos, center, angle_deg)
    return side + 10.0 * cost

# ---- Construct ----
def construct():
    t0 = time.time()
    DEADLINE = 135.0  # safety margin under 150s

    # Structured hexagonal lattice starting positions (centered)
    # Use a hexagonal arrangement of N=11 hexagons
    # Hex grid: rings. For 11 = 1 + 6 + 4? No, 1+6=7, 1+6+12=19.
    # 11 doesn't form a perfect hexagon. Use a compact cluster.
    # Place on hex lattice with spacing ~2 (touching).
    # Hex lattice basis for flat-top:
    # neighbor offsets for flat-top hex (side 1):
    #   E: (2, 0), NE: (1, sqrt(3)), SE: (1, -sqrt(3)), W: (-2,0), NW: (-1, sqrt(3)), SW: (-1,-sqrt(3))
    sqrt3 = np.sqrt(3.0)
    # Build a small cluster of 11 hexagons on the lattice
    # Use axial coords: center + 6 neighbors + 4 more
    # Axial to pixel for flat-top:
    def axial_to_xy(q, r):
        x = 2.0 * q + r
        y = sqrt3 * r
        return x, y

    # 11 hexagons: center, 6 ring, then 4 from second ring (to keep compact)
    axial = [(0, 0)]
    # ring 1 (6)
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    axial += ring1
    # ring 2 partial (4 more to reach 11)
    ring2 = [(2, 0), (2, -1), (1, -2), (0, -2)]
    axial += ring2[:4]
    axial = axial[:N]
    pos0 = np.array([axial_to_xy(q, r) for q, r in axial])
    # Center the cluster
    pos0 = pos0 - pos0.mean(axis=0)

    center = (0.0, 0.0)
    angle_deg = 0.0

    best_x = pos0.flatten().copy()
    best_cost = _objective(best_x, angle_deg, center)

    # Refine with L-BFGS-B
    x0 = pos0.flatten().copy()
    try:
        res = minimize(_objective, x0, args=(angle_deg, center),
                       method='L-BFGS-B',
                       options={'maxiter': 200, 'maxfun': 4000, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < best_cost:
            best_cost = res.fun
            best_x = res.x
    except Exception:
        pass

    if time.time() > DEADLINE:
        pos = best_x.reshape(N, 2)
        side = _outer_side(pos, center, angle_deg)
        inner = [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in range(N)]
        return inner, center, float(side), angle_deg

    # Try a few outer rotations and re-optimize positions for each
    for ang in [30.0, 0.0, 15.0, 45.0]:
        if time.time() > DEADLINE:
            break
        try:
            res = minimize(_objective, x0, args=(ang, center),
                           method='L-BFGS-B',
                           options={'maxiter': 150, 'maxfun': 3000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x
                angle_deg = ang
        except Exception:
            pass

    pos = best_x.reshape(N, 2)
    side = _outer_side(pos, center, angle_deg)
    inner = [[float(pos[i, 0]), float(pos[i, 1]), 0.0] for i in range(N)]
    return inner, center, float(side), angle_deg

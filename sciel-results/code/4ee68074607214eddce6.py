"""Combined program: lattice-topology search + support-function objective + SAT-constrained refinement."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)

# Unit hexagon vertices (circumradius = 1 = side length)
_UNIT_VERTS = np.array([
    [1.0, 0.0],
    [0.5, SQRT3 / 2.0],
    [-0.5, SQRT3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -SQRT3 / 2.0],
    [0.5, -SQRT3 / 2.0],
])

# Outer hexagon normals (inradius direction) for angle 0: at 30 + k*60 degrees
_OUTER_NORMAL_ANGLES = np.array([math.radians(30.0 + 60.0 * k) for k in range(6)])
_OUTER_NORMALS = np.column_stack([np.cos(_OUTER_NORMAL_ANGLES), np.sin(_OUTER_NORMAL_ANGLES)])


def _support_unit(nvec):
    """Max dot product of unit hexagon vertices with direction nvec (2-vector)."""
    return float(np.max(_UNIT_VERTS @ nvec))


def _outer_side_support(positions, alpha_deg, beta_deg):
    """
    Fast support-function objective: minimal outer hexagon side length to contain
    N unit hexagons at `positions`, all rotated by alpha_deg, inside outer hexagon
    rotated by beta_deg (centered at origin).
    positions: (N,2) array.
    """
    a = math.radians(alpha_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Rotation matrix R(-alpha)
    R = np.array([[ca, sa], [-sa, ca]])  # R(-a) = [[cos a, sin a],[-sin a, cos a]]

    # For each outer normal n, rotate by -alpha: n' = R(-alpha) n
    n_rot = _OUTER_NORMALS @ R.T  # (6,2)
    h = np.array([_support_unit(n) for n in n_rot])  # (6,)

    # positions @ n.T -> (N,6); add h -> (N,6); max over N -> (6,)
    proj = positions @ _OUTER_NORMALS.T + h[None, :]
    max_proj = proj.max(axis=0)
    return float(2.0 * np.max(max_proj) / SQRT3)


def _get_positions(topo, scale=2.0):
    """Convert axial coords to centered cartesian positions at given spacing scale."""
    pts = []
    for q, r in topo:
        x = scale * (q + r / 2.0)
        y = scale * (r * SQRT3 / 2.0)
        pts.append((x, y))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return np.array([(x - cx, y - cy) for x, y in pts])


def _hex_vertices(cx, cy, angle_deg):
    a = math.radians(angle_deg)
    angles = a + np.arange(6) * (math.pi / 3.0)
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _separation(v1, v2):
    """
    Continuous separation measure between two convex hexagons via SAT.
    Returns >0 if disjoint, <0 if overlapping (approx penetration), 0 if touching.
    """
    min_sep = float('inf')
    for verts in (v1, v2):
        for i in range(6):
            edge = verts[(i + 1) % 6] - verts[i]
            nrm = np.array([-edge[1], edge[0]])
            nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
            p1 = v1 @ nrm
            p2 = v2 @ nrm
            min1, max1 = p1.min(), p1.max()
            min2, max2 = p2.min(), p2.max()
            if max1 < min2:
                sep = min2 - max1
            elif max2 < min1:
                sep = min1 - max2
            else:
                overlap = min(max1 - min2, max2 - min1)
                sep = -overlap
            if sep < min_sep:
                min_sep = sep
    return min_sep


def _all_separations(verts_list):
    """Return list of separations for all pairs (i<j)."""
    seps = []
    for i in range(N):
        for j in range(i + 1, N):
            seps.append(_separation(verts_list[i], verts_list[j]))
    return seps


def _make_topologies():
    topologies = []
    # Parallelogram grids
    for cols, rows in [(3, 4), (4, 3), (2, 6), (6, 2), (1, 12), (12, 1), (2, 3, 2), (3, 2, 2)]:
        pass
    # Simple rectangular-ish parallelograms
    topologies.append([(q, r) for q in range(3) for r in range(4)])
    topologies.append([(q, r) for q in range(4) for r in range(3)])
    topologies.append([(q, r) for q in range(2) for r in range(6)])
    topologies.append([(q, r) for q in range(6) for r in range(2)])
    # 3x2x2 block (two rows of 3, two rows of 3 offset) -> 12
    topologies.append([(q, r) for q in range(3) for r in range(2)] +
                      [(q, r + 2) for q in range(3) for r in range(2)])
    # Compact hexagonal: center + ring1 (6) + 5 from ring2
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    # ring2 candidates
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            d = (abs(q) + abs(r) + abs(q + r)) // 2
            if d == 2:
                ring2.append((q, r))
    base = [(0, 0)] + ring1
    # Select 5 evenly spaced from ring2
    def ang(p):
        x = p[0] + p[1] / 2.0
        y = p[1] * SQRT3 / 2.0
        return math.atan2(y, x)
    ring2.sort(key=ang)
    picks = [ring2[i] for i in range(0, len(ring2), 2)][:5]
    while len(picks) < 5:
        picks.append(ring2[len(picks)])
    topologies.append(base + picks)
    # Cross shape
    cross = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (2, -1),
    ]
    if len(set(cross)) == 12:
        topologies.append(cross)
    # Another ring2 selection
    picks2 = ring2[:5]
    topologies.append(base + picks2)
    return topologies


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    topologies = _make_topologies()

    best_side = float('inf')
    best_positions = None
    best_alpha = 0.0
    best_beta = 0.0

    # ---- Phase 1: topology enumeration + grid search over (alpha, beta) ----
    for topo in topologies:
        if time.time() > deadline - 25.0:
            break
        positions = _get_positions(topo, scale=2.0)
        best_local = float('inf')
        best_local_a = 0.0
        best_local_b = 0.0
        # Grid over alpha in [0,60), beta in [0,60), step 1.0 deg
        for a in np.arange(0.0, 60.0, 1.0):
            if time.time() > deadline - 25.0:
                break
            for b in np.arange(0.0, 60.0, 1.0):
                s = _outer_side_support(positions, a, b)
                if s < best_local:
                    best_local = s
                    best_local_a = a
                    best_local_b = b
        if best_local < best_side:
            best_side = best_local
            best_positions = positions
            best_alpha = best_local_a
            best_beta = best_local_b

    if best_positions is None:
        # Absolute fallback
        positions = _get_positions([(q, r) for q in range(3) for r in range(4)], scale=2.0)
        best_positions = positions
        best_alpha, best_beta = 0.0, 0.0
        best_side = _outer_side_support(positions, 0.0, 0.0)

    # ---- Phase 2: Nelder-Mead refinement of (alpha, beta) for best topology ----
    if time.time() < deadline - 30.0:
        pos = best_positions

        def obj_ab(params):
            return _outer_side_support(pos, params[0], params[1])

        try:
            res = minimize(obj_ab, [best_alpha, best_beta], method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_alpha = res.x[0]
                best_beta = res.x[1]
        except Exception:
            pass

    # ---- Phase 3: constrained SLSQP polish with true SAT non-overlap ----
    # Optimize (alpha, beta, tx, ty) with hard constraint that all pairs are disjoint.
    if time.time() < deadline - 15.0:
        base_pos = best_positions

        def build(pos, alpha_deg, beta_deg, tx, ty):
            # base_pos already centered; apply small translation
            return base_pos + np.array([tx, ty]), alpha_deg, beta_deg

        def objective(x):
            if time.time() > deadline - 5.0:
                return 1e9
            alpha, beta, tx, ty = x
            pos = base_pos + np.array([tx, ty])
            return _outer_side_support(pos, alpha, beta)

        def constraints(x):
            if time.time() > deadline - 5.0:
                return np.zeros(N * (N - 1) // 2)
            alpha, beta, tx, ty = x
            pos = base_pos + np.array([tx, ty])
            verts = [_hex_vertices(pos[i, 0], pos[i, 1], alpha) for i in range(N)]
            return np.array(_all_separations(verts))

        x0 = np.array([best_alpha, best_beta, 0.0, 0.0])
        bounds = [(0.0, 60.0), (0.0, 60.0), (-0.5, 0.5), (-0.5, 0.5)]
        cons = [{'type': 'ineq', 'fun': constraints}]
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 150, 'ftol': 1e-8, 'disp': False})
            if res.success or res.fun < best_side:
                alpha, beta, tx, ty = res.x
                pos = base_pos + np.array([tx, ty])
                verts = [_hex_vertices(pos[i, 0], pos[i, 1], alpha) for i in range(N)]
                seps = _all_separations(verts)
                if min(seps) >= -1e-5:
                    s = _outer_side_support(pos, alpha, beta)
                    if s < best_side:
                        best_side = s
                        best_positions = pos
                        best_alpha = alpha
                        best_beta = beta
        except Exception:
            pass

    # ---- Finalize: recenter and verify ----
    positions = best_positions
    # Center the configuration so outer hexagon is at origin
    cx = float(np.mean(positions[:, 0]))
    cy = float(np.mean(positions[:, 1]))
    positions = positions - np.array([cx, cy])

    # Recompute side with final parameters
    final_side = _outer_side_support(positions, best_alpha, best_beta)

    # Verify non-overlap with true SAT
    verts = [_hex_vertices(positions[i, 0], positions[i, 1], best_alpha) for i in range(N)]
    seps = _all_separations(verts)
    if min(seps) < -1e-4:
        # Fallback: spread out to guarantee validity
        positions = np.array([[ (i % 4) * 2.6 + (i // 4 % 2) * 1.3, (i // 4) * 2.4 ] for i in range(N)])
        cx = float(np.mean(positions[:, 0]))
        cy = float(np.mean(positions[:, 1]))
        positions = positions - np.array([cx, cy])
        best_alpha = 0.0
        best_beta = 0.0
        verts = [_hex_vertices(positions[i, 0], positions[i, 1], best_alpha) for i in range(N)]
        final_side = _outer_side_support(positions, best_alpha, best_beta)

    inner = [[float(positions[i, 0]), float(positions[i, 1]), float(best_alpha)] for i in range(N)]
    return inner, (0.0, 0.0), float(final_side), float(best_beta)

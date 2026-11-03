import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0
SQRT3 = math.sqrt(3)
TWO_OVER_SQRT3 = 2.0 / SQRT3

_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

# Minkowski sum of unit hexagon (angle 0) and negative of unit hexagon (angle a)
# gives the set of center-offsets where the two hexagons just touch.
# Compute as convex hull of {v1 - v2 : v1 in H1, v2 in H2}.
def _minkowski_sum(a):
    c, s = math.cos(a), math.sin(a)
    H2 = np.column_stack((_UNIT_HEX_VERTS[:, 0] * c - _UNIT_HEX_VERTS[:, 1] * s,
                          _UNIT_HEX_VERTS[:, 0] * s + _UNIT_HEX_VERTS[:, 1] * c))
    pts = []
    for v1 in _UNIT_HEX_VERTS:
        for v2 in H2:
            pts.append(v1 - v2)
    pts = np.array(pts)
    # convex hull via scipy
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts)
    return pts[hull.vertices]

# Precompute Minkowski polygons for a grid of relative angles (period pi/3, symmetric).
# We'll interpolate r(delta, dir) for arbitrary delta.
_MINK_GRID = 360  # steps over [0, pi/3)
_mink_polys = {}
def _get_mink(a):
    # reduce a to [0, pi/3)
    a = a % (math.pi / 3.0)
    key = round(a * _MINK_GRID / (math.pi/3.0))
    if key not in _mink_polys:
        _mink_polys[key] = _minkowski_sum(a)
    return _mink_polys[key]

def _mink_radius(poly, dx, dy):
    # distance from origin to boundary of convex polygon containing origin, in direction (dx,dy)
    # Use edge-based: for each edge (p_i, p_{i+1}), find if ray from origin in dir (dx,dy)
    # intersects the edge; the closest positive intersection is the boundary.
    n = len(poly)
    best = float('inf')
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i+1) % n]
        # line through p1, p2: find t>0 such that t*(dx,dy) = p1 + s*(p2-p1), s in [0,1]
        ex = p2[0] - p1[0]
        ey = p2[1] - p1[1]
        # solve t*dx = p1x + s*ex ; t*dy = p1y + s*ey
        # cross multiply: t*dx*ey - t*dy*ex = p1x*ey - p1y*ex
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-12:
            continue
        t = (p1[0] * ey - p1[1] * ex) / denom
        s = (dx * p1[1] - dy * p1[0]) / denom  # from t*dx = p1x + s*ex -> s = (t*dx - p1x)/ex
        # better: s = (t*dx - p1[0]) / ex if ex != 0 else (t*dy - p1[1])/ey
        if abs(ex) > 1e-12:
            s = (t * dx - p1[0]) / ex
        else:
            s = (t * dy - p1[1]) / ey
        if t > 1e-9 and -1e-9 <= s <= 1.0 + 1e-9:
            if t < best:
                best = t
    if best == float('inf'):
        # fallback: min distance to vertices
        best = min(math.hypot(p[0], p[1]) for p in poly)
    return best

def _collision_radius(delta, dx, dy):
    # delta = relative angle (rad), (dx,dy) = normalized direction of center offset
    poly = _get_mink(delta)
    return _mink_radius(poly, dx, dy)

def construct():
    t0 = time.time()

    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _UNIT_HEX_VERTS[None, :, 0] * c[:, None] - _UNIT_HEX_VERTS[None, :, 1] * s[:, None]
        ry = _UNIT_HEX_VERTS[None, :, 0] * s[:, None] + _UNIT_HEX_VERTS[None, :, 1] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = TWO_OVER_SQRT3 * max_projs
            return np.max(side_vals)

        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar_local(eval_phi, lo, hi)
            if res < best_side:
                best_side = res
        except Exception:
            pass
        return best_side, best_phi, center[0], center[1]

    def minimize_scalar_local(f, lo, hi):
        # golden-section
        phi = (math.sqrt(5) - 1) / 2
        a, b = lo, hi
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(80):
            if fc < fd:
                b = d; d = c; fd = fc
                c = b - phi * (b - a)
                fc = f(c)
            else:
                a = c; c = d; fc = fd
                d = a + phi * (b - a)
                fd = f(d)
            if abs(b - a) < 1e-9:
                break
        return min(f((a + b) / 2), fc, fd)

    # Smooth collision objective: for each pair, compute required separation r(delta)
    # and penalize violations smoothly.
    def smooth_collision_penalty(positions, angles_rad):
        # returns a scalar penalty (0 if all disjoint, >0 if any overlap)
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = math.hypot(dx, dy)
                if dist < 1e-9:
                    pen += 10.0
                    continue
                delta = angles_rad[i] - angles_rad[j]
                r = _collision_radius(delta, dx / dist, dy / dist)
                # if dist < r, overlap. Smooth penalty: max(0, r - dist)^2
                if dist < r:
                    pen += (r - dist) ** 2
        return pen

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = math.hypot(dx, dy)
                if dist < 1e-9:
                    return False
                delta = angles_rad[i] - angles_rad[j]
                r = _collision_radius(delta, dx / dist, dy / dist)
                if dist < r - 1e-7:
                    return False
        return True

    def objective(params):
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3 * i]
            positions[i, 1] = params[3 * i + 1]
            angles_deg[i] = params[3 * i + 2]
        angles_rad = np.radians(angles_deg)
        pen = smooth_collision_penalty(positions, angles_rad)
        if pen > 1e-6:
            return 1e4 + pen
        side, _, _, _ = compute_min_outer_side(positions, angles_deg)
        return side

    # Honeycomb initial configuration
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0

    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Evaluate base
    s, ph, cx, cy = compute_min_outer_side(base_positions, base_angles)
    if s < best_side:
        best_side = s
        best_phi = ph
        best_cx = cx
        best_cy = cy

    def init_params_from(pos, ang):
        p = np.zeros(3 * N)
        for i in range(N):
            p[3 * i] = pos[i, 0]
            p[3 * i + 1] = pos[i, 1]
            p[3 * i + 2] = ang[i]
        return p

    def extract(pos, ang, params):
        for i in range(N):
            pos[i, 0] = params[3 * i]
            pos[i, 1] = params[3 * i + 1]
            ang[i] = params[3 * i + 2]

    # Multi-start L-BFGS-B
    random.seed(42)
    np.random.seed(42)

    starts = []
    # Start 1: honeycomb
    starts.append(init_params_from(base_positions, base_angles))

    # Generate random starts: jittered honeycomb + fully random
    for k in range(6):
        pos = base_positions.copy()
        ang = base_angles.copy()
        for i in range(N):
            pos[i, 0] += random.gauss(0, 0.5)
            pos[i, 1] += random.gauss(0, 0.5)
            ang[i] += random.gauss(0, 10.0)
        starts.append(init_params_from(pos, ang))

    # Fully random starts
    for k in range(4):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = random.uniform(-4, 4)
            pos[i, 1] = random.uniform(-4, 4)
            ang[i] = random.uniform(-30, 30)
        starts.append(init_params_from(pos, ang))

    time_per_start = (DEADLINE - 20.0) / len(starts)

    for start_params in starts:
        if time.time() - t0 > DEADLINE - 25.0:
            break
        t_start = time.time()
        try:
            res = minimize(objective, start_params, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-8,
                                    'maxfun': 20000, 'disp': False})
        except Exception:
            continue

        if time.time() - t_start > time_per_start + 5.0:
            continue

        new_pos = np.zeros((N, 2))
        new_ang = np.zeros(N)
        extract(new_pos, new_ang, res.x)

        if not is_valid_configuration(new_pos, new_ang):
            continue

        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
        if new_side < best_side:
            best_side = new_side
            best_positions = new_pos.copy()
            best_angles = new_ang.copy()
            best_phi = new_phi
            best_cx = new_cx
            best_cy = new_cy

    # Polish: local L-BFGS-B with tighter tolerance from best
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5.0:
        try:
            res = minimize(objective, init_params_from(best_positions, best_angles),
                           method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxfun': 50000, 'disp': False})
            new_pos = np.zeros((N, 2))
            new_ang = np.zeros(N)
            extract(new_pos, new_ang, res.x)
            if is_valid_configuration(new_pos, new_ang):
                new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_pos.copy()
                    best_angles = new_ang.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
        except Exception:
            pass

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle

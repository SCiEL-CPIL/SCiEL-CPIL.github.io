"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
TIME_LIMIT = 140.0  # safety margin below 150s

# Pre-compute base hexagon vertices (side=1, angle=0)
_BASE_ANGLES = np.deg2rad(np.arange(0, 360, 60))
_BASE_VERTS = np.column_stack((np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)))

def _hex_vertices(cx, cy, angle_deg):
    """Returns (6, 2) array of vertices for a hexagon."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return (R @ _BASE_VERTS.T).T + np.array([cx, cy])

def _min_overlap_depth(v1, v2):
    """
    Compute minimum overlap depth between two convex hexagons using SAT.
    Returns 0 if no overlap, positive value if overlapping.
    """
    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack((-edges[:, 1], edges[:, 0]))
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        return normals / lengths

    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack((n1, n2))

    proj1 = v1 @ all_normals.T
    proj2 = v2 @ all_normals.T

    min1 = np.min(proj1, axis=1)
    max1 = np.max(proj1, axis=1)
    min2 = np.min(proj2, axis=1)
    max2 = np.max(proj2, axis=1)

    overlaps = np.minimum(max1 - min2, max2 - min1)
    min_overlap = np.min(overlaps)

    if min_overlap <= 0:
        return 0.0
    return min_overlap

def _smooth_collision_penalty(cfg, weight=10.0):
    """
    Compute smooth collision penalty for all pairs.
    Uses a smooth approximation of overlap depth.
    """
    penalty = 0.0
    verts_list = [_hex_vertices(cfg[i][0], cfg[i][1], cfg[i][2]) for i in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            dx = cfg[i][0] - cfg[j][0]
            dy = cfg[i][1] - cfg[j][1]
            dist_sq = dx*dx + dy*dy
            if dist_sq > 4.5:
                continue
            depth = _min_overlap_depth(verts_list[i], verts_list[j])
            if depth > 0:
                penalty += weight * depth * depth
    return penalty

def _outer_hex_side(cfg, center, angle_deg):
    """
    Compute minimum side length S of outer hexagon containing all inner hexagons.
    """
    a = np.deg2rad(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0

    normals_angles = a + np.deg2rad(90 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(normals_angles), np.sin(normals_angles)))

    cx, cy = center
    max_need = 0.0

    for p in cfg:
        verts = _hex_vertices(p[0], p[1], p[2])
        rel_verts = verts - np.array([cx, cy])
        projections = rel_verts @ normals.T
        max_proj = np.max(projections)
        if max_proj > max_need:
            max_need = max_proj

    return max_need / apothem_unit

def _smooth_outer_side(cfg, center, angle_deg):
    """
    Smooth version of outer hex side that penalizes vertices near boundary.
    """
    a = np.deg2rad(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0

    normals_angles = a + np.deg2rad(90 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(normals_angles), np.sin(normals_angles)))

    cx, cy = center
    max_need = 0.0

    for p in cfg:
        verts = _hex_vertices(p[0], p[1], p[2])
        rel_verts = verts - np.array([cx, cy])
        projections = rel_verts @ normals.T
        max_proj = np.max(projections)
        if max_proj > max_need:
            max_need = max_proj

    return max_need / apothem_unit

def _is_valid(cfg):
    """Check if all pairs are disjoint."""
    verts_list = [_hex_vertices(cfg[i][0], cfg[i][1], cfg[i][2]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            dx = cfg[i][0] - cfg[j][0]
            dy = cfg[i][1] - cfg[j][1]
            if dx*dx + dy*dy > 4.0:
                continue
            depth = _min_overlap_depth(verts_list[i], verts_list[j])
            if depth > 1e-9:
                return False
    return True

def _decode_vector(x):
    """
    Decode optimization vector into configuration.
    x: array of length 3*N + 3
    First 3*N: (x_i, y_i, angle_i) for each hexagon
    Last 3: (cx, cy, outer_angle)
    """
    cfg = []
    for i in range(N):
        cfg.append([x[3*i], x[3*i+1], x[3*i+2]])
    center = (x[3*N], x[3*N+1])
    angle = x[3*N+2]
    return cfg, center, angle

def _objective(x):
    """
    Objective function: minimize outer hex side + collision penalty.
    """
    cfg, center, angle = _decode_vector(x)
    side = _outer_hex_side(cfg, center, angle)
    penalty = _smooth_collision_penalty(cfg, weight=50.0)
    return side + penalty

def _lattice_init():
    """Generate initial lattice configurations for seeding."""
    sqrt3 = math.sqrt(3)
    inits = []

    # 4-3-4-1 topology
    offsets = [
        (0, 0), (3, 0), (6, 0), (9, 0),
        (1.5, sqrt3/2), (4.5, sqrt3/2), (7.5, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3), (9, sqrt3),
        (4.5, 3*sqrt3/2),
    ]
    for theta in [0, 30, 60]:
        a_rad = np.deg2rad(theta)
        ca, sa = math.cos(a_rad), math.sin(a_rad)
        vec = np.zeros(3*N + 3)
        for i, (dx, dy) in enumerate(offsets):
            vec[3*i] = ca * dx - sa * dy
            vec[3*i+1] = sa * dx + ca * dy
            vec[3*i+2] = theta
        vec[3*N] = 0.0
        vec[3*N+1] = 0.0
        vec[3*N+2] = theta
        inits.append(vec)

    # 3-4-3-2 topology
    offsets2 = [
        (0, 0), (3, 0), (6, 0),
        (0, sqrt3/2), (3, sqrt3/2), (6, sqrt3/2), (9, sqrt3/2),
        (1.5, sqrt3), (4.5, sqrt3), (7.5, sqrt3),
        (3, 3*sqrt3/2), (6, 3*sqrt3/2),
    ]
    for theta in [0, 30]:
        a_rad = np.deg2rad(theta)
        ca, sa = math.cos(a_rad), math.sin(a_rad)
        vec = np.zeros(3*N + 3)
        for i, (dx, dy) in enumerate(offsets2):
            vec[3*i] = ca * dx - sa * dy
            vec[3*i+1] = sa * dx + ca * dy
            vec[3*i+2] = theta
        vec[3*N] = 0.0
        vec[3*N+1] = 0.0
        vec[3*N+2] = theta
        inits.append(vec)

    # 3-3-3-3 topology
    offsets3 = [
        (0, 0), (3, 0), (6, 0),
        (1.5, sqrt3/2), (4.5, sqrt3/2), (7.5, sqrt3/2),
        (0, sqrt3), (3, sqrt3), (6, sqrt3),
        (1.5, 3*sqrt3/2), (4.5, 3*sqrt3/2), (7.5, 3*sqrt3/2),
    ]
    for theta in [0, 15, 45]:
        a_rad = np.deg2rad(theta)
        ca, sa = math.cos(a_rad), math.sin(a_rad)
        vec = np.zeros(3*N + 3)
        for i, (dx, dy) in enumerate(offsets3):
            vec[3*i] = ca * dx - sa * dy
            vec[3*i+1] = sa * dx + ca * dy
            vec[3*i+2] = theta
        vec[3*N] = 0.0
        vec[3*N+1] = 0.0
        vec[3*N+2] = theta
        inits.append(vec)

    return inits

def construct():
    t0 = time.time()
    random.seed(42)

    best_score = float('inf')
    best_vec = None

    # Phase 1: Differential Evolution for global search
    bounds = [(-8, 8), (-8, 8), (-90, 90)] * N + [(-2, 2), (-2, 2), (-90, 90)]

    # Time budget for DE
    de_budget = 60.0
    de_deadline = t0 + de_budget

    try:
        # Use fewer generations but more pop size for diversity
        de_result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=30,
            popsize=15,
            tol=1e-6,
            mut=(0.3, 0.8),
            recomb=0.7,
            polish=False,
            callback=lambda xk, convergence: (
                convergence > 0 and time.time() > de_deadline
            )
        )

        if de_result.success or de_result.fun < best_score:
            best_score = de_result.fun
            best_vec = de_result.x.copy()

    except Exception:
        pass

    # Phase 2: Local refinement with SLSQP from multiple starting points
    if time.time() - t0 < TIME_LIMIT - 20:
        # Generate diverse starting points
        starts = _lattice_init()
        if best_vec is not None:
            starts.append(best_vec.copy())

        # Add random perturbations around best
        if best_vec is not None:
            for _ in range(3):
                perturbed = best_vec.copy()
                perturbed[:3*N] += random.gauss(0, 0.5, size=3*N)
                perturbed[3*N:3*N+3] += random.gauss(0, 0.2, size=3)
                starts.append(perturbed)

        slsqp_budget = TIME_LIMIT - 15
        slsqp_deadline = t0 + slsqp_budget

        for start in starts:
            if time.time() > slsqp_deadline:
                break

            try:
                res = minimize(
                    _objective,
                    x0=start,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )

                if res.fun < best_score:
                    best_score = res.fun
                    best_vec = res.x.copy()

            except Exception:
                continue

    # Phase 3: Final polish with tighter tolerance
    if best_vec is not None and time.time() - t0 < TIME_LIMIT - 10:
        try:
            res = minimize(
                _objective,
                x0=best_vec,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_vec = res.x.copy()
        except Exception:
            pass

    # Extract final configuration
    if best_vec is None:
        # Fallback: simple ring configuration
        cfg = []
        for i in range(N):
            theta = 2 * math.pi * i / N
            r = 10.0
            cfg.append([r * math.cos(theta), r * math.sin(theta), 0.0])
        center = (0.0, 0.0)
        angle = 0.0
        final_score = 20.0
    else:
        cfg, center, angle = _decode_vector(best_vec)
        # Compute actual score without penalty
        final_score = _outer_hex_side(cfg, center, angle)
        # Check validity
        if not _is_valid(cfg):
            # If invalid, try to find a valid one from our starts
            found_valid = False
            for start in _lattice_init():
                c2, ct2, a2 = _decode_vector(start)
                if _is_valid(c2):
                    final_score = _outer_hex_side(c2, ct2, a2)
                    cfg = c2
                    center = ct2
                    angle = a2
                    found_valid = True
                    break
            if not found_valid:
                # Last resort fallback
                cfg = []
                for i in range(N):
                    theta = 2 * math.pi * i / N
                    r = 10.0
                    cfg.append([r * math.cos(theta), r * math.sin(theta), 0.0])
                center = (0.0, 0.0)
                angle = 0.0
                final_score = 20.0

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
    return inner, (float(center[0]), float(center[1])), float(final_score), float(angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)

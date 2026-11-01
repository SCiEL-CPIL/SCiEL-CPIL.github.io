"""
Unit regular hexagons inside a regular hexagon – 11 hexagons.
Strategy:
  1. Build a contact-graph based initial layout (hexagonal lattice with one hex moved inward).
  2. Use a rigid-composite support-function objective: treat all 11 hexagons as one object
     with fixed relative positions (from the lattice), then optimize the composite pose
     (center, rotation) against the outer hexagon's support function.
  3. Jointly optimize inner positions + outer pose with a smooth surrogate (log-sum-exp)
     and contact-graph penalty.
  4. Final high-precision polish with exact support function.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0


def hex_vertices(cx, cy, side, angle_rad):
    """6 vertices of a regular hexagon."""
    thetas = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])


def hex_vertices_local(side, angle_rad):
    """Vertices relative to hex center."""
    thetas = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack([side * np.cos(thetas), side * np.sin(thetas)])


def container_normals(outer_angle_deg):
    """6 outward normals of the outer hexagon sides."""
    base = np.array([30.0 + i * 60.0 for i in range(6)]) * math.pi / 180.0
    ca = math.cos(math.radians(outer_angle_deg))
    sa = math.sin(math.radians(outer_angle_deg))
    R = np.array([[ca, -sa], [sa, ca]])
    normals = np.column_stack([np.cos(base), np.sin(base)])
    return normals @ R.T


def support_function(inner, outer_center, outer_angle_deg):
    """
    Exact minimum side length of outer hexagon to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    Returns the side length.
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = verts - np.array([ox, oy])
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        return 0.0
    return 2.0 * max_proj / SQRT3


def support_function_smooth(inner, outer_center, outer_angle_deg, beta=60.0):
    """
    Smooth surrogate via log-sum-exp over all (vertex, normal) projections.
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    n_inner = len(inner)
    all_verts = np.zeros((n_inner, 6, 2))
    for i, (ix, iy, iang) in enumerate(inner):
        all_verts[i] = hex_vertices(ix, iy, 1.0, math.radians(iang))
    local = (all_verts - np.array([ox, oy])) @ normals.T  # (n_inner, 6, 6)
    P = local.ravel()
    m = P.max()
    if m < -1e6:
        return 0.0
    shifted = beta * (P - m)
    shifted = np.clip(shifted, -60, 60)
    max_proj = float(m + np.log(np.sum(np.exp(shifted))) / beta)
    if max_proj < 0:
        return 0.0
    return 2.0 * max_proj / SQRT3


def sat_overlap(v1, v2, eps=1e-9):
    """Separating Axis Theorem for two convex polygons."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            nrm = np.linalg.norm(normal)
            if nrm > 1e-9:
                axes.append(normal / nrm)
    for axis in axes:
        d1 = v1 @ axis
        d2 = v2 @ axis
        if d1.max() < d2.min() + eps or d2.max() < d1.min() + eps:
            return False
    return True


def check_disjoint(inner, eps=1e-6):
    """Check all pairs are disjoint."""
    n = len(inner)
    polys = [hex_vertices(x, y, 1.0, math.radians(a)) for (x, y, a) in inner]
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True


def overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """Smooth penalty based on center distances (approximates non-overlap)."""
    n = len(inner)
    if n < 2:
        return 0.0
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists = np.sqrt(dx * dx + dy * dy)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    diff = np.maximum(0.0, min_dist - d)
    return float(np.sum(diff * diff))


def build_lattice_config():
    """
    Build a compact 11-hexagon lattice configuration.
    Use a hexagonal close-packed arrangement:
    Row 0: 3 hexes at y=0
    Row 1: 4 hexes at y=sqrt(3)
    Row 2: 3 hexes at y=2*sqrt(3)
    Row 3: 1 hex at y=3*sqrt(3) (centered)
    This gives a roughly diamond shape.
    """
    # Axial coordinates (q, r) -> Cartesian: x = 1.5*q, y = sqrt(3)*(r + q/2)
    # Compact 11-hex layout:
    #   (0,0), (1,0), (2,0)          row r=0
    #   (-1,1), (0,1), (1,1), (2,1)  row r=1
    #   (0,2), (1,2), (2,2)          row r=2
    #   (1,3)                        row r=3
    axial = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    centers = []
    for q, r in axial:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))
    return centers


def build_lattice_config2():
    """Alternative: center + 6 ring + 4 outer."""
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 1)
    ]
    # Fix duplicate: use (2,-1) instead of (-1,1)
    axial[10] = (2, -1)
    centers = []
    for q, r in axial:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))
    return centers


def build_lattice_config3():
    """Alternative: 3-4-3-1 but shifted to be more compact."""
    axial = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    # Shift to center
    xs = [1.5 * q for q, r in axial]
    ys = [SQRT3 * (r + q / 2.0) for q, r in axial]
    cx = np.mean(xs)
    cy = np.mean(ys)
    centers = [(x - cx, y - cy) for x, y in zip(xs, ys)]
    return centers


def rigid_composite_support(inner, outer_center, outer_angle_deg):
    """
    Support function for a rigid composite: all inner hexagons have fixed
    relative positions (angles are 0). The composite can only translate and rotate.
    We precompute all vertices in local frame, then apply a single rotation + translation.
    """
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    # All vertices in local (unrotated) frame
    all_verts = []
    for (ix, iy, iang) in inner:
        # For rigid composite, we assume all inner hexes have angle 0 in local frame
        v = hex_vertices(ix, iy, 1.0, 0.0)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # (66, 2)
    local = all_verts - np.array([ox, oy])
    projs = local @ normals.T
    max_proj = projs.max()
    if max_proj < 0:
        return 0.0
    return 2.0 * max_proj / SQRT3


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Generate multiple initial configurations
    configs = []
    for builder in [build_lattice_config, build_lattice_config2, build_lattice_config3]:
        centers = builder()
        # Center the configuration
        cx = np.mean([c[0] for c in centers])
        cy = np.mean([c[1] for c in centers])
        centered = [(x - cx, y - cy) for x, y in centers]
        configs.append(centered)

    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0

    for cfg_idx, centers in enumerate(configs):
        if time.time() > deadline - 35:
            break

        initial_inner = [[x, y, 0.0] for x, y in centers]

        # --- Phase 1: Optimize outer container (center, angle) with fixed inner layout ---
        def obj_outer_smooth(params, beta=50.0):
            cx, cy, ang = params
            return support_function_smooth(initial_inner, (cx, cy), ang, beta=beta)

        def obj_outer_exact(params):
            cx, cy, ang = params
            return support_function(initial_inner, (cx, cy), ang)

        best_val_p1 = float('inf')
        best_params_p1 = None

        for init_ang in [0, 15, 30, 45, 60, 75, 90]:
            if time.time() > deadline - 30:
                break
            p0 = [0.0, 0.0, float(init_ang)]
            try:
                res1 = minimize(obj_outer_smooth, p0, method='Nelder-Mead',
                                options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12,
                                         'maxfev': 3000, 'adaptive': True})
                res2 = minimize(obj_outer_exact, res1.x, method='Nelder-Mead',
                                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15,
                                         'maxfev': 6000, 'adaptive': True})
                if res2.fun < best_val_p1:
                    best_val_p1 = res2.fun
                    best_params_p1 = res2.x.copy()
            except Exception:
                pass

        if best_params_p1 is not None:
            ocx, ocy, oang = best_params_p1
        else:
            ocx, ocy, oang = 0.0, 0.0, 0.0
            best_val_p1 = support_function(initial_inner, (ocx, ocy), oang)

        # --- Phase 2: Joint optimization (inner positions + angles + outer pose) ---
        # Build adjacency for contact penalty
        adjacency = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                if math.sqrt(dx * dx + dy * dy) < SQRT3 * 1.15:
                    adjacency.append((i, j))

        def objective_joint(params, beta=45.0):
            inner_list = []
            for i in range(N):
                inner_list.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
            ocx = params[N * 3]
            ocy = params[N * 3 + 1]
            oang = params[N * 3 + 2]
            side = support_function_smooth(inner_list, (ocx, ocy), oang, beta=beta)
            pen = overlap_penalty(inner_list, min_dist=SQRT3 - 0.004)
            # Contact spring penalty
            spring_pen = 0.0
            for (i, j) in adjacency:
                dx = params[i * 3] - params[j * 3]
                dy = params[i * 3 + 1] - params[j * 3 + 1]
                d = math.sqrt(dx * dx + dy * dy)
                target = SQRT3 * 0.995
                spring_pen += (d - target) ** 2
            return side + 300.0 * pen + 0.05 * spring_pen

        init_params = []
        for x, y, a in initial_inner:
            init_params.extend([x, y, a])
        init_params.extend([ocx, ocy, oang])

        # Simulated annealing
        current = np.array(init_params, dtype=float)
        current_val = objective_joint(current, beta=40.0)
        best_params_sa = current.copy()
        best_val_sa = current_val

        T = 0.6
        T_min = 0.0005
        cooling = 0.99993
        max_iter = 15000
        patience = 2500
        no_improve = 0

        for it in range(max_iter):
            if time.time() > deadline - 20:
                break

            progress = 1.0 - T / 0.6
            beta_sa = 30.0 + 35.0 * min(progress, 1.0)
            beta_sa = min(beta_sa, 65.0)

            neighbor = current.copy()
            r = np.random.random()
            if r < 0.45:
                # Move one inner hex
                hi = np.random.randint(0, N)
                di = hi * 3 + np.random.randint(0, 3)
                if di % 3 == 2:
                    delta = np.random.normal(0, 0.12 * (1 + 0.3 * T))
                else:
                    delta = np.random.normal(0, 0.08 * (1 + 0.5 * T))
                neighbor[di] += delta
            elif r < 0.70:
                # Move two inner hexes
                hi = np.random.randint(0, N)
                hj = np.random.randint(0, N)
                if hi != hj:
                    for h in [hi, hj]:
                        di = h * 3 + np.random.randint(0, 2)
                        neighbor[di] += np.random.normal(0, 0.06 * (1 + 0.4 * T))
            else:
                # Move outer
                di = np.random.randint(N * 3, N * 3 + 3)
                if di == N * 3 + 2:
                    delta = np.random.normal(0, 1.2 * (1 + 0.3 * T))
                else:
                    delta = np.random.normal(0, 0.05 * (1 + 0.5 * T))
                neighbor[di] += delta

            nval = objective_joint(neighbor, beta=beta_sa)
            if nval < current_val or np.random.random() < math.exp(-(nval - current_val) / max(T, T_min)):
                current = neighbor
                current_val = nval
                if current_val < best_val_sa:
                    best_params_sa = current.copy()
                    best_val_sa = current_val
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve > patience:
                    no_improve = 0
                    T = max(T * 0.5, T_min * 2)
            T *= cooling
            if T < T_min:
                break

        # Extract best from SA
        best_inner = []
        for i in range(N):
            best_inner.append([best_params_sa[i * 3], best_params_sa[i * 3 + 1], best_params_sa[i * 3 + 2]])
        bocx = best_params_sa[N * 3]
        bocy = best_params_sa[N * 3 + 1]
        boang = best_params_sa[N * 3 + 2]

        # --- Phase 3: High-precision polish of outer container ---
        def obj_polish(params):
            cx, cy, ang = params
            return support_function(best_inner, (cx, cy), ang)

        def obj_polish_smooth(params, beta=80.0):
            cx, cy, ang = params
            return support_function_smooth(best_inner, (cx, cy), ang, beta=beta)

        best_polish_val = support_function(best_inner, (bocx, bocy), boang)
        best_polish_params = np.array([bocx, bocy, boang])

        # Multiple restarts around current best
        for da in [-6, -3, -1, 0, 1, 3, 6]:
            for dx in [-0.08, 0.0, 0.08]:
                for dy in [-0.08, 0.0, 0.08]:
                    if time.time() > deadline - 3:
                        break
                    p0 = [bocx + dx, bocy + dy, boang + da]
                    try:
                        rs = minimize(lambda p: support_function_smooth(best_inner, (p[0], p[1]), p[2], beta=80.0),
                                     p0, method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13,
                                              'maxfev': 3000, 'adaptive': True})
                        re = minimize(obj_polish, rs.x, method='Nelder-Mead',
                                     options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-16,
                                              'maxfev': 15000, 'adaptive': True})
                        if re.fun < best_polish_val:
                            best_polish_val = re.fun
                            best_polish_params = re.x.copy()
                    except Exception:
                        pass

        final_inner = best_inner
        final_center = (float(best_polish_params[0]), float(best_polish_params[1]))
        final_angle = float(best_polish_params[2])
        final_side = float(best_polish_val)

        # Verify disjointness
        if check_disjoint(final_inner, eps=1e-5):
            if final_side < best_global_side:
                best_global_side = final_side
                best_global_inner = final_inner
                best_global_center = final_center
                best_global_angle = final_angle
        else:
            # Try to fix: nudge overlapping pairs
            fixed = [list(h) for h in final_inner]
            for _ in range(80):
                overlap_found = False
                polys = [hex_vertices(fixed[i][0], fixed[i][1], 1.0, math.radians(fixed[i][2])) for i in range(N)]
                for i in range(N):
                    for j in range(i + 1, N):
                        if sat_overlap(polys[i], polys[j], eps=1e-6):
                            overlap_found = True
                            dx = fixed[j][0] - fixed[i][0]
                            dy = fixed[j][1] - fixed[i][1]
                            d = math.sqrt(dx * dx + dy * dy) + 1e-9
                            nx, ny = dx / d, dy / d
                            fixed[i][0] -= 0.008 * nx
                            fixed[i][1] -= 0.008 * ny
                            fixed[j][0] += 0.008 * nx
                            fixed[j][1] += 0.008 * ny
                            polys[i] = hex_vertices(fixed[i][0], fixed[i][1], 1.0, math.radians(fixed[i][2]))
                            polys[j] = hex_vertices(fixed[j][0], fixed[j][1], 1.0, math.radians(fixed[j][2]))
                if not overlap_found:
                    break
            if check_disjoint(fixed, eps=1e-5):
                s = support_function(fixed, final_center, final_angle)
                if s < best_global_side:
                    best_global_side = s
                    best_global_inner = fixed
                    best_global_center = final_center
                    best_global_angle = final_angle

    # Fallback
    if best_global_inner is None:
        centers = build_lattice_config()
        cx = np.mean([c[0] for c in centers])
        cy = np.mean([c[1] for c in centers])
        best_global_inner = [[x - cx, y - cy, 0.0] for x, y in centers]
        best_global_center = (0.0, 0.0)
        best_global_angle = 0.0
        best_global_side = support_function(best_global_inner, best_global_center, best_global_angle)

    return (best_global_inner, best_global_center,
            float(best_global_side), float(best_global_angle))

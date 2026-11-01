"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth surrogate of compute_required_side via log-sum-exp over all
    (vertex, normal) projections. As beta -> inf this converges to the
    exact max-projection side length. Vectorized for speed.
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    n = len(inner)
    # all vertices: (n*6, 2)
    thetas = np.repeat(np.arange(6) * (math.pi / 3.0), n) + np.radians([p[2] for p in inner])
    verts = np.column_stack([
        np.repeat([p[0] for p in inner], 6) + np.cos(thetas),
        np.repeat([p[1] for p in inner], 6) + np.sin(thetas),
    ])
    dx = verts[:, 0] - ox
    dy = verts[:, 1] - oy
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a

    # projections: (n*6, 6)
    proj = lx[:, None] * normals[0, :] + ly[:, None] * normals[1, :]
    flat = proj.ravel()

    m = float(flat.max())
    if m < -1e12:
        return 0.0
    # stable log-sum-exp
    e = np.exp(beta * (flat - m))
    s = float(e.sum())
    if s <= 0.0:
        smooth_max = m
    else:
        smooth_max = m + math.log(s) / beta
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for i in range(len(v1)):
        p1 = v1[i]; p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    for i in range(len(v2)):
        p1 = v2[i]; p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """Smooth penalty for overlaps based on center distances (proxy)."""
    n = len(inner)
    if n < 2:
        return 0.0
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])
    diff = np.maximum(0.0, min_dist - dists)
    return float(np.sum(diff * diff))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s limit

    # 1. 11 hexagons on a hexagonal lattice (center + 6 + 4 outer)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]

    epsilon = 1e-6
    scale = 1.0 + epsilon
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))

    initial_inner = [[x, y, 0.0] for x, y in centers]

    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    # 3. Outer-only optimization: smooth basin-find -> exact polish, multi-start
    def objective_outer_smooth(params):
        return compute_required_side_smooth(initial_inner, (params[0], params[1]), params[2], beta=60.0)

    def objective_outer_exact(params):
        return compute_required_side(initial_inner, (params[0], params[1]), params[2])

    best_outer_params = None
    best_outer_val = float('inf')

    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165,
                   180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]

    for init_angle in init_angles:
        if time.time() > deadline - 30:
            break
        init_params = [cx_avg, cy_avg, float(init_angle)]
        # Phase A: smooth basin finding
        try:
            res_s = minimize(objective_outer_smooth, init_params, method='Nelder-Mead',
                             options={'maxiter': 400, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 800})
            # Phase B: exact polish from the smooth basin
            res_e = minimize(objective_outer_exact, res_s.x, method='Nelder-Mead',
                             options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 3000})
            if res_e.fun < best_outer_val:
                best_outer_val = res_e.fun
                best_outer_params = res_e.x
        except Exception:
            pass

    if best_outer_params is not None:
        best_outer_center = (float(best_outer_params[0]), float(best_outer_params[1]))
        best_outer_angle = float(best_outer_params[2])
        best_side = best_outer_val
    else:
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)

    # 4. Joint optimization (inner + outer) with smooth objective for SA coherence
    def objective_joint(params):
        inner_list = []
        for i in range(N):
            inner_list.append([params[i*3], params[i*3+1], params[i*3+2]])
        ocx = params[N*3]; ocy = params[N*3+1]; oangle = params[N*3+2]
        side = compute_required_side_smooth(inner_list, (ocx, ocy), oangle, beta=40.0)
        penalty = compute_overlap_penalty(inner_list)
        return side + 1000.0 * penalty

    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

    current_params = np.array(initial_params)
    current_val = objective_joint(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val

    T = 1.0
    T_min = 0.001
    cooling = 0.9997
    base_step_inner = 0.3
    base_step_outer = 0.2
    max_iter = 20000
    no_improve_count = 0
    patience = 2000

    for i in range(max_iter):
        if time.time() > deadline - 25:
            break
        neighbor = current_params.copy()
        temp_scale = max(0.1, T)
        if np.random.random() < 0.8:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 3)
            if dim == 2:
                delta = np.random.normal(0, 5.0 * temp_scale)
            else:
                delta = np.random.normal(0, base_step_inner * temp_scale)
            neighbor[idx*3 + dim] += delta
        else:
            dim = np.random.randint(N*3, N*3+3)
            if dim == N*3+2:
                delta = np.random.normal(0, 4.0 * temp_scale)
            else:
                delta = np.random.normal(0, base_step_outer * temp_scale)
            neighbor[dim] += delta

        neighbor_val = objective_joint(neighbor)
        if neighbor_val < current_val or (T > 1e-10 and np.random.random() < math.exp(-(neighbor_val - current_val) / T)):
            current_params = neighbor
            current_val = neighbor_val
            if current_val < best_val_joint:
                best_params = current_params.copy()
                best_val_joint = current_val
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        if no_improve_count % 500 == 0 and no_improve_count > 0:
            base_step_inner *= 0.95
            base_step_outer *= 0.95
        T *= cooling
        if T < T_min:
            break
        if no_improve_count > patience and T < 0.05:
            break

    best_inner_joint = []
    for i in range(N):
        best_inner_joint.append([float(best_params[i*3]), float(best_params[i*3+1]), float(best_params[i*3+2])])
    best_ocx = float(best_params[N*3])
    best_ocy = float(best_params[N*3+1])
    best_oangle = float(best_params[N*3+2])

    # Re-evaluate with exact side; accept only if disjoint and improves
    joint_exact_side = compute_required_side(best_inner_joint, (best_ocx, best_ocy), best_oangle)
    if check_all_disjoint(best_inner_joint, eps=1e-5) and joint_exact_side < best_side + 1e-9:
        best_inner = best_inner_joint
        best_outer_center = (best_ocx, best_ocy)
        best_outer_angle = best_oangle
        best_side = joint_exact_side

    # 5. Final polish: smooth -> exact two-phase on outer params around current best
    def objective_outer_polish_smooth(params):
        return compute_required_side_smooth(best_inner, (params[0], params[1]), params[2], beta=150.0)

    def objective_outer_polish_exact(params):
        return compute_required_side(best_inner, (params[0], params[1]), params[2])

    polish_starts = [
        [best_outer_center[0], best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 1],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 1],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 3],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 3],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 7],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 7],
        [best_outer_center[0] + 0.03, best_outer_center[1], best_outer_angle],
        [best_outer_center[0] - 0.03, best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1] + 0.03, best_outer_angle],
        [best_outer_center[0], best_outer_center[1] - 0.03, best_outer_angle],
        [best_outer_center[0] + 0.02, best_outer_center[1] + 0.02, best_outer_angle],
        [best_outer_center[0] - 0.02, best_outer_center[1] - 0.02, best_outer_angle],
        [best_outer_center[0] + 0.02, best_outer_center[1] - 0.02, best_outer_angle],
        [best_outer_center[0] - 0.02, best_outer_center[1] + 0.02, best_outer_angle],
    ]

    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]

    for ps in polish_starts:
        if time.time() > deadline - 6:
            break
        try:
            res_s = minimize(objective_outer_polish_smooth, ps, method='Nelder-Mead',
                             options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1500})
            res_e = minimize(objective_outer_polish_exact, res_s.x, method='Nelder-Mead',
                             options={'maxiter': 4000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 10000})
            if res_e.fun < best_polish_val:
                best_polish_val = res_e.fun
                best_polish_params = res_e.x
        except Exception:
            pass

    # 6. Ultra-tight exact round around the best polish point
    if time.time() < deadline - 2:
        fine_starts = [
            [best_polish_params[0], best_polish_params[1], best_polish_params[2]],
            [best_polish_params[0], best_polish_params[1], best_polish_params[2] + 0.5],
            [best_polish_params[0], best_polish_params[1], best_polish_params[2] - 0.5],
            [best_polish_params[0] + 0.01, best_polish_params[1], best_polish_params[2]],
            [best_polish_params[0] - 0.01, best_polish_params[1], best_polish_params[2]],
            [best_polish_params[0], best_polish_params[1] + 0.01, best_polish_params[2]],
            [best_polish_params[0], best_polish_params[1] - 0.01, best_polish_params[2]],
        ]
        for ps in fine_starts:
            if time.time() > deadline - 1:
                break
            try:
                res = minimize(objective_outer_polish_exact, ps, method='Nelder-Mead',
                               options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-17, 'maxfev': 20000})
                if res.fun < best_polish_val:
                    best_polish_val = res.fun
                    best_polish_params = res.x
            except Exception:
                pass

    if best_polish_params is not None:
        best_outer_center = (float(best_polish_params[0]), float(best_polish_params[1]))
        best_outer_angle = float(best_polish_params[2])
        best_side = best_polish_val

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END

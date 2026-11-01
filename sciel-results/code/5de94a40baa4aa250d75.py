"""Improved 11-hexagon packing: contact-graph based, rigid composite, multi-start, high-precision polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = math.sqrt(3) / 2.0

# EVOLVE-BLOCK-START

def hex_vertices_local(angle_rad):
    """Return 6 vertices of a unit regular hexagon centered at origin."""
    thetas = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack([np.cos(thetas), np.sin(thetas)])

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    thetas = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    base_angles = np.array([30 + i * 60 for i in range(6)]) * math.pi / 180.0
    normals = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    normals_rot = normals @ R.T

    max_proj = -np.inf
    for (ix, iy, iangle) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iangle))
        local = verts - np.array([ox, oy])
        projs = local @ normals_rot.T
        cur = np.max(projs)
        if cur > max_proj:
            max_proj = cur

    if max_proj < 0:
        return 0.0
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth surrogate using log-sum-exp over all vertex-normal projections.
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    base_angles = np.array([30 + i * 60 for i in range(6)]) * math.pi / 180.0
    normals = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    normals_rot = normals @ R.T

    n_inner = len(inner)
    all_verts = np.zeros((n_inner, 6, 2))
    for i, (ix, iy, iangle) in enumerate(inner):
        all_verts[i] = hex_vertices(ix, iy, 1.0, math.radians(iangle))
    
    local = (all_verts - np.array([ox, oy])) @ R.T
    projs = local @ normals_rot.T
    
    P = projs.ravel()
    if P.size == 0:
        return 0.0

    m = np.max(P)
    if m < -1e6:
        return 0.0
    shifted = beta * (P - m)
    shifted = np.clip(shifted, -50, 50)
    max_proj = float(m + np.log(np.sum(np.exp(shifted))) / beta)

    if max_proj < 0:
        return 0.0
    return 2.0 * max_proj / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
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

def check_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def overlap_penalty(inner, min_dist=SQRT3 - 0.003):
    """Smooth penalty for center-distance violations."""
    n = len(inner)
    if n < 2:
        return 0.0
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists = np.sqrt(dx * dx + dy * dy + 1e-12)
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    diff = np.maximum(0.0, min_dist - pair_dists)
    return float(np.sum(diff * diff))

def build_lattice_config(n_hex):
    """Build a compact hexagonal lattice configuration for n_hex hexagons."""
    configs = []
    
    axial_a = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    configs.append(axial_a[:n_hex])
    
    axial_b = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    configs.append(axial_b[:n_hex])
    
    axial_c = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    configs.append(axial_c[:n_hex])
    
    axial_d = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    configs.append(axial_d[:n_hex])
    
    axial_e = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    configs.append(axial_e[:n_hex])
    
    axial_f = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    configs.append(axial_f[:n_hex])
    
    return configs

def axial_to_cartesian(axial_coords, scale=1.0):
    """Convert axial coordinates to Cartesian."""
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
    return centers

def try_orientation_optimization(inner_fixed, time_budget=10.0):
    """Optimize outer container orientation and position for fixed inner layout."""
    start = time.time()
    cx_avg = sum(h[0] for h in inner_fixed) / len(inner_fixed)
    cy_avg = sum(h[1] for h in inner_fixed) / len(inner_fixed)
    
    best_val = float('inf')
    best_params = None
    
    for init_angle in [0, 15, 30, 45, 60, 75, 90, 105, 120]:
        if time.time() - start > time_budget:
            break
        
        def obj(params):
            return compute_required_side_fast(inner_fixed, (params[0], params[1]), params[2])
        
        try:
            res = minimize(obj, [cx_avg, cy_avg, float(init_angle)], method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 3000,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        return (cx_avg, cy_avg), 0.0, compute_required_side_fast(inner_fixed, (cx_avg, cy_avg), 0.0)
    
    return (best_params[0], best_params[1]), best_params[2], best_val

def joint_optimize(initial_inner, outer_center, outer_angle, adjacency, time_budget=60.0):
    """Joint optimization of inner positions/angles and outer container using simulated annealing."""
    start = time.time()
    deadline = start + time_budget
    
    n = N
    params = []
    for x, y, a in initial_inner:
        params.extend([x, y, a])
    params.extend([outer_center[0], outer_center[1], outer_angle])
    params = np.array(params, dtype=float)
    
    def objective(p, beta=40.0):
        inner_list = []
        for i in range(n):
            inner_list.append([p[i*3], p[i*3+1], p[i*3+2]])
        ocx, ocy, oangle = p[n*3], p[n*3+1], p[n*3+2]
        side = compute_required_side_smooth(inner_list, (ocx, ocy), oangle, beta=beta)
        pen = overlap_penalty(inner_list, min_dist=SQRT3 - 0.003)
        return side + 150.0 * pen
    
    current_val = objective(params, beta=40.0)
    best_params = params.copy()
    best_val = current_val
    
    T = 0.5
    T_min = 0.0005
    cooling = 0.99985
    no_improve = 0
    patience = 1500
    
    max_iter = 15000
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        progress = 1.0 - T / 0.5
        beta = 30.0 + 40.0 * min(progress, 1.0)
        beta = min(beta, 70.0)
        
        neighbor = current_params.copy()
        r = np.random.random()
        
        if r < 0.55:
            hidx = np.random.randint(0, n)
            dim = hidx * 3 + np.random.randint(0, 3)
            if dim % 3 == 2:
                neighbor[dim] += np.random.normal(0, 0.12 * (1 + 0.4 * T))
            else:
                neighbor[dim] += np.random.normal(0, 0.08 * (1 + 0.5 * T))
        elif r < 0.8:
            h1 = np.random.randint(0, n)
            h2 = np.random.randint(0, n)
            if h1 != h2:
                for h in [h1, h2]:
                    dim = h * 3 + np.random.randint(0, 2)
                    neighbor[dim] += np.random.normal(0, 0.07 * (1 + 0.4 * T))
        else:
            dim = np.random.randint(n*3, n*3+3)
            if dim == n*3+2:
                neighbor[dim] += np.random.normal(0, 1.2 * (1 + 0.3 * T))
            else:
                neighbor[dim] += np.random.normal(0, 0.05 * (1 + 0.5 * T))
        
        neighbor_val = objective(neighbor, beta=beta)
        
        if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / max(T, T_min)):
            current_params = neighbor
            current_val = neighbor_val
            if current_val < best_val:
                best_params = current_params.copy()
                best_val = current_val
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > patience:
                no_improve = 0
                T = max(T * 0.5, T_min * 2)
        
        T *= cooling
        if T < T_min:
            break
    
    best_inner = []
    for i in range(n):
        best_inner.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
    best_oc = (best_params[n*3], best_params[n*3+1])
    best_oa = best_params[n*3+2]
    
    return best_inner, best_oc, best_oa, best_val

def final_polish(inner, outer_center, outer_angle, time_budget=15.0):
    """High-precision polish of the outer container."""
    start = time.time()
    deadline = start + time_budget
    
    def obj_exact(params):
        return compute_required_side_fast(inner, (params[0], params[1]), params[2])
    
    def obj_smooth(params):
        return compute_required_side_smooth(inner, (params[0], params[1]), params[2], beta=100.0)
    
    best_val = compute_required_side_fast(inner, outer_center, outer_angle)
    best_params = np.array([outer_center[0], outer_center[1], outer_angle])
    
    base_angle = outer_angle
    starts = []
    for da in [-6, -3, -1, 0, 1, 3, 6]:
        for dc in [(-0.05, 0), (0.05, 0), (0, -0.05), (0, 0.05), (0, 0)]:
            starts.append([outer_center[0] + dc[0], outer_center[1] + dc[1], base_angle + da])
    
    for ps in starts:
        if time.time() > deadline:
            break
        try:
            res1 = minimize(obj_smooth, ps, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 2500,
                                    'adaptive': True})
            res2 = minimize(obj_exact, res1.x, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 10000,
                                    'adaptive': True})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    return (best_params[0], best_params[1]), best_params[2], best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0
    
    configs = build_lattice_config(N)
    
    for cfg_idx, axial_coords in enumerate(configs):
        if time.time() > deadline - 25:
            break
        
        if len(axial_coords) < N:
            axial_coords = list(axial_coords) + [(0, 0)] * (N - len(axial_coords))
        elif len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        
        scale = 1.001
        centers = axial_to_cartesian(axial_coords, scale)
        initial_inner = [[x, y, 0.0] for x, y in centers]
        
        outer_center, outer_angle, side_val = try_orientation_optimization(
            initial_inner, time_budget=8.0)
        
        adjacency = set()
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < SQRT3 * 1.15:
                    adjacency.add((i, j))
        
        remaining = deadline - time.time()
        joint_budget = min(50.0, remaining - 20.0)
        if joint_budget < 5.0:
            joint_budget = 5.0
        
        final_inner, final_center, final_angle, joint_val = joint_optimize(
            initial_inner, outer_center, outer_angle, adjacency, time_budget=joint_budget)
        
        remaining = deadline - time.time()
        polish_budget = min(15.0, remaining - 3.0)
        if polish_budget < 2.0:
            polish_budget = 2.0
        
        final_center, final_angle, final_side = final_polish(
            final_inner, final_center, final_angle, time_budget=polish_budget)
        
        if check_disjoint(final_inner, eps=1e-4):
            if final_side < best_global_side:
                best_global_side = final_side
                best_global_inner = final_inner
                best_global_center = final_center
                best_global_angle = final_angle
        else:
            fixed_inner = [list(h) for h in final_inner]
            for attempt in range(30):
                overlap_found = False
                polys = [hex_vertices(h[0], h[1], 1.0, math.radians(h[2])) for h in fixed_inner]
                for i in range(N):
                    for j in range(i + 1, N):
                        if sat_overlap(polys[i], polys[j], eps=1e-6):
                            overlap_found = True
                            dx = fixed_inner[j][0] - fixed_inner[i][0]
                            dy = fixed_inner[j][1] - fixed_inner[i][1]
                            d = math.sqrt(dx*dx + dy*dy) + 1e-9
                            nx, ny = dx/d, dy/d
                            fixed_inner[i][0] -= 0.015 * nx
                            fixed_inner[i][1] -= 0.015 * ny
                            fixed_inner[j][0] += 0.015 * nx
                            fixed_inner[j][1] += 0.015 * ny
                            break
                    if overlap_found:
                        break
                if not overlap_found:
                    break
            
            if check_disjoint(fixed_inner, eps=1e-5):
                side_fixed = compute_required_side_fast(fixed_inner, final_center, final_angle)
                if side_fixed < best_global_side:
                    best_global_side = side_fixed
                    best_global_inner = fixed_inner
                    best_global_center = final_center
                    best_global_angle = final_angle
            else:
                fallback_side = compute_required_side_fast(initial_inner, outer_center, outer_angle)
                if fallback_side < best_global_side:
                    best_global_side = fallback_side
                    best_global_inner = initial_inner
                    best_global_center = outer_center
                    best_global_angle = outer_angle
    
    if best_global_inner is None:
        centers = axial_to_cartesian([(0,0)]*N, 1.001)
        best_global_inner = [[x, y, 0.0] for x, y in centers]
        cx = sum(c[0] for c in centers) / N
        cy = sum(c[1] for c in centers) / N
        best_global_center = (cx, cy)
        best_global_angle = 0.0
        best_global_side = compute_required_side_fast(best_global_inner, best_global_center, 0.0)
    
    return (best_global_inner, best_global_center,
            float(best_global_side), float(best_global_angle))

# EVOLVE-BLOCK-END

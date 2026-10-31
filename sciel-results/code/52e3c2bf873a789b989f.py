"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    """
    axes = []
    
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
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
    """
    Compute a smooth penalty for overlaps based on center distances.
    """
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < min_dist:
                diff = min_dist - dist
                penalty += diff * diff
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Define 11 hexagons on a hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian
    epsilon = 1e-6
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 3. Optimize Outer Hexagon using scipy with multiple starts
    def objective_outer(params):
        cx, cy, angle_deg = params
        side = compute_required_side(initial_inner, (cx, cy), angle_deg)
        return side
    
    def objective_outer_smooth(params):
        cx, cy, angle_deg = params
        side = compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=50.0)
        return side

    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    best_result = None
    best_val = float('inf')
    
    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
    
    for init_angle in init_angles:
        if time.time() > deadline - 25:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            # First pass: smooth optimization
            res_smooth = minimize(objective_outer_smooth, init_params, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 1000})
            
            # Second pass: exact optimization starting from smooth result
            res = minimize(objective_outer, res_smooth.x, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 2000})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass
            
    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = best_val
    else:
        # Fallback
        best_side = float('inf')
        for outer_angle_deg in range(0, 360, 5):
            current_cx = cx_avg
            current_cy = cy_avg
            step = 0.5
            for _ in range(3):
                improved = False
                for dx in np.linspace(-step, step, 5):
                    for dy in np.linspace(-step, step, 5):
                        oc = (current_cx + dx, current_cy + dy)
                        side = compute_required_side(initial_inner, oc, outer_angle_deg)
                        if side < best_side - 1e-6:
                            best_side = side
                            best_outer_center = oc
                            best_outer_angle = outer_angle_deg
                            current_cx = oc[0]
                            current_cy = oc[1]
                            improved = True
                if not improved:
                    break
                step *= 0.5
            if time.time() > deadline - 5:
                break

    # 4. Joint Optimization using Differential Evolution
    def objective_joint(params):
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        ocx = params[N*3]
        ocy = params[N*3+1]
        oangle = params[N*3+2]
        
        side = compute_required_side(inner_list, (ocx, ocy), oangle)
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 1000.0 * penalty
    
    bounds = []
    for i in range(N):
        bounds.extend([(-5, 5), (-5, 5), (0, 360)])
    bounds.extend([(-3, 3), (-3, 3), (0, 360)])
    
    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    best_de_result = None
    best_de_val = float('inf')
    
    try:
        de_result = differential_evolution(
            objective_joint,
            bounds=bounds,
            seed=42,
            maxiter=50,
            tol=1e-8,
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=False,
            init='sobol',
            update_interval=1,
            callback=None
        )
        
        if de_result.fun < best_de_val:
            best_de_val = de_result.fun
            best_de_result = de_result.x
    except Exception:
        pass
    
    # Simulated annealing
    current_params = np.array(initial_params)
    current_val = objective_joint(current_params)
    best_params_sa = current_params.copy()
    best_val_sa = current_val
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    max_iter_sa = 10000
    for i in range(max_iter_sa):
        if time.time() > deadline - 15:
            break
        
        neighbor = current_params.copy()
        
        if np.random.random() < 0.8:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 3)
            delta = np.random.normal(0, 0.3)
            neighbor[idx*3 + dim] += delta
        else:
            dim = np.random.randint(N*3, N*3+3)
            if dim == N*3+2:
                delta = np.random.normal(0, 5.0)
            else:
                delta = np.random.normal(0, 0.2)
            neighbor[dim] += delta
        
        neighbor_val = objective_joint(neighbor)
        
        if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / T):
            current_params = neighbor
            current_val = neighbor_val
            
            if current_val < best_val_sa:
                best_params_sa = current_params.copy()
                best_val_sa = current_val
        
        T *= cooling
        if T < T_min:
            break
    
    if best_de_result is not None and best_de_val < best_val_sa:
        best_params = best_de_result
        best_val_joint = best_de_val
    else:
        best_params = best_params_sa
        best_val_joint = best_val_sa
    
    best_inner_joint = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        best_inner_joint.append([x, y, a])
    
    best_ocx = best_params[N*3]
    best_ocy = best_params[N*3+1]
    best_oangle = best_params[N*3+2]
    
    if check_all_disjoint(best_inner_joint, eps=1e-5):
        best_inner = best_inner_joint
        best_outer_center = (best_ocx, best_ocy)
        best_outer_angle = best_oangle
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    else:
        best_inner = initial_inner
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    # 5. Final Polish: Tighten the outer container around the fixed inner layout
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    def objective_outer_polish_smooth(params):
        cx, cy, angle_deg = params
        return compute_required_side_smooth(best_inner, (cx, cy), angle_deg, beta=50.0)
    
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 5],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 5],
        [best_outer_center[0] + 0.1, best_outer_center[1], best_outer_angle],
        [best_outer_center[0] - 0.1, best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1] + 0.1, best_outer_angle],
        [best_outer_center[0], best_outer_center[1] - 0.1, best_outer_angle],
    ]
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    for ps in polish_starts:
        if time.time() > deadline - 2:
            break
        try:
            # Smooth pass first
            res_smooth = minimize(objective_outer_polish_smooth, ps, method='Nelder-Mead',
                                  options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 1000})
            
            # Exact pass
            res = minimize(objective_outer_polish, res_smooth.x, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 10000})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x
        except Exception:
            pass
    
    if best_polish_params is not None:
        best_outer_center = (best_polish_params[0], best_polish_params[1])
        best_outer_angle = best_polish_params[2]
        best_side = best_polish_val
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END

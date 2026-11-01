"""Combined: Rigid-composite initialization + Smooth Surrogate SA + High-Precision Polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = math.sqrt(3) / 2.0
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# EVOLVE-BLOCK-START

def hex_verts(cx, cy, side, angle_rad):
    """6 vertices of a regular hexagon as numpy array."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def container_normals(outer_angle_deg):
    """6 outward normals of the container hexagon."""
    base = np.radians(30)
    a = math.radians(outer_angle_deg)
    cs, sn = math.cos(a), math.sin(a)
    normals = []
    for i in range(6):
        t = base + i * np.pi / 3.0
        nx = math.cos(t)
        ny = math.sin(t)
        rx = nx * cs - ny * sn
        ry = nx * sn + ny * cs
        normals.append((rx, ry))
    return normals

def support_function_exact(inner, outer_center, outer_angle_deg):
    """
    Max projection of all inner hexagon vertices onto the 6 container normals.
    Returns the required side length.
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    max_proj = 0.0
    for (ix, iy, iang) in inner:
        verts = hex_verts(ix, iy, 1.0, math.radians(iang))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            for (nx, ny) in normals:
                p = dx * nx + dy * ny
                if p > max_proj:
                    max_proj = p
    return 2.0 * max_proj / SQRT3

def support_function_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """Smooth log-sum-exp version of support function."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    projs = []
    for (ix, iy, iang) in inner:
        verts = hex_verts(ix, iy, 1.0, math.radians(iang))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            for (nx, ny) in normals:
                projs.append(dx * nx + dy * ny)
    if not projs:
        return 0.0
    P = np.array(projs)
    m = P.max()
    if m < -1e6:
        return 0.0
    # Log-sum-exp trick for stability
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap for two convex polygons (numpy arrays)."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            ln = np.linalg.norm(n)
            if ln > 1e-12:
                axes.append(n / ln)
    for ax in axes:
        min1 = np.min(v1 @ ax)
        max1 = np.max(v1 @ ax)
        min2 = np.min(v2 @ ax)
        max2 = np.max(v2 @ ax)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT disjointness."""
    n = len(inner)
    polys = [hex_verts(x, y, 1.0, math.radians(a)) for (x, y, a) in inner]
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def contact_penalty(inner, min_dist=SQRT3 - 0.001):
    """Quadratic penalty for center-distance violations."""
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

def build_axial(axial_coords, s=1.0 + 1e-8):
    """Build inner hexagons from axial coords."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + q / 2.0)) for q, r in axial_coords]
    return [[x, y, 0.0] for x, y in centers]

def generate_candidates():
    """Generate diverse initial layouts."""
    candidates = []
    
    # 1. Symmetric Cross (Center + 6 ring + 4 extended)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_1)

    # 2. Compact 3-4-3 Cluster
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    candidates.append(axial_2)

    # 3. 4-4-3 Block
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_3)

    # 4. Diamond + Extensions
    axial_4 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_4)

    # 5. 5-6 Rows
    axial_5 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_5)
    
    # 6. Compact Hexagon 1-3-3-3-1 (Total 11? No, 1+3+3+3+1 = 11)
    # Axial coords for a hexagon of radius 2 is 19. We need 11.
    # Let's try a dense cluster:
    axial_6 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), # 7
        (2,0), (1,1), (0,2), # 3 more
        (-2,0) # 1 more
    ]
    candidates.append(axial_6)

    return candidates

def optimize_joint_sa(local_positions, deadline, seed=42):
    """
    Simulated Annealing for joint optimization of all inner hexagons and outer container.
    """
    np.random.seed(seed)
    n = len(local_positions)
    
    # Initial state: Use the provided local positions, center them, angle 0
    cx0 = sum(p[0] for p in local_positions) / n
    cy0 = sum(p[1] for p in local_positions) / n
    
    # Shift to center
    initial_params = []
    for x, y, a in local_positions:
        initial_params.extend([x - cx0, y - cy0, a])
    initial_params.extend([0.0, 0.0, 0.0]) # Outer center (0,0), Outer angle 0
    
    current_params = np.array(initial_params)
    
    def get_inner(params):
        inner = []
        for i in range(n):
            inner.append([params[i*3], params[i*3+1], params[i*3+2]])
        return inner
    
    def get_outer(params):
        return (params[n*3], params[n*3+1]), params[n*3+2]

    def objective(params, beta=50.0, penalty_weight=100.0):
        inner = get_inner(params)
        ocx, ocy = params[n*3], params[n*3+1]
        oangle = params[n*3+2]
        
        side = support_function_smooth(inner, (ocx, ocy), oangle, beta=beta)
        pen = contact_penalty(inner, SQRT3 - 0.001)
        
        return side + penalty_weight * pen

    current_val = objective(current_params)
    best_params = current_params.copy()
    best_val = current_val
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9998
    max_iter = 15000
    no_improve_count = 0
    patience = 2000
    
    start_time = time.time()
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Dynamic penalty weight: increase as T drops to enforce constraints
        progress = 1.0 - (T - T_min) / (1.0 - T_min)
        penalty_weight = 50.0 + 500.0 * (progress ** 2)
        
        # Step size schedule
        scale = 0.15 * (0.1 + T)
        
        neighbor = current_params.copy()
        
        # Randomly pick a dimension to perturb
        # 80% chance to perturb an inner hexagon param
        # 20% chance to perturb outer container param
        if np.random.random() < 0.8:
            dim_idx = np.random.randint(0, n * 3)
            # Angle dimensions (every 3rd) might need different scaling
            if dim_idx % 3 == 2:
                delta = np.random.normal(0, 5.0 * (0.1 + T)) # Degrees
            else:
                delta = np.random.normal(0, scale)
            neighbor[dim_idx] += delta
        else:
            dim_idx = np.random.randint(n * 3, n * 3 + 3)
            if dim_idx == n * 3 + 2: # Outer angle
                delta = np.random.normal(0, 10.0 * (0.1 + T))
            else:
                delta = np.random.normal(0, scale)
            neighbor[dim_idx] += delta
            
        neighbor_val = objective(neighbor, penalty_weight=penalty_weight)
        
        # Acceptance criterion
        if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / max(T, T_min)):
            current_params = neighbor
            current_val = neighbor_val
            
            if current_val < best_val:
                best_params = current_params.copy()
                best_val = current_val
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count > patience:
                no_improve_count = 0
                T = max(T * 0.5, T_min * 2)
                
        T *= cooling
        if T < T_min:
            break
            
    return best_params, best_val

def polish_inner_positions(inner, outer_center, outer_angle, deadline):
    """
    High-precision polish of inner positions with outer container fixed.
    """
    n = len(inner)
    
    def obj_inner(params):
        inner_list = []
        for i in range(n):
            inner_list.append([params[i*3], params[i*3+1], params[i*3+2]])
        side = support_function_exact(inner_list, outer_center, outer_angle)
        pen = contact_penalty(inner_list, SQRT3 - 0.0005)
        return side + 1000.0 * pen
        
    x0 = []
    for x, y, a in inner:
        x0.extend([x, y, a])
        
    try:
        res = minimize(obj_inner, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 3000})
        if res.success or res.fun < obj_inner(x0):
            new_inner = []
            for i in range(n):
                new_inner.append([res.x[i*3], res.x[i*3+1], res.x[i*3+2]])
            return new_inner
    except Exception:
        pass
    return inner

def polish_outer_container(inner, deadline):
    """
    High-precision polish of outer container parameters.
    """
    def obj_outer(params):
        cx, cy, angle = params
        return support_function_exact(inner, (cx, cy), angle)
        
    # Multi-start for outer polish
    best_val = float('inf')
    best_params = None
    
    # Start from current best
    starts = []
    # We don't have current best here, so we'll just use a few generic starts
    # Actually, we should pass the current best center/angle
    # Let's assume we call this after SA, so we have a good starting point
    # We'll modify signature to accept initial guess
    
    return None # Placeholder, will be implemented in main loop

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    candidates = generate_candidates()
    
    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0
    
    for idx, axial_coords in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        local_positions = build_axial(axial_coords)
        
        # Phase 1: SA Optimization
        best_params_sa, _ = optimize_joint_sa(local_positions, deadline - 10, seed=idx*100+42)
        
        n = len(local_positions)
        best_inner_sa = []
        for i in range(n):
            best_inner_sa.append([best_params_sa[i*3], best_params_sa[i*3+1], best_params_sa[i*3+2]])
            
        best_ocx_sa = best_params_sa[n*3]
        best_ocy_sa = best_params_sa[n*3+1]
        best_oangle_sa = best_params_sa[n*3+2]
        
        # Phase 2: Polish Outer Container
        def obj_outer_polish(params):
            cx, cy, angle = params
            return support_function_exact(best_inner_sa, (cx, cy), angle)
            
        # Multi-start for outer polish
        polish_starts = []
        for da in [-15, -7, 0, 7, 15]:
            for dc in [(-0.2, 0), (0.2, 0), (0, -0.2), (0, 0.2), (0, 0)]:
                polish_starts.append([best_ocx_sa + dc[0], best_ocy_sa + dc[1], best_oangle_sa + da])
                
        best_polish_val = support_function_exact(best_inner_sa, (best_ocx_sa, best_ocy_sa), best_oangle_sa)
        best_polish_params = [best_ocx_sa, best_ocy_sa, best_oangle_sa]
        
        for ps in polish_starts:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(obj_outer_polish, ps, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 4000})
                if res.fun < best_polish_val:
                    best_polish_val = res.fun
                    best_polish_params = res.x
            except Exception:
                pass
        
        current_inner = best_inner_sa
        current_center = (best_polish_params[0], best_polish_params[1])
        current_angle = best_polish_params[2]
        current_side = best_polish_val
        
        # Phase 3: Polish Inner Positions
        if time.time() < deadline - 2:
            polished_inner = polish_inner_positions(current_inner, current_center, current_angle, deadline - 2)
            if check_disjoint(polished_inner, eps=1e-5):
                # Recalculate side with polished inner
                final_side = support_function_exact(polished_inner, current_center, current_angle)
                if final_side < current_side:
                    current_inner = polished_inner
                    current_side = final_side
        
        # Final Validation
        if check_disjoint(current_inner, eps=1e-5):
            if current_side < best_global_side:
                best_global_side = current_side
                best_global_inner = current_inner
                best_global_center = current_center
                best_global_angle = current_angle
        else:
            # Fallback to rigid layout if SA failed disjointness
            cx0 = sum(p[0] for p in local_positions) / n
            cy0 = sum(p[1] for p in local_positions) / n
            fallback_side = support_function_exact(local_positions, (cx0, cy0), 0.0)
            if fallback_side < best_global_side:
                best_global_side = fallback_side
                best_global_inner = local_positions
                best_global_center = (cx0, cy0)
                best_global_angle = 0.0

    if best_global_inner is None:
        # Absolute fallback
        axial_coords = candidates[0][:N]
        centers = [(1.5 * q, SQRT3 * (r + q / 2.0)) for q, r in axial_coords]
        best_global_inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_global_center = (cx_avg, cy_avg)
        best_global_angle = 0.0
        best_global_side = support_function_exact(best_global_inner, best_global_center, best_global_angle)

    return best_global_inner, best_global_center, best_global_side, best_global_angle

# EVOLVE-BLOCK-END

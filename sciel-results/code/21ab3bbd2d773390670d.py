"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

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

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using SAT.
    Returns True if they overlap (including touching).
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

def smooth_penalty(params, outer_center, outer_angle, beta=50.0):
    """
    Smooth penalty for overlap using soft-min of distances.
    Uses log-sum-exp style soft-min for smoothness.
    """
    inner_list = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner_list.append([x, y, a])
    
    # For each pair, compute minimum distance between edges
    # Use a simplified approach: center-to-center distance with angular correction
    penalty = 0.0
    min_center_dist = SQRT3 - 0.005  # Minimum center distance for non-overlapping unit hexes
    
    for i in range(N):
        xi, yi, ai = inner_list[i]
        for j in range(i+1, N):
            xj, yj, aj = inner_list[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            # The actual minimum distance depends on relative orientation
            # For aligned hexes, min center dist = sqrt(3)
            # For rotated, it can be less
            # Use a conservative estimate
            if dist < min_center_dist:
                # Soft penalty: exponential for smoothness
                diff = min_center_dist - dist
                penalty += math.exp(beta * diff) / beta
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
    
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    best_result = None
    best_val = float('inf')
    
    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
    
    for init_angle in init_angles:
        if time.time() > deadline - 30:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            res = minimize(objective_outer, init_params, method='Nelder-Mead', 
                          options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})
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
        # Fallback to grid search
        best_side = float('inf')
        for outer_angle_deg in range(0, 360, 3):
            current_cx = cx_avg
            current_cy = cy_avg
            step = 1.0
            for _ in range(4):
                improved = False
                for dx in np.linspace(-step, step, 7):
                    for dy in np.linspace(-step, step, 7):
                        oc = (current_cx + dx, current_cy + dy)
                        side = compute_required_side(initial_inner, oc, outer_angle_deg)
                        if side < best_side - 1e-7:
                            best_side = side
                            best_outer_center = oc
                            best_outer_angle = outer_angle_deg
                            current_cx = oc[0]
                            current_cy = oc[1]
                            improved = True
                if not improved:
                    break
                step *= 0.4
            if time.time() > deadline - 10:
                break

    # 4. Joint optimization of inner positions and outer parameters
    # Use coordinate descent with scipy polish
    
    def full_objective(params):
        """
        params: [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10, ocx, ocy, oangle]
        """
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
        
        # Smooth overlap penalty
        pen = smooth_penalty(params[:N*3], (ocx, ocy), oangle, beta=80.0)
        
        return side + 50.0 * pen
    
    # Initialize full parameter vector
    full_params = np.array([v for inner in initial_inner for v in inner])
    full_params = np.append(full_params, [best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    current_full = full_params.copy()
    current_val = full_objective(current_full)
    best_full = current_full.copy()
    best_full_val = current_val
    
    # Coordinate descent: optimize inner positions, then outer, alternating
    for iteration in range(40):
        if time.time() > deadline - 20:
            break
        
        # Optimize inner positions (first N*3 params)
        def obj_inner_only(inner_params):
            p = np.concatenate([inner_params, current_full[N*3:]])
            return full_objective(p)
        
        try:
            res_inner = minimize(obj_inner_only, current_full[:N*3], method='Nelder-Mead',
                                options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 800})
            if res_inner.fun < current_val - 1e-8:
                current_full[:N*3] = res_inner.x
                current_val = res_inner.fun
                if current_val < best_full_val:
                    best_full = current_full.copy()
                    best_full_val = current_val
        except Exception:
            pass
        
        if time.time() > deadline - 20:
            break
        
        # Optimize outer parameters (last 3 params)
        def obj_outer_only(outer_params):
            p = np.concatenate([current_full[:N*3], outer_params])
            return full_objective(p)
        
        try:
            res_outer = minimize(obj_outer_only, current_full[N*3:], method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 600})
            if res_outer.fun < current_val - 1e-8:
                current_full[N*3:] = res_outer.x
                current_val = res_outer.fun
                if current_val < best_full_val:
                    best_full = current_full.copy()
                    best_full_val = current_val
        except Exception:
            pass
    
    # Reconstruct best inner and outer
    best_inner_list = []
    for i in range(N):
        x = best_full[i*3]
        y = best_full[i*3+1]
        a = best_full[i*3+2]
        best_inner_list.append([x, y, a])
    
    best_oc = (best_full[N*3], best_full[N*3+1])
    best_ou = best_full[N*3+2]
    
    # Verify with SAT
    if check_all_disjoint(best_inner_list, eps=1e-5):
        best_inner = best_inner_list
        best_outer_center = best_oc
        best_outer_angle = best_ou
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    else:
        # Try to fix overlaps by nudging hexagons apart
        fixed_inner = best_inner_list.copy()
        for _ in range(20):
            if check_all_disjoint(fixed_inner, eps=1e-5):
                break
            for i in range(N):
                for j in range(i+1, N):
                    vi = get_hex_vertices(fixed_inner[i][0], fixed_inner[i][1], 1.0, math.radians(fixed_inner[i][2]))
                    vj = get_hex_vertices(fixed_inner[j][0], fixed_inner[j][1], 1.0, math.radians(fixed_inner[j][2]))
                    if sat_overlap_check(vi, vj, eps=1e-6):
                        # Nudge apart
                        dx = fixed_inner[i][0] - fixed_inner[j][0]
                        dy = fixed_inner[i][1] - fixed_inner[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-9:
                            dx, dy = 1.0, 0.0
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        nudge = 0.01
                        fixed_inner[i] = [fixed_inner[i][0] + nudge*dx, fixed_inner[i][1] + nudge*dy, fixed_inner[i][2]]
                        fixed_inner[j] = [fixed_inner[j][0] - nudge*dx, fixed_inner[j][1] - nudge*dy, fixed_inner[j][2]]
        
        if check_all_disjoint(fixed_inner, eps=1e-5):
            best_inner = fixed_inner
            best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
        else:
            # Fallback to initial if SAT fails
            best_inner = initial_inner
            best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    # 5. Final high-precision polish
    if time.time() < deadline - 5:
        def polish_objective(params):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])
            
            side = compute_required_side(inner_list, best_outer_center, best_outer_angle)
            
            # Very tight penalty
            pen = 0.0
            min_dist = SQRT3 - 0.001
            for i in range(N):
                xi, yi, _ = inner_list[i]
                for j in range(i+1, N):
                    xj, yj, _ = inner_list[j]
                    dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                    if dist < min_dist:
                        pen += (min_dist - dist) ** 2
            return side + 1000.0 * pen
        
        try:
            polish_params = np.array([v for inner in best_inner for v in inner])
            res_polish = minimize(polish_objective, polish_params, method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 3000})
            
            polish_inner = []
            for i in range(N):
                x = res_polish.x[i*3]
                y = res_polish.x[i*3+1]
                a = res_polish.x[i*3+2]
                polish_inner.append([x, y, a])
            
            if check_all_disjoint(polish_inner, eps=1e-6):
                polish_side = compute_required_side(polish_inner, best_outer_center, best_outer_angle)
                if polish_side < best_side - 1e-8:
                    best_inner = polish_inner
                    best_side = polish_side
        except Exception:
            pass
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END

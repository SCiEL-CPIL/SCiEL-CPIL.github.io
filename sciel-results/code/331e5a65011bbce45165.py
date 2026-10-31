import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

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
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
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
        
        # Check for gap
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
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
    # Variables: [cx, cy, angle_deg]
    
    def objective_outer(params):
        cx, cy, angle_deg = params
        side = compute_required_side(initial_inner, (cx, cy), angle_deg)
        return side
    
    # Initial guess: centroid and 0
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Try multiple diverse initial angles to avoid local minima
    best_result = None
    best_val = float('inf')
    
    init_angles = [0, 10, 20, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    for init_angle in init_angles:
        if time.time() > deadline - 15:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            res = minimize(objective_outer, init_params, method='Nelder-Mead', 
                          options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 1500})
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
        # Fallback to grid search if scipy fails or times out
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

    # 4. Optimize inner positions with SAT-based validity checking
    # Use a penalty approach but verify with SAT at the end
    
    def objective_inner(params):
        # Unpack params to inner list
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        side = compute_required_side(inner_list, best_outer_center, best_outer_angle)
        
        # Conservative penalty: center distance < sqrt(3) - margin
        penalty = 0.0
        min_dist = math.sqrt(3) - 0.01  # Safe margin
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, _ = inner_list[i]
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                if dist < min_dist:
                    overlap = min_dist - dist
                    penalty += (overlap * 50.0)**2
                    
        return side + penalty

    if time.time() < deadline - 25:
        # Initialize params from initial_inner
        init_params_inner = []
        for x, y, a in initial_inner:
            init_params_inner.extend([x, y, a])
            
        # Try multiple random perturbations for inner optimization
        best_inner_side = best_side
        best_inner_candidate = best_inner
        
        for trial in range(3):
            if time.time() > deadline - 10:
                break
                
            if trial == 0:
                trial_params = init_params_inner[:]
            else:
                # Small random perturbation
                trial_params = init_params_inner[:]
                for i in range(len(trial_params)):
                    if i % 3 != 2:  # Don't perturb angles randomly
                        trial_params[i] += np.random.uniform(-0.1, 0.1)
            
            try:
                res_inner = minimize(objective_inner, trial_params, method='Nelder-Mead', 
                                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 500})
                
                # Extract result
                inner_list_final = []
                for i in range(N):
                    x = res_inner.x[i*3]
                    y = res_inner.x[i*3+1]
                    a = res_inner.x[i*3+2]
                    inner_list_final.append([x, y, a])
                    
                # Verify with SAT that all hexagons are disjoint
                if check_all_disjoint(inner_list_final):
                    side_final = compute_required_side(inner_list_final, best_outer_center, best_outer_angle)
                    if side_final < best_inner_side - 1e-8:
                        best_inner_side = side_final
                        best_inner_candidate = inner_list_final
            except Exception:
                pass
        
        if best_inner_candidate is not best_inner:
            best_side = best_inner_side
            best_inner = best_inner_candidate

    # 5. Final refinement: joint optimization of inner and outer if time permits
    if time.time() < deadline - 15:
        def objective_joint(params):
            # First N*3 params are inner, last 3 are outer
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])
            
            ox = params[N*3]
            oy = params[N*3+1]
            oa = params[N*3+2]
            
            side = compute_required_side(inner_list, (ox, oy), oa)
            
            # Penalty for overlaps
            penalty = 0.0
            min_dist = math.sqrt(3) - 0.01
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, _ = inner_list[i]
                    xj, yj, _ = inner_list[j]
                    dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                    if dist < min_dist:
                        overlap = min_dist - dist
                        penalty += (overlap * 50.0)**2
                        
            return side + penalty
        
        joint_params = []
        for x, y, a in best_inner:
            joint_params.extend([x, y, a])
        joint_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
        
        try:
            res_joint = minimize(objective_joint, joint_params, method='Nelder-Mead',
                                options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 400})
            
            inner_list_joint = []
            for i in range(N):
                x = res_joint.x[i*3]
                y = res_joint.x[i*3+1]
                a = res_joint.x[i*3+2]
                inner_list_joint.append([x, y, a])
            
            ox_joint = res_joint.x[N*3]
            oy_joint = res_joint.x[N*3+1]
            oa_joint = res_joint.x[N*3+2]
            
            # Verify with SAT
            if check_all_disjoint(inner_list_joint):
                side_joint = compute_required_side(inner_list_joint, (ox_joint, oy_joint), oa_joint)
                if side_joint < best_side - 1e-8:
                    best_side = side_joint
                    best_inner = inner_list_joint
                    best_outer_center = (ox_joint, oy_joint)
                    best_outer_angle = oa_joint
        except Exception:
            pass

    # Add tiny epsilon to side to ensure validity
    best_side += 1e-9

    return best_inner, best_outer_center, best_side, best_outer_angle

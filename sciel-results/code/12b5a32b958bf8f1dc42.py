"""Improved packing of 11 unit hexagons in a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 145.0

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
    Smooth surrogate for compute_required_side using log-sum-exp soft-max.
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
    
    proj_arr = np.array(projections)
    max_proj = float(np.log(np.sum(np.exp(beta * (proj_arr - np.max(proj_arr)))) / beta) + np.max(proj_arr))
    
    return 2.0 * max_proj / SQRT3

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

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.002):
    """Compute a smooth penalty for overlaps based on center distances."""
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
    deadline = start_time + DEADLINE
    
    # Generate multiple candidate initial layouts
    def generate_layout(scale=1.0, epsilon=1e-5):
        """Generate a hexagonal lattice layout."""
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2)
        ]
        
        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        
        s = scale + epsilon
        centers = []
        for q, r in axial_coords:
            x = 1.5 * s * q
            y = SQRT3 * s * (r + q / 2.0)
            centers.append((x, y))
        
        return [[x, y, 0.0] for x, y in centers]
    
    def generate_compact_layout():
        """Generate a more compact, slightly distorted layout."""
        # Center + 6 neighbors + 4 more, but with tighter spacing
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2)
        ]
        
        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        
        # Apply a slight radial compression to make it more compact
        centers = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            # Slight radial compression
            dist = math.sqrt(x*x + y*y)
            if dist > 0.01:
                factor = 0.98
                x *= factor
                y *= factor
            centers.append((x, y))
        
        return [[x, y, 0.0] for x, y in centers]
    
    def generate_radial_layout():
        """Generate a layout with radial symmetry."""
        # Center hexagon
        centers = [(0.0, 0.0)]
        
        # 6 hexagons in a ring
        ring_dist = SQRT3 * 0.99
        for i in range(6):
            angle = math.pi / 6 + i * math.pi / 3
            x = ring_dist * math.cos(angle)
            y = ring_dist * math.sin(angle)
            centers.append((x, y))
        
        # 4 more hexagons in an outer ring
        outer_dist = SQRT3 * 1.7
        for i in range(4):
            angle = math.pi / 3 + i * math.pi / 2 + math.pi / 6
            x = outer_dist * math.cos(angle)
            y = outer_dist * math.sin(angle)
            centers.append((x, y))
        
        return [[x, y, 0.0] for x, y in centers[:N]]
    
    # Try multiple initial layouts
    initial_layouts = [
        generate_layout(1.0),
        generate_compact_layout(),
        generate_radial_layout(),
    ]
    
    best_side = float('inf')
    best_inner = initial_layouts[0]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Optimize each initial layout
    for layout_idx, initial_inner in enumerate(initial_layouts):
        if time.time() > deadline - 30:
            break
        
        cx_avg = sum(c[0] for c in initial_inner) / len(initial_inner)
        cy_avg = sum(c[1] for c in initial_inner) / len(initial_inner)
        
        # Phase 1: Optimize outer container with fixed inner layout
        best_val = float('inf')
        best_result = None
        
        init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
        
        for init_angle in init_angles:
            if time.time() > deadline - 25:
                break
            
            init_params = [cx_avg, cy_avg, float(init_angle)]
            
            try:
                # First pass with smooth objective
                def objective_outer_smooth(params):
                    cx, cy, angle_deg = params
                    return compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=80.0)
                
                def objective_outer_exact(params):
                    cx, cy, angle_deg = params
                    return compute_required_side(initial_inner, (cx, cy), angle_deg)
                
                res = minimize(objective_outer_smooth, init_params, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 2000})
                
                # Second pass with exact objective
                res2 = minimize(objective_outer_exact, res.x, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 3000})
                
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_result = res2.x
            except Exception:
                pass
        
        if best_result is None:
            continue
        
        cur_outer_center = (best_result[0], best_result[1])
        cur_outer_angle = best_result[2]
        cur_side = best_val
        
        # Phase 2: Joint optimization
        def full_objective(params):
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
            pen = compute_overlap_penalty(inner_list, min_dist=SQRT3 - 0.002)
            return side + 50.0 * pen
        
        full_params = np.array([v for inner in initial_inner for v in inner])
        full_params = np.append(full_params, [cur_outer_center[0], cur_outer_center[1], cur_outer_angle])
        
        current_full = full_params.copy()
        current_val = full_objective(current_full)
        best_full = current_full.copy()
        best_full_val = current_val
        
        # Coordinate descent
        for iteration in range(20):
            if time.time() > deadline - 15:
                break
            
            # Optimize inner positions
            def obj_inner_only(inner_params):
                p = np.concatenate([inner_params, current_full[N*3:]])
                return full_objective(p)
            
            try:
                res_inner = minimize(obj_inner_only, current_full[:N*3], method='Nelder-Mead',
                                    options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 600})
                if res_inner.fun < current_val - 1e-8:
                    current_full[:N*3] = res_inner.x
                    current_val = res_inner.fun
                    if current_val < best_full_val:
                        best_full = current_full.copy()
                        best_full_val = current_val
            except Exception:
                pass
            
            if time.time() > deadline - 15:
                break
            
            # Optimize outer parameters
            def obj_outer_only(outer_params):
                p = np.concatenate([current_full[:N*3], outer_params])
                return full_objective(p)
            
            try:
                res_outer = minimize(obj_outer_only, current_full[N*3:], method='Nelder-Mead',
                                    options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 400})
                if res_outer.fun < current_val - 1e-8:
                    current_full[N*3:] = res_outer.x
                    current_val = res_outer.fun
                    if current_val < best_full_val:
                        best_full = current_full.copy()
                        best_full_val = current_val
            except Exception:
                pass
        
        # Reconstruct
        best_inner_list = []
        for i in range(N):
            x = best_full[i*3]
            y = best_full[i*3+1]
            a = best_full[i*3+2]
            best_inner_list.append([x, y, a])
        
        best_oc = (best_full[N*3], best_full[N*3+1])
        best_ou = best_full[N*3+2]
        
        # Verify and fix if needed
        if check_all_disjoint(best_inner_list, eps=1e-5):
            cur_side = compute_required_side(best_inner_list, best_oc, best_ou)
            if cur_side < best_side:
                best_side = cur_side
                best_inner = best_inner_list
                best_outer_center = best_oc
                best_outer_angle = best_ou
        else:
            # Try to fix overlaps
            fixed_inner = best_inner_list.copy()
            for _ in range(30):
                if check_all_disjoint(fixed_inner, eps=1e-5):
                    break
                for i in range(N):
                    for j in range(i+1, N):
                        vi = get_hex_vertices(fixed_inner[i][0], fixed_inner[i][1], 1.0, math.radians(fixed_inner[i][2]))
                        vj = get_hex_vertices(fixed_inner[j][0], fixed_inner[j][1], 1.0, math.radians(fixed_inner[j][2]))
                        if sat_overlap_check(vi, vj, eps=1e-6):
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
                cur_side = compute_required_side(fixed_inner, best_oc, best_ou)
                if cur_side < best_side:
                    best_side = cur_side
                    best_inner = fixed_inner
                    best_outer_center = best_oc
                    best_outer_angle = best_ou
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5 and best_inner is not None:
        def polish_objective(params):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])
            
            side = compute_required_side(inner_list, best_outer_center, best_outer_angle)
            
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

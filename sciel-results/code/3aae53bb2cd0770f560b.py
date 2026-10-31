"""Rewritten program with stronger joint optimization for hexagon packing."""
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

def compute_required_side_vec(inner, outer_center, outer_angle_deg):
    """
    Vectorized computation of minimum outer side length.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Outward normals for unit hexagon (vertices at 0,60,...) are at 30,90,...
    normals = np.array([[math.cos(math.radians(30 + i*60)), 
                         math.sin(math.radians(30 + i*60))] for i in range(6)])
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    max_proj = 0.0
    for (ix, iy, iangle) in inner:
        # Get vertices
        angle_rad = math.radians(iangle)
        cos_i = math.cos(angle_rad)
        sin_i = math.sin(angle_rad)
        # Unit hex vertices
        base_vx = np.array([math.cos(i * math.pi/3) for i in range(6)])
        base_vy = np.array([math.sin(i * math.pi/3) for i in range(6)])
        vx = ix + base_vx
        vy = iy + base_vy
        
        # Translate to outer center
        dx = vx - ox
        dy = vy - oy
        
        # Rotate by -outer_angle
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        
        # Project onto all 6 normals
        projs = lx[:, None] * normals[0] + ly[:, None] * normals[1]
        m = np.max(projs)
        if m > max_proj:
            max_proj = m
    
    return 2.0 * max_proj / SQRT3

def compute_required_side(inner, outer_center, outer_angle_deg):
    """Compute the minimum outer side length to contain all inner hexagons."""
    return compute_required_side_vec(inner, outer_center, outer_angle_deg)

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
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
        min1 = min(p[0]*axis[0] + p[1]*axis[1] for p in v1)
        max1 = max(p[0]*axis[0] + p[1]*axis[1] for p in v1)
        min2 = min(p[0]*axis[0] + p[1]*axis[1] for p in v2)
        max2 = max(p[0]*axis[0] + p[1]*axis[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    verts_list = []
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        verts_list.append(vi)
    for i in range(n):
        for j in range(i+1, n):
            if sat_overlap_check(verts_list[i], verts_list[j], eps):
                return False
    return True

def full_objective(params, N=N):
    """
    Full objective: outer side length + overlap penalty.
    params: [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10, ocx, ocy, oangle]
    """
    inner_list = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner_list.append((x, y, a))
    
    ocx = params[N*3]
    ocy = params[N*3+1]
    oangle = params[N*3+2]
    
    side = compute_required_side(inner_list, (ocx, ocy), oangle)
    
    # Smooth overlap penalty using center distances
    # For unit hexagons, minimum center distance depends on relative orientation
    # Conservative: use sqrt(3) - small margin
    penalty = 0.0
    min_center_dist = SQRT3 - 0.01
    for i in range(N):
        xi, yi, _ = inner_list[i]
        for j in range(i+1, N):
            xj, yj, _ = inner_list[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < min_center_dist:
                diff = min_center_dist - dist
                penalty += diff * diff
    
    return side + 200.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Axial coordinates for hexagonal packing
    # Center + 6 neighbors + 4 more for 11 total
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    def axial_to_cartesian(q, r, scale=1.0):
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        return (x, y)
    
    # Generate initial configurations
    def make_initial_config(scale=1.0, angle_offset=0.0):
        centers = [axial_to_cartesian(q, r, scale) for q, r in axial_coords]
        inner = [[x, y, angle_offset] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        return inner, (cx_avg, cy_avg), 0.0
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Multiple initial configurations to try
    initial_configs = [
        (1.0, 0.0),      # Standard
        (1.0, 30.0),     # Rotated inner
        (1.0, 15.0),     # Half-rotated
        (0.999, 0.0),    # Slightly compressed
        (1.001, 0.0),    # Slightly expanded
    ]
    
    for scale, angle_offset in initial_configs:
        if time.time() > deadline - 40:
            break
        
        inner_init, oc_init, oa_init = make_initial_config(scale, angle_offset)
        
        # Quick outer optimization first
        def obj_outer(params):
            cx, cy, angle = params
            return compute_required_side(inner_init, (cx, cy), angle)
        
        best_outer_val = float('inf')
        best_outer_params = [oc_init[0], oc_init[1], oa_init]
        
        # Try several angle starts
        for init_angle in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
            if time.time() > deadline - 35:
                break
            init_p = [oc_init[0], oc_init[1], float(init_angle)]
            try:
                res = minimize(obj_outer, init_p, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 500})
                if res.fun < best_outer_val:
                    best_outer_val = res.fun
                    best_outer_params = res.x
            except Exception:
                pass
        
        # Now joint optimization
        full_params = np.array([v for inner in inner_init for v in inner] + 
                               [best_outer_params[0], best_outer_params[1], best_outer_params[2]])
        
        current_val = full_objective(full_params)
        
        # Coordinate descent with multiple iterations
        for it in range(30):
            if time.time() > deadline - 15:
                break
            
            # Optimize inner
            def obj_inner(ip):
                p = np.concatenate([ip, full_params[N*3:]])
                return full_objective(p)
            
            try:
                res_i = minimize(obj_inner, full_params[:N*3], method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 600})
                if res_i.fun < current_val - 1e-9:
                    full_params[:N*3] = res_i.x
                    current_val = res_i.fun
            except Exception:
                pass
            
            # Optimize outer
            def obj_outer2(op):
                p = np.concatenate([full_params[:N*3], op])
                return full_objective(p)
            
            try:
                res_o = minimize(obj_outer2, full_params[N*3:], method='Nelder-Mead',
                                options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 400})
                if res_o.fun < current_val - 1e-9:
                    full_params[N*3:] = res_o.x
                    current_val = res_o.fun
            except Exception:
                pass
        
        # Extract result
        inner_list = []
        for i in range(N):
            inner_list.append([full_params[i*3], full_params[i*3+1], full_params[i*3+2]])
        oc = (full_params[N*3], full_params[N*3+1])
        oa = full_params[N*3+2]
        
        # Verify and compute true side
        if check_all_disjoint(inner_list, eps=1e-5):
            side = compute_required_side(inner_list, oc, oa)
            if side < best_side:
                best_side = side
                best_inner = inner_list
                best_outer_center = oc
                best_outer_angle = oa
        else:
            # Try to fix overlaps
            fixed = [list(h) for h in inner_list]
            for _ in range(30):
                if check_all_disjoint(fixed, eps=1e-5):
                    break
                for i in range(N):
                    for j in range(i+1, N):
                        vi = get_hex_vertices(fixed[i][0], fixed[i][1], 1.0, math.radians(fixed[i][2]))
                        vj = get_hex_vertices(fixed[j][0], fixed[j][1], 1.0, math.radians(fixed[j][2]))
                        if sat_overlap_check(vi, vj, eps=1e-6):
                            dx = fixed[i][0] - fixed[j][0]
                            dy = fixed[i][1] - fixed[j][1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist < 1e-9:
                                dx, dy = 1.0, 0.0
                                dist = 1.0
                            else:
                                dx /= dist
                                dy /= dist
                            nudge = 0.02
                            fixed[i][0] += nudge*dx
                            fixed[i][1] += nudge*dy
                            fixed[j][0] -= nudge*dx
                            fixed[j][1] -= nudge*dy
                
                if check_all_disjoint(fixed, eps=1e-5):
                    side = compute_required_side(fixed, oc, oa)
                    if side < best_side:
                        best_side = side
                        best_inner = fixed
                        best_outer_center = oc
                        best_outer_angle = oa
                    break
    
    # Final polish if we have a valid solution
    if best_inner is not None and time.time() < deadline - 5:
        def polish_obj(params):
            inner_list = []
            for i in range(N):
                inner_list.append((params[i*3], params[i*3+1], params[i*3+2]))
            side = compute_required_side(inner_list, best_outer_center, best_outer_angle)
            pen = 0.0
            min_d = SQRT3 - 0.005
            for i in range(N):
                xi, yi, _ = inner_list[i]
                for j in range(i+1, N):
                    xj, yj, _ = inner_list[j]
                    d = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                    if d < min_d:
                        pen += (min_d - d)**2
            return side + 1000.0 * pen
        
        try:
            pp = np.array([v for h in best_inner for v in h])
            res = minimize(polish_obj, pp, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2500})
            polish_inner = []
            for i in range(N):
                polish_inner.append([res.x[i*3], res.x[i*3+1], res.x[i*3+2]])
            if check_all_disjoint(polish_inner, eps=1e-6):
                ps = compute_required_side(polish_inner, best_outer_center, best_outer_angle)
                if ps < best_side - 1e-9:
                    best_inner = polish_inner
                    best_side = ps
        except Exception:
            pass
    
    # Fallback if no valid solution found
    if best_inner is None:
        inner_init, oc_init, oa_init = make_initial_config(1.0, 0.0)
        best_inner = inner_init
        best_outer_center = oc_init
        best_outer_angle = oa_init
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END

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
    # Outward normals for a standard hexagon (vertices at 0, 60, ...):
    # Normals at 30, 90, 150, 210, 270, 330 degrees.
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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth version of compute_required_side using log-sum-exp for the max.
    This makes the objective differentiable for gradient-based methods.
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
    
    # Collect all projections
    projs = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projs.append(proj)
    
    projs = np.array(projs)
    # Log-sum-exp for smooth max
    m = np.max(projs)
    if m < 0:
        m = 0.0
    log_sum_exp = m + math.log(np.sum(np.exp((projs - m) * beta)) / len(projs)) / beta
    max_proj_smooth = log_sum_exp / beta * beta  # Actually just use: m + log(sum(exp((p-m)*beta)))/beta
    # Correct log-sum-exp:
    shifted = projs - m
    log_sum_exp_val = m + math.log(np.sum(np.exp(shifted * beta))) / beta
    max_proj_smooth = log_sum_exp_val
    
    return 2.0 * max_proj_smooth / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
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

def axial_to_cartesian(q, r):
    """Convert axial hex coordinates to Cartesian."""
    x = 1.5 * q
    y = SQRT3 * (r + q / 2.0)
    return (x, y)

def generate_lattice_configs():
    """Generate multiple compact hex lattice configurations for 11 hexagons."""
    configs = []
    
    # Config 1: Flower + 4 extensions (center, 6 neighbors, 4 at distance 2)
    # Center + ring of 6 + 4 more
    flower_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    configs.append(flower_coords)
    
    # Config 2: Row-based 4-3-4
    # Row 0: 4 hexes, Row 1: 3 hexes, Row 2: 4 hexes
    row_434 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    configs.append(row_434)
    
    # Config 3: 3-5-3 arrangement
    row_353 = [
        (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    configs.append(row_353)
    
    # Config 4: 5-3-3 arrangement
    row_533 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    configs.append(row_533)
    
    # Config 5: Variant flower with different extensions
    flower_v2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (2, -1), (0, 2)
    ]
    configs.append(flower_v2)
    
    # Config 6: Compact diamond-like
    diamond = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-1, 2), (1, 1), (0, 2)
    ]
    configs.append(diamond)
    
    return configs

def center_config(coords):
    """Center a configuration so its centroid is at origin."""
    points = [axial_to_cartesian(q, r) for q, r in coords]
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    centered = [(p[0] - cx, p[1] - cy) for p in points]
    return centered

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Generate lattice configurations
    configs = generate_lattice_configs()
    
    # For each config, create Cartesian positions (slightly scaled to avoid touching)
    epsilon = 1e-5
    scale = 1.0 + epsilon
    
    all_candidates = []
    for coords in configs:
        points = []
        for q, r in coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            points.append((x, y))
        
        # Center the configuration
        cx_avg = sum(p[0] for p in points) / len(points)
        cy_avg = sum(p[1] for p in points) / len(points)
        centered_points = [(p[0] - cx_avg, p[1] - cy_avg) for p in points]
        
        all_candidates.append(centered_points)
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # For each candidate arrangement, optimize the outer hexagon
    for candidate_points in all_candidates:
        if time.time() > deadline - 40:
            break
        
        # Initial layout: All hexagons aligned (angle 0)
        initial_inner = [[x, y, 0.0] for x, y in candidate_points]
        
        # Optimize outer hexagon parameters (center x, center y, angle)
        def objective_outer(params):
            cx, cy, angle_deg = params
            side = compute_required_side(initial_inner, (cx, cy), angle_deg)
            return side
        
        best_result = None
        best_val = float('inf')
        
        # Try multiple initial angles
        init_angles = [0, 30, 60, 90, 120, 150]
        
        for init_angle in init_angles:
            if time.time() > deadline - 30:
                break
                
            init_params = [0.0, 0.0, float(init_angle)]
            
            try:
                res = minimize(objective_outer, init_params, method='Nelder-Mead', 
                              options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 15000})
                if res.fun < best_val:
                    best_val = res.fun
                    best_result = res.x
            except Exception:
                pass
        
        if best_result is not None:
            cur_side = best_val
            if cur_side < best_side:
                best_side = cur_side
                best_inner = initial_inner
                best_outer_center = (best_result[0], best_result[1])
                best_outer_angle = best_result[2]
    
    # If no valid arrangement was found, use the first config as fallback
    if best_inner is None:
        points = all_candidates[0]
        best_inner = [[x, y, 0.0] for x, y in points]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    # Now do a joint optimization: optimize inner positions and rotations along with outer
    # This is the key improvement - allow inner hexagons to move and rotate
    
    if time.time() < deadline - 20:
        # Build joint optimization
        # Parameters: for each inner hex: x, y, angle; plus outer: cx, cy, angle
        # Total: 3*N + 3 = 36 parameters for N=11
        
        n_inner = len(best_inner)
        
        def joint_objective(params):
            # Extract inner positions and outer parameters
            inner = []
            for i in range(n_inner):
                ix = params[i*3]
                iy = params[i*3 + 1]
                iangle = params[i*3 + 2]
                inner.append((ix, iy, iangle))
            
            ocx = params[n_inner*3]
            ocy = params[n_inner*3 + 1]
            oangle = params[n_inner*3 + 2]
            
            # Compute required side
            side = compute_required_side(inner, (ocx, ocy), oangle)
            
            # Add penalty for overlapping inner hexagons
            # Use a soft penalty based on minimum distance
            penalty = 0.0
            for i in range(n_inner):
                for j in range(i+1, n_inner):
                    dx = inner[i][0] - inner[j][0]
                    dy = inner[i][1] - inner[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Minimum distance for non-overlapping unit hexagons is 2 (center to center)
                    # Actually for hexagons, the minimum center distance depends on orientation
                    # Use a conservative bound: 2.0
                    min_dist = 2.0
                    if dist < min_dist:
                        # Soft penalty
                        penalty += (min_dist - dist) ** 2 * 100.0
            
            return side + penalty
        
        # Initial parameters from current best
        init_params = []
        for (ix, iy, iangle) in best_inner:
            init_params.extend([ix, iy, iangle])
        init_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
        
        try:
            res = minimize(joint_objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 20000})
            
            if res.fun < best_side:
                # Extract results
                new_inner = []
                for i in range(n_inner):
                    new_inner.append([res.x[i*3], res.x[i*3+1], res.x[i*3+2]])
                
                new_ocx = res.x[n_inner*3]
                new_ocy = res.x[n_inner*3 + 1]
                new_oangle = res.x[n_inner*3 + 2]
                
                # Verify no overlaps
                if check_all_disjoint(new_inner, eps=1e-5):
                    best_inner = new_inner
                    best_outer_center = (new_ocx, new_ocy)
                    best_outer_angle = new_oangle
                    best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
        except Exception:
            pass
    
    # Final high-precision polish on outer parameters only
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    # Try a few starting points for polish
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], best_outer_angle],
        [best_outer_center[0] + 0.01, best_outer_center[1], best_outer_angle + 1.0],
        [best_outer_center[0], best_outer_center[1] + 0.01, best_outer_angle - 1.0],
    ]
    
    for ps in polish_starts:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 15000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 30000})
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

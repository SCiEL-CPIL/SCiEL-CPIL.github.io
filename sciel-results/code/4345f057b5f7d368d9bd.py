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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    This provides better gradient information for optimization.
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
    
    # Log-sum-exp for smooth max
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    # Smooth max via log-sum-exp: (1/beta) * log(sum(exp(beta * p_i)))
    # This approximates max(p_i) as beta -> infinity
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

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
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Generate candidate arrangements from hex lattice
    # We'll use a pool of axial coordinates within a reasonable range
    # and enumerate connected subsets of size N
    
    # Pool of axial coordinates (q, r) where hex distance from origin <= 3
    axial_pool = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if max(abs(q), abs(r), abs(q+r)) <= 3:
                axial_pool.append((q, r))
    
    # Convert axial to Cartesian
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return (x, y)
    
    # Build adjacency: two points are adjacent if hex distance = 1
    def hex_dist(q1, r1, q2, r2):
        return max(abs(q1-q2), abs(r1-r2), abs((q1+r1)-(q2+r2)))
    
    # Generate candidate clusters: all subsets of size N that are connected
    # To make this tractable, we'll use a heuristic: start from center and grow
    # But first, let's try a few known good arrangements
    
    # Known good arrangement: center + 6 neighbors + 4 more in a compact shape
    # Axial coords for a compact 11-hex cluster:
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (0,2), (-1,2)
    original_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    # Alternative compact arrangement
    alt_coords_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (2, -1), (0, 2)
    ]
    
    alt_coords_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-1, 2), (1, 1), (0, 2)
    ]
    
    # Generate more candidates by permuting the outer 4 positions
    # The 6 neighbors are fixed, we vary the 4 additional positions
    # from the set of points at hex distance 2 from origin
    
    # Points at hex distance exactly 2 from origin:
    dist2_points = [(q, r) for q, r in axial_pool if max(abs(q), abs(r), abs(q+r)) == 2]
    
    # Generate all combinations of 4 from dist2_points, but only keep those
    # that form a connected cluster with the center+ring
    # For efficiency, we'll just try a subset of promising combinations
    
    candidate_arrangements = [original_coords, alt_coords_1, alt_coords_2]
    
    # Add a few more by sampling combinations
    import random
    random.seed(42)
    
    # Try to add more diverse candidates
    for _ in range(20):
        if time.time() > deadline - 100:
            break
        
        # Pick 4 random points from dist2_points
        if len(dist2_points) >= 4:
            outer4 = random.sample(dist2_points, 4)
            candidate = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + outer4
            
            # Check if all points are in the pool
            valid = all(p in axial_pool for p in candidate)
            if valid and len(set(candidate)) == N:
                candidate_arrangements.append(candidate)
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # For each candidate arrangement, optimize the outer hexagon
    for axial_coords in candidate_arrangements:
        if time.time() > deadline - 30:
            break
        
        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        
        # Convert axial to Cartesian with small scale to avoid touching
        epsilon = 1e-5
        scale = 1.0 + epsilon
        
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
            
        # Initial layout: All hexagons aligned (angle 0)
        initial_inner = [[x, y, 0.0] for x, y in centers]
        
        # Calculate centroid
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        
        # Optimize outer hexagon parameters
        def objective_outer_smooth(params):
            cx, cy, angle_deg = params
            side = compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=50.0)
            return side
        
        def objective_outer_exact(params):
            cx, cy, angle_deg = params
            side = compute_required_side(initial_inner, (cx, cy), angle_deg)
            return side
        
        best_result = None
        best_val = float('inf')
        
        # Try multiple initial angles
        init_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        
        for init_angle in init_angles:
            if time.time() > deadline - 20:
                break
                
            init_params = [cx_avg, cy_avg, float(init_angle)]
            
            try:
                # First pass: smooth optimization for basin finding
                res_smooth = minimize(objective_outer_smooth, init_params, method='Nelder-Mead', 
                                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 1000})
                
                # Second pass: exact optimization from the smooth result
                if res_smooth.fun < best_val:
                    res_exact = minimize(objective_outer_exact, res_smooth.x, method='Nelder-Mead', 
                                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 5000})
                    if res_exact.fun < best_val:
                        best_val = res_exact.fun
                        best_result = res_exact.x
                    else:
                        best_val = res_smooth.fun
                        best_result = res_smooth.x
            except Exception:
                pass
                
        if best_result is not None:
            cur_side = best_val
            if cur_side < best_side:
                best_side = cur_side
                best_inner = initial_inner
                best_outer_center = (best_result[0], best_result[1])
                best_outer_angle = best_result[2]
    
    # If no valid arrangement was found, use the original as fallback
    if best_inner is None:
        axial_coords = original_coords[:N]
        epsilon = 1e-5
        scale = 1.0 + epsilon
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
        best_inner = [[x, y, 0.0] for x, y in centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    # Final polish: tighten the outer container
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], best_outer_angle],
    ]
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    for ps in polish_starts:
        if time.time() > deadline - 2:
            break
        try:
            res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 20000})
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

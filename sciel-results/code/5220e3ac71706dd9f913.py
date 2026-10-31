"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon (flat-top aligned with angle_rad)."""
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
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    # 1. Define 11 hexagons on a hexagonal lattice - use tighter, more symmetric arrangement
    # Center hexagon + 6 around it + 4 more in a compact cluster
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian with tight packing
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
    # Use smooth surrogate for better optimization landscape
    
    def objective_outer_smooth(params):
        cx, cy, angle_deg = params
        side = compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=50.0)
        return side
    
    def objective_outer_exact(params):
        cx, cy, angle_deg = params
        side = compute_required_side(initial_inner, (cx, cy), angle_deg)
        return side
    
    # Initial guess: centroid and 0
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Try multiple diverse initial angles to avoid local minima
    best_result = None
    best_val = float('inf')
    
    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
    
    for init_angle in init_angles:
        if time.time() > deadline - 20:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            # First pass: smooth optimization for basin finding
            res_smooth = minimize(objective_outer_smooth, init_params, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 1000})
            
            # Second pass: exact optimization from the smooth result
            res_exact = minimize(objective_outer_exact, res_smooth.x, method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000})
            
            val = res_exact.fun
            if val < best_val:
                best_val = val
                best_result = res_exact.x
        except Exception:
            continue
    
    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)
        best_inner = [list(h) for h in initial_inner]
    else:
        # Fallback: use centroid and 0 angle
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)
        best_inner = [list(h) for h in initial_inner]
    
    # 4. Joint optimization: optimize inner positions AND outer container simultaneously
    # This is the key structural improvement - don't fix the inner layout
    
    if time.time() < deadline - 30:
        # Parameterize: 3*N inner params (x, y, angle) + 3 outer params (cx, cy, angle)
        # Total: 3*N + 3 = 36 parameters
        
        def joint_objective_smooth(params):
            # Unpack
            inner_params = params[:3*N]
            outer_params = params[3*N:]
            
            inner = []
            for i in range(N):
                x = inner_params[3*i]
                y = inner_params[3*i+1]
                angle = inner_params[3*i+2]
                inner.append([x, y, angle])
            
            cx, cy, angle_deg = outer_params
            
            # Check for overlaps (penalty)
            overlap_penalty = 0.0
            for i in range(N):
                vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
                for j in range(i + 1, N):
                    vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
                    if sat_overlap_check(vi, vj, eps=-1e-4):  # Negative eps means we allow slight touching
                        overlap_penalty += 100.0
            
            side = compute_required_side_smooth(inner, (cx, cy), angle_deg, beta=30.0)
            return side + overlap_penalty
        
        def joint_objective_exact(params):
            inner_params = params[:3*N]
            outer_params = params[3*N:]
            
            inner = []
            for i in range(N):
                x = inner_params[3*i]
                y = inner_params[3*i+1]
                angle = inner_params[3*i+2]
                inner.append([x, y, angle])
            
            cx, cy, angle_deg = outer_params
            
            # Check for overlaps (penalty)
            overlap_penalty = 0.0
            for i in range(N):
                vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
                for j in range(i + 1, N):
                    vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
                    if sat_overlap_check(vi, vj, eps=1e-6):
                        overlap_penalty += 1000.0
            
            side = compute_required_side(inner, (cx, cy), angle_deg)
            return side + overlap_penalty
        
        # Initialize from current best
        x0 = []
        for i in range(N):
            x0.extend([best_inner[i][0], best_inner[i][1], best_inner[i][2]])
        x0.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
        
        # Try a few random restarts
        np.random.seed(42)
        
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            
            if restart == 0:
                x_start = x0[:]
            else:
                x_start = x0 + np.random.randn(3*N + 3) * 0.3
            
            try:
                # Smooth phase
                res_smooth = minimize(joint_objective_smooth, x_start, method='Nelder-Mead',
                                      options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6, 'maxfev': 800})
                
                # Exact phase
                res_exact = minimize(joint_objective_exact, res_smooth.x, method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'maxfev': 1500})
                
                # Extract and validate
                inner_params = res_exact.x[:3*N]
                outer_params = res_exact.x[3*N:]
                
                candidate_inner = []
                for i in range(N):
                    candidate_inner.append([inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]])
                
                candidate_center = (outer_params[0], outer_params[1])
                candidate_angle = outer_params[2]
                
                # Check validity
                if check_all_disjoint(candidate_inner):
                    candidate_side = compute_required_side(candidate_inner, candidate_center, candidate_angle)
                    if candidate_side < best_side:
                        best_side = candidate_side
                        best_inner = candidate_inner
                        best_outer_center = candidate_center
                        best_outer_angle = candidate_angle
            except Exception:
                continue
    
    # 5. Final polish: fine-tune outer container with the best inner layout
    if time.time() < deadline - 5:
        def final_objective(params):
            cx, cy, angle_deg = params
            return compute_required_side(best_inner, (cx, cy), angle_deg)
        
        try:
            res_final = minimize(final_objective, [best_outer_center[0], best_outer_center[1], best_outer_angle],
                                 method='Nelder-Mead',
                                 options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 5000})
            
            if res_final.fun < best_side:
                best_side = res_final.fun
                best_outer_center = (res_final.x[0], res_final.x[1])
                best_outer_angle = res_final.x[2]
        except Exception:
            pass
    
    # Ensure strict containment with tiny epsilon
    best_side *= (1.0 + 1e-6)
    
    # Format output
    inner = [[best_inner[i][0], best_inner[i][1], best_inner[i][2]] for i in range(N)]
    return inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END

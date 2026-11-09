"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (evaluator limit 120s, keep 10s buffer)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        # Flat-top hexagon vertices at angles 0, 60, 120, 180, 240, 300
        # Rotated by angle_deg
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Check if two hexagons overlap
    # For convex polygons, they overlap if an edge of one separates the other
    # Or simply: check if any vertex of A is inside B OR any vertex of B is inside A
    # For regular hexagons, a robust check is point-in-polygon for vertices.
    # To be safe and fast, we can also check edge intersections, but point-in-poly is often sufficient for disjointness if we assume general position.
    # Actually, for strict disjointness, we need to ensure no intersection.
    # A simpler proxy for optimization: distance between centers > min distance.
    # But min distance depends on orientation.
    # Let's use a robust geometric check: 
    # Two convex polygons are disjoint if there is a separating line.
    # Separating Axis Theorem (SAT) is perfect here.
    
    def get_axes(poly):
        axes = []
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            # Normal to edge
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Normal is (-dy, dx)
            norm = math.sqrt(dx*dx + dy*dy)
            if norm == 0: continue
            nx = -dy / norm
            ny = dx / norm
            axes.append((nx, ny))
        return axes

    def project(poly, axis):
        dots = [p[0]*axis[0] + p[1]*axis[1] for p in poly]
        return min(dots), max(dots)

    def is_disjoint(poly1, poly2):
        axes = get_axes(poly1) + get_axes(poly2)
        for axis in axes:
            min1, max1 = project(poly1, axis)
            min2, max2 = project(poly2, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False

    # Helper: Calculate minimum outer hexagon side length containing all given hexagons
    def calc_outer_side(hexagons, outer_angle_deg):
        # Outer hexagon is regular, centered at origin (we will center the cluster first)
        # We need to find the smallest S such that all vertices of all inner hexagons
        # are inside the outer hexagon.
        
        # Outer hexagon normals (flat-top, rotated by outer_angle_deg)
        # Sides are perpendicular to normals at 30, 90, 150, 210, 270, 330 + outer_angle
        base_angles = [30, 90, 150, 210, 270, 330]
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for a in base_angles:
                    rad = math.radians(a + outer_angle_deg)
                    nx = math.cos(rad)
                    ny = math.sin(rad)
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # Apothem of outer hex is S * SQRT3 / 2
        # Max projection must be <= Apothem
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    # Objective function: Minimize outer side length, penalize overlaps
    def objective(params):
        # params: [x1, y1, a1, x2, y2, a2, ...]
        # Extract positions and angles
        centers = []
        angles = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            centers.append((x, y))
            angles.append(a)
        
        # Center the configuration at origin to remove translational degrees of freedom
        # This helps the optimizer and ensures the outer hexagon is centered
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        # Calculate vertices for all hexagons
        hexagons = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
        
        # Check overlaps
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(hexagons[i], hexagons[j]):
                    # Calculate a penalty based on penetration depth?
                    # For now, a large fixed penalty if overlapping
                    overlap_penalty += 100.0
                    # Early exit if too many overlaps?
        
        if overlap_penalty > 0:
            return overlap_penalty + calc_outer_side(hexagons, 0.0)
            
        # Find best outer angle (coarse search)
        best_side = float('inf')
        for oa_deg in range(0, 60, 5):
            side = calc_outer_side(hexagons, oa_deg)
            if side < best_side:
                best_side = side
        
        return best_side

    # Initial setup: Start with a loose lattice configuration
    # Use axial coordinates to generate initial positions
    def axial_to_cartesian(q, r, spacing=2.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Base 7 hexagons + 4 more
    base_7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # Add 4 in a way that is likely disjoint
    extra = [(2,0), (2,-1), (2,1), (1,1)]
    coords = base_7 + extra
    
    initial_params = []
    for q, r in coords:
        x, y = axial_to_cartesian(q, r, spacing=2.0) # Loose start
        initial_params.extend([x, y, 0.0])
        
    initial_params = np.array(initial_params)
    
    # Use Nelder-Mead from scipy.optimize
    from scipy.optimize import minimize
    
    best_result = None
    best_score = float('inf')
    
    # Run multiple restarts with different initial configurations or perturbations
    num_restarts = 5
    restarts_done = 0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        if restart == 0:
            x0 = initial_params
        else:
            # Perturb the initial configuration
            x0 = initial_params + np.random.normal(0, 0.5, len(initial_params))
            # Randomize angles slightly
            for i in range(N):
                x0[i*3+2] = random.uniform(-30, 30)
                
        try:
            # Nelder-Mead is good for this kind of non-smooth problem (due to min in objective)
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.success or res.fun < best_score:
                # Validate the result
                final_params = res.x
                centers = []
                angles = []
                for i in range(N):
                    x = final_params[i*3]
                    y = final_params[i*3+1]
                    a = final_params[i*3+2]
                    centers.append((x, y))
                    angles.append(a)
                    
                cx_avg = sum(c[0] for c in centers) / N
                cy_avg = sum(c[1] for c in centers) / N
                shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
                
                hexagons = []
                for i in range(N):
                    cx, cy = shifted_centers[i]
                    a = angles[i]
                    hex_verts = get_hex_vertices(cx, cy, a)
                    hexagons.append(hex_verts)
                    
                # Check disjointness again
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not is_disjoint(hexagons[i], hexagons[j]):
                            valid = False
                            break
                    if not valid: break
                            
                if valid:
                    # Find best outer angle
                    best_side = float('inf')
                    best_oa = 0.0
                    for oa_deg in range(0, 60, 1): # Finer search for final result
                        side = calc_outer_side(hexagons, oa_deg)
                        if side < best_side:
                            best_side = side
                            best_oa = oa_deg
                            
                    if best_side < best_score:
                        best_score = best_side
                        # Construct inner list
                        inner_list = []
                        for i in range(N):
                            inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
                        best_result = (inner_list, (0.0, 0.0), best_score, best_oa)
                        
        except Exception as e:
            pass
            
        restarts_done += 1
        
    # If no valid result found, fallback to the loose lattice
    if best_result is None:
        inner_list = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=2.0)
            inner_list.append([x, y, 0.0])
        
        # Center
        cx_avg = sum(p[0] for p in inner_list) / N
        cy_avg = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg, y - cy_avg, a] for x, y, a in inner_list]
        
        # Calculate side
        hexagons = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        side = calc_outer_side(hexagons, 0.0)
        best_result = (inner_list, (0.0, 0.0), side, 0.0)
        
    return best_result

# EVOLVE-BLOCK-END

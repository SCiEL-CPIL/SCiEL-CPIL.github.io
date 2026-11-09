import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator limit 150s)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Get axes for SAT (normals to edges)
    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def project_poly(vertices, axis):
        nx, ny = axis
        min_val = float('inf')
        max_val = float('-inf')
        for vx, vy in vertices:
            proj = vx * nx + vy * ny
            if proj < min_val:
                min_val = proj
            if proj > max_val:
                max_val = proj
        return min_val, max_val

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2, d_sq):
        """
        Fast disjoint check.
        1. If d_sq > 4.0 (d > 2.0), definitely disjoint.
        2. If d_sq < 3.0 (d < sqrt(3)), definitely overlap (min distance for disjoint is sqrt(3)).
        3. Otherwise, use SAT.
        """
        if d_sq > 4.0:
            return True
        if d_sq < 3.0:
            return False
            
        # SAT check
        all_axes = axes1 + axes2
        for axis in all_axes:
            min1, max1 = project_poly(hex1_verts, axis)
            min2, max2 = project_poly(hex2_verts, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
        """
        Calculate the minimum side length S of a regular hexagon centered at (0,0)
        with rotation outer_angle_deg that contains all given hexagons.
        """
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # Apothem of outer hex is S * SQRT3 / 2
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    def get_config_data(p):
        """Extract centers and angles, center the configuration, and compute geometry."""
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[i*3], p[i*3+1]))
            angles.append(p[i*3+2])
        
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
            
        return shifted_centers, angles, hexagons, hex_axes

    def evaluate(p, outer_angle_deg=0.0):
        """
        Objective function for SLSQP.
        Returns (side_length, constraints_list).
        Constraints are of the form: min_distance - sqrt(3) >= 0
        """
        centers, angles, hexagons, hex_axes = get_config_data(p)
        
        # Calculate outer side length
        side = calc_outer_side(hexagons, outer_angle_deg)
        
        # Calculate constraints (non-overlap)
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d_sq = dx*dx + dy*dy
                
                # If definitely disjoint, constraint is satisfied (large positive value)
                if d_sq > 4.0:
                    constraints.append(10.0) # Large positive number
                    continue
                
                # If definitely overlapping, constraint is violated (negative value)
                if d_sq < 3.0:
                    constraints.append(-10.0) # Large negative number
                    continue
                
                # SAT check
                if is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                    constraints.append(10.0)
                else:
                    constraints.append(-10.0)
                    
        return side, constraints

    def objective_func(p):
        """Wrapper for minimize(). Returns only the objective value."""
        side, _ = evaluate(p)
        return side

    def constraint_func(p, i, j):
        """
        Constraint function for pair (i, j).
        Returns min_distance - sqrt(3). Must be >= 0.
        """
        centers, angles, hexagons, hex_axes = get_config_data(p)
        
        dx = centers[i][0] - centers[j][0]
        dy = centers[i][1] - centers[j][1]
        d_sq = dx*dx + dy*dy
        
        if d_sq > 4.0:
            return 10.0
        if d_sq < 3.0:
            return -10.0
            
        if is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
            return 10.0
        else:
            return -10.0

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Initial configurations
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    coords3 = [(0,0), (1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (0,1), (1,1), (2,1), (3,1)]
    coords4 = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(2,0),(2,1)]
    coords5 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,1), (-2,1), (-2,0)]
    coords6 = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(0,2),(1,2)]
    
    configs = [coords1, coords2, coords3, coords4, coords5, coords6]
    
    best_params = None
    best_score = float('inf')
    
    # Setup constraints list for SLSQP
    # We need to define a list of constraint dictionaries for each pair
    pair_indices = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    def build_constraints_list():
        cons = []
        for i, j in pair_indices:
            # Create a lambda that captures i and j
            def c(p, i=i, j=j):
                return constraint_func(p, i, j)
            cons.append({'type': 'ineq', 'fun': c})
        return cons

    constraints = build_constraints_list()
    
    # Run SLSQP from multiple initial configurations
    for config in configs:
        if time.time() > deadline - 20.0:
            break
            
        # Initialize parameters
        init_params = []
        base_centers = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            init_params.extend([x, y, 0.0])
            base_centers.append((x, y))
            
        # Center the initial parameters to help optimizer
        cx_avg = sum(p[0] for p in base_centers) / N
        cy_avg = sum(p[1] for p in base_centers) / N
        for i in range(N):
            init_params[i*3] -= cx_avg
            init_params[i*3+1] -= cy_avg
            
        init_params = np.array(init_params)
        
        try:
            # Run SLSQP
            res = minimize(
                objective_func, 
                init_params, 
                method='SLSQP', 
                bounds=[(-5, 5)] * (2*N) + [(-180, 180)] * N,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
            )
            
            if res.success or (res.fun < best_score and res.fun < 20.0):
                # Validate result
                side, cons_vals = evaluate(res.x)
                valid = all(v >= -1e-5 for v in cons_vals)
                
                if valid and side < best_score:
                    best_score = side
                    best_params = res.x
                    
        except Exception:
            pass

    # If SLSQP failed to find a valid solution, fallback to a known valid configuration
    if best_params is None:
        # Use coords1 as fallback, manually centered
        init_params = []
        base_centers = []
        for q, r in configs[0]:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            init_params.extend([x, y, 0.0])
            base_centers.append((x, y))
        cx_avg = sum(p[0] for p in base_centers) / N
        cy_avg = sum(p[1] for p in base_centers) / N
        for i in range(N):
            init_params[i*3] -= cx_avg
            init_params[i*3+1] -= cy_avg
        best_params = np.array(init_params)
        
        # Calculate score for fallback
        side, _ = evaluate(best_params)
        best_score = side

    # Final Validation and Outer Angle Optimization
    final_params = best_params.copy()
    
    centers, angles, hexagons, hex_axes = get_config_data(final_params)
    
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            d_sq = dx*dx + dy*dy
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Emergency fallback: Return the raw initial config 1
        inner_list = []
        for q, r in configs[0]:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg_fb = sum(p[0] for p in inner_list) / N
        cy_avg_fb = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg_fb, y - cy_avg_fb, a] for x, y, a in inner_list]
        hexagons_fb = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side(hexagons_fb, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Optimize outer angle
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 600):
        oa = oa_deg * 0.1
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa = oa
            
    inner_list = []
    for i in range(N):
        inner_list.append([centers[i][0], centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END

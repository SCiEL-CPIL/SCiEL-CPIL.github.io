"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill

    # Helper: Get vertices of a unit hexagon
    def get_vertices(cx, cy, angle_deg):
        rad = np.deg2rad(angle_deg)
        # Flat-top hexagon vertices at 0, 60, 120, 180, 240, 300
        # If angle=0, vertex at (1,0)
        angles = rad + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))

    # Helper: Calculate minimum outer side length S to contain a set of points
    def calc_min_S(vertices_list):
        # Outer hexagon: flat-top, center (0,0), side S
        # Inequalities:
        # 1. |x| <= S
        # 2. |sqrt(3)/2 * x + 1/2 * y| <= sqrt(3)/2 * S  => |x + y/sqrt(3)| <= S
        # 3. |sqrt(3)/2 * x - 1/2 * y| <= sqrt(3)/2 * S  => |x - y/sqrt(3)| <= S
        
        max_S = 0.0
        for verts in vertices_list:
            for x, y in verts:
                s1 = abs(x)
                s2 = abs(x + y / np.sqrt(3))
                s3 = abs(x - y / np.sqrt(3))
                s_point = max(s1, s2, s3)
                if s_point > max_S:
                    max_S = s_point
        return max_S

    # Helper: SAT overlap depth between two hexagons
    def sat_depth(v1, v2):
        # v1, v2: (6,2)
        axes = []
        for v in (v1, v2):
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    normal /= norm
                    axes.append(normal)
        
        min_depth = float('inf')
        for axis in axes:
            proj1 = v1 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            proj2 = v2 @ axis
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 <= min2 or max2 <= min1:
                return 0.0
            
            depth = min(max1, max2) - max(min1, min2)
            if depth < min_depth:
                min_depth = depth
                
        return min_depth

    # Objective Function
    def objective(params):
        # params: [x1, y1, a1, x2, y2, a2, ...]
        vertices_list = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            verts = get_vertices(x, y, a)
            vertices_list.append(verts)
        
        S_req = calc_min_S(vertices_list)
        
        # Overlap Penalty
        overlap_sum = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = sat_depth(vertices_list[i], vertices_list[j])
                if d > 0:
                    overlap_sum += d * d
                    
        # Penalty weight
        # We want to strongly penalize overlap
        return S_req + 100.0 * overlap_sum

    # Initialization
    # Create a hexagonal lattice of 12 points
    # Radius 1: 7 points. Radius 2: 19 points.
    # Pick 12 closest to center from a larger grid.
    points = []
    # Generate axial coordinates
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Convert axial to cartesian
            # x = 1.5 * q
            # y = sqrt(3) * (r + 0.5 * q)
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * q)
            points.append((x, y))
            
    # Sort by distance to origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take 12 closest
    init_points = points[:N]
    
    # Scale up slightly to ensure no initial overlap (lattice spacing is exactly touching if side=1)
    # Hexagon side 1 -> distance between centers for touching is sqrt(3)
    # In my lattice generation:
    # q=1, r=0 -> x=1.5, y=sqrt(3)/2. dist = sqrt(2.25 + 0.75) = sqrt(3). Correct.
    # So they are touching. Add small epsilon.
    scale = 1.01
    init_params = []
    for x, y in init_points:
        init_params.extend([x * scale, y * scale, 0.0])
        
    init_params = np.array(init_params)
    
    # Optimization
    best_params = init_params
    best_score = objective(init_params)
    
    # Use Nelder-Mead
    # Set maxiter based on time
    # Each eval is ~12*6*12*2 ops? Fast.
    # Let's run for a few seconds.
    
    try:
        result = minimize(
            objective, 
            init_params, 
            method='Nelder-Mead', 
            options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False}
        )
        
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
            
    except Exception as e:
        pass
        
    # Check if current best is valid (no overlaps)
    # If not, we might need to try a different initialization or accept a valid one with larger S
    # Let's verify validity of best_params
    def is_valid(params):
        vertices_list = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            verts = get_vertices(x, y, a)
            vertices_list.append(verts)
        for i in range(N):
            for j in range(i + 1, N):
                if sat_depth(vertices_list[i], vertices_list[j]) > 1e-6:
                    return False
        return True

    if not is_valid(best_params):
        # Fallback: Return a known valid configuration (e.g., spread out)
        # Or retry with larger initial scale
        print("Warning: Best params invalid, using fallback")
        # Simple fallback: Grid with large spacing
        fallback_params = []
        for i in range(N):
            # Place in a line or grid with large spacing
            x = (i % 4) * 3.0
            y = (i // 4) * 3.0
            fallback_params.extend([x, y, 0.0])
        best_params = np.array(fallback_params)
        # Recalculate S for this valid config
        vertices_list = []
        for i in range(N):
            x = best_params[i * 3]
            y = best_params[i * 3 + 1]
            a = best_params[i * 3 + 2]
            verts = get_vertices(x, y, a)
            vertices_list.append(verts)
        S_final = calc_min_S(vertices_list)
        # Center the configuration?
        # For now, just return it. The outer hexagon is at 0,0.
        # This fallback is likely suboptimal but valid.
        
    # Extract final positions
    inner = []
    for i in range(N):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        a = best_params[i * 3 + 2]
        inner.append([float(x), float(y), float(a)])
        
    # Calculate final S
    vertices_list = []
    for i in range(N):
        x = inner[i][0]
        y = inner[i][1]
        a = inner[i][2]
        verts = get_vertices(x, y, a)
        vertices_list.append(verts)
    S_final = float(calc_min_S(vertices_list))
    
    # Outer hexagon center and rotation
    center = (0.0, 0.0)
    side = S_final
    angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END

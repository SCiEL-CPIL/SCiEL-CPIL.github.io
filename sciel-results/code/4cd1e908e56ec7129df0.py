"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def hex_vertices(cx, cy, side, angle_deg=0.0):
        """Get vertices of a regular hexagon."""
        angle = np.radians(angle_deg)
        # For a regular hexagon, vertices are at angles 0, 60, 120, 180, 240, 300 degrees
        # plus the rotation
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle
        # A regular hexagon with side length s has circumradius = s
        # But we want the flat-top orientation where the "width" is 2*s
        # Actually, let's use the standard: vertices at distance s from center
        # at angles offset by 30 degrees for flat-top
        angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + angle
        verts = np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
        return verts
    
    def point_in_hex(px, py, cx, cy, side, angle_deg=0.0):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and angle."""
        # Transform to hexagon's local coordinates
        angle = np.radians(angle_deg)
        dx = px - cx
        dy = py - cy
        # Rotate back
        x = dx * np.cos(-angle) - dy * np.sin(-angle)
        y = dx * np.sin(-angle) + dy * np.cos(-angle)
        
        # For a regular hexagon with flat top, the boundaries are:
        # |x| <= side (for the flat sides)
        # |y| <= side * sqrt(3) / 2 (for the pointed sides)
        # Actually, let me use the proper hexagon containment test
        # A regular hexagon with circumradius R (= side length for regular hex)
        # has inradius = R * sqrt(3)/2
        # The hexagon is the intersection of 6 half-planes
        # For flat-top orientation:
        # x/2 + y*sqrt(3)/2 <= R*sqrt(3)/2  (right side)
        # -x/2 + y*sqrt(3)/2 <= R*sqrt(3)/2  (upper right)
        # -x/2 - y*sqrt(3)/2 <= R*sqrt(3)/2  (left side)
        # x/2 - y*sqrt(3)/2 <= R*sqrt(3)/2  (lower right)
        # |y| <= R*sqrt(3)/2  (top and bottom)
        # Actually simpler: for a regular hexagon with side s (circumradius = s),
        # in flat-top orientation, the vertices are at:
        # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
        # The containing region is: |y| <= s*sqrt(3)/2 AND |x| + |y|/sqrt(3) <= s
        
        R = side
        inradius = R * np.sqrt(3) / 2
        # Check the 6 half-planes
        # Normal vectors for flat-top hexagon
        # The six edges have outward normals at angles 0, 60, 120, 180, 240, 300
        # Each edge is at distance inradius from center
        for k in range(6):
            theta = k * np.pi / 3  # 0, 60, 120, ...
            nx = np.cos(theta)
            ny = np.sin(theta)
            if nx * x + ny * y > inradius:
                return False
        return True
    
    def hex_overlap(c1, c2, side, angle1=0.0, angle2=0.0):
        """Check if two hexagons overlap. Uses separating axis theorem."""
        v1 = hex_vertices(c1[0], c1[1], side, angle1)
        v2 = hex_vertices(c2[0], c2[1], side, angle2)
        
        # Collect all edge normals from both hexagons
        normals = []
        for v in [v1, v2]:
            for i in range(6):
                e1 = v[i]
                e2 = v[(i+1) % 6]
                edge = e2 - e1
                # Normal to this edge
                nx = -edge[1]
                ny = edge[0]
                norm = np.sqrt(nx*nx + ny*ny)
                if norm > 1e-10:
                    normals.append((nx/norm, ny/norm))
        
        # For each normal, project both hexagons and check for separation
        for nx, ny in normals:
            proj1 = np.array([nx * p[0] + ny * p[1] for p in v1])
            proj2 = np.array([nx * p[0] + ny * p[1] for p in v2])
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False  # Separated
        
        return True  # Overlapping
    
    def compute_min_side(positions, angles_deg, center):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        # The outer hexagon must contain all vertices of all inner hexagons
        # For a given outer hexagon center and rotation, find the minimum side
        # such that all points are inside
        
        # We'll try to find the optimal outer hexagon orientation
        # For simplicity, assume outer hexagon is axis-aligned (angle=0)
        # Then find minimum side
        
        # Actually, we need to optimize the outer hexagon too
        # Let's fix outer angle to 0 for now and find min side
        
        # For a flat-top regular hexagon with center (cx, cy) and side s:
        # A point (px, py) is inside if for all 6 directions:
        # (px-cx)*cos(theta_k) + (py-cy)*sin(theta_k) <= s*sqrt(3)/2
        # where theta_k = k*pi/3
        
        # So s >= max over all points and all k of:
        # [(px-cx)*cos(theta_k) + (py-cy)*sin(theta_k)] / (sqrt(3)/2)
        
        cx, cy = center
        max_val = 0.0
        
        # Collect all vertices of all inner hexagons
        all_points = []
        for i in range(len(positions)):
            verts = hex_vertices(positions[i][0], positions[i][1], 1.0, angles_deg[i])
            all_points.extend(verts.tolist())
        
        all_points = np.array(all_points)
        
        # Try several outer orientations
        best_side = float('inf')
        for outer_angle in [0, 30, 60]:
            outer_angle_rad = np.radians(outer_angle)
            # The 6 normal directions for the outer hexagon
            theta_base = np.radians(30) + outer_angle_rad  # For flat-top
            for k in range(6):
                theta = theta_base + k * np.pi / 3
                nx = np.cos(theta)
                ny = np.sin(theta)
                projections = (all_points[:, 0] - cx) * nx + (all_points[:, 1] - cy) * ny
                max_proj = np.max(projections)
                if max_proj > 0:
                    side_needed = max_proj / (np.sqrt(3) / 2)
                    if side_needed > best_side:
                        best_side = side_needed
        
        return best_side
    
    def objective(params):
        """Objective function: minimize outer hexagon side length."""
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        
        # Parse parameters: for each hexagon, (x, y, angle)
        positions = []
        angles_deg = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            angle = params[3*i+2]
            positions.append((x, y))
            angles_deg.append(np.degrees(angle))
        
        # Check for overlaps
        for i in range(N):
            for j in range(i+1, N):
                if hex_overlap(positions[i], positions[j], 1.0, angles_deg[i], angles_deg[j]):
                    # Penalize overlap
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # Minimum distance for non-overlap is roughly 2*inradius = sqrt(3)
                    penalty = max(0, np.sqrt(3) - dist) * 100
                    if penalty > 0:
                        return penalty + compute_min_side(positions, angles_deg, (0, 0))
        
        # Compute outer side
        return compute_min_side(positions, angles_deg, (0, 0))
    
    # Initial placement: hexagonal grid pattern
    # Try a 3-4-3 arrangement or similar
    # For 12 hexagons, let's try a rectangular-ish grid
    
    # Hexagonal grid spacing: for flat-top hexagons, horizontal spacing = 1.5, vertical = sqrt(3)
    # But for non-overlap, we need center distance >= sqrt(3) for same orientation
    
    # Let's try a 4x3 grid with hexagonal offset
    init_params = []
    sqrt3 = np.sqrt(3)
    
    # Try arrangement: rows of 3, 4, 3, 2 or similar
    # Let's use a compact hexagonal arrangement
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    rows = [3, 4, 3, 2]
    y_offset = 0
    for row_count in rows:
        x_start = -(row_count - 1) * 0.75  # Offset by half spacing
        for k in range(row_count):
            x = x_start + k * 1.5
            y = y_offset
            init_params.extend([x, y, 0.0])
        y_offset += sqrt3 * 0.866  # Vertical spacing for hex grid
    
    # Optimize
    best_result = None
    best_obj = float('inf')
    
    # Try multiple initial points
    initial_candidates = []
    
    # Candidate 1: The grid above
    initial_candidates.append(init_params.copy())
    
    # Candidate 2: More compact - try rotating some hexagons
    init_params2 = []
    y_offset = 0
    for row_count in rows:
        x_start = -(row_count - 1) * 0.75
        for k in range(row_count):
            x = x_start + k * 1.5
            y = y_offset
            angle = 0.0 if (y_offset == 0 or y_offset == 3*sqrt3*0.866) else np.pi/6
            init_params2.extend([x, y, angle])
        y_offset += sqrt3 * 0.866
    initial_candidates.append(init_params2.copy())
    
    # Candidate 3: Tighter packing with all same orientation
    init_params3 = []
    y_offset = 0
    for row_count in rows:
        x_start = -(row_count - 1) * 0.75
        for k in range(row_count):
            x = x_start + k * 1.5
            y = y_offset
            init_params3.extend([x, y, 0.0])
        y_offset += sqrt3 * 0.866
    initial_candidates.append(init_params3.copy())
    
    for init in initial_candidates:
        if time.time() > deadline - 30:
            break
            
        try:
            result = minimize(objective, init, method='L-BFGS-B', 
                            options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-8})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except Exception as e:
            pass
    
    # Also try a simple local search / Nelder-Mead
    if time.time() < deadline - 20:
        try:
            result = minimize(objective, init_params, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except:
            pass
    
    # Use best result or fallback
    if best_result is not None:
        positions = []
        angles_deg = []
        for i in range(N):
            x = best_result[3*i]
            y = best_result[3*i+1]
            angle_deg = np.degrees(best_result[3*i+2])
            positions.append([x, y, angle_deg])
            angles_deg.append(angle_deg)
        
        outer_side = compute_min_side([(p[0], p[1]) for p in positions], angles_deg, (0, 0))
        
        # Verify validity
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if hex_overlap((positions[i][0], positions[i][1]), (positions[j][0], positions[j][1]), 1.0, angles_deg[i], angles_deg[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            inner = [[p[0], p[1], p[2]] for p in positions]
            return inner, (0.0, 0.0), outer_side, 0.0
    # Fallback: use initial grid
    positions = []
    angles_deg = []
    for i in range(N):
        x = init_params[3*i]
        y = init_params[3*i+1]
        angle_deg = np.degrees(init_params[3*i+2])
        positions.append([x, y, angle_deg])
        angles_deg.append(angle_deg)
    
    outer_side = compute_min_side([(p[0], p[1]) for p in positions], angles_deg, (0, 0))
    inner = [[p[0], p[1], p[2]] for p in positions]
    return inner, (0.0, 0.0), outer_side, 0.0
# EVOLVE-BLOCK-END

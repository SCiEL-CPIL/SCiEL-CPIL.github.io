import numpy as np
from math import sqrt, cos, sin, pi, floor, ceil
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_result = None
    best_side = float('inf')
    
    def hex_to_xy(q, r, size=1.0):
        """Convert axial hex coordinates to pixel coordinates for pointy-top hexagons."""
        x = size * (sqrt(3) * q + sqrt(3)/2 * r)
        y = size * (3/2 * r)
        return x, y
    
    def hex_to_xy_flat(q, r, size=1.0):
        """Convert axial hex coordinates to pixel coordinates for flat-top hexagons."""
        x = size * (3/2 * q)
        y = size * (sqrt(3) * r + sqrt(3)/2 * q)
        return x, y
    
    def compute_bounding_hex_side(positions, angles, hex_side=1.0):
        """
        Given positions (x,y) and angles of unit hexagons, compute the minimum
        side length of a regular hexagon (with given orientation) that contains them all.
        We'll try to find the minimal containing hexagon by checking constraints.
        """
        if not positions:
            return 0.0, (0.0, 0.0), 0.0
        
        # For a regular hexagon containing all points, we need to find the minimal S
        # such that all points are inside a hexagon of side S at some center and rotation.
        # 
        # A regular hexagon of side S centered at (cx, cy) with rotation theta has 6 edges.
        # Each edge is a line; the hexagon is the intersection of 6 half-planes.
        # 
        # For a point to be inside, it must satisfy all 6 half-plane constraints.
        # The minimal S for a given center and rotation can be found by computing the
        # maximum "distance" to each of the 6 supporting lines.
        
        # We'll search over center and rotation to minimize S.
        # For simplicity, assume the outer hexagon has the same orientation as the inner ones (angle=0)
        # and center at the centroid of inner hexagons.
        
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        
        # For a flat-top hexagon of side S centered at (cx,cy) with rotation 0:
        # The 6 edge normals are at angles 0, 60, 120, 180, 240, 300 degrees.
        # The distance from center to each edge (apothem) = S * sqrt(3)/2.
        # A point (px,py) is inside if for all 6 directions, the projection onto the normal
        # is <= S*sqrt(3)/2.
        
        # So S >= max_over_points, max_over_normals |dot(point - center, normal)| / (sqrt(3)/2)
        
        # But we can also shift the center and rotate the outer hexagon.
        # For now, let's try: center at centroid, rotation 0, and compute required S.
        
        # Actually, let's be more careful. The inner hexagons have their own extent.
        # A unit hexagon at position (px,py) with angle a occupies a region.
        # The vertices of this hexagon are at distance 1 from (px,py) in 6 directions.
        
        # For the bounding hexagon, we need all vertices of all inner hexagons to be inside.
        
        all_vertices = []
        for (px, py, ang) in positions:
            for k in range(6):
                theta = ang * pi / 180.0 + k * pi / 3.0
                vx = px + cos(theta)
                vy = py + sin(theta)
                all_vertices.append((vx, vy))
        
        all_vertices = np.array(all_vertices)
        
        # Now find minimal S, center (cx,cy), rotation phi such that all vertices are inside
        # a regular hexagon of side S.
        
        # For a given rotation phi and center (cx,cy), the required S is:
        # S = (2/sqrt(3)) * max_i max_j |dot(v_i - c, n_j)|
        # where n_j are the 6 unit normals of the hexagon edges.
        
        # We can optimize over phi and center. For a fixed phi, the optimal center is the
        # center of the minimal enclosing hexagon, which can be found by binary search on S
        # and checking feasibility.
        
        # For simplicity, let's try a few rotations and centers.
        
        best_S = float('inf')
        best_cx, best_cy = cx, cy
        best_phi = 0.0
        
        # Try rotations from 0 to 60 degrees in small steps
        for phi_deg in range(0, 60, 5):
            phi = phi_deg * pi / 180.0
            # Normals for the 6 edges
            normals = []
            for j in range(6):
                angle = phi + j * pi / 3.0
                normals.append((cos(angle), sin(angle)))
            normals = np.array(normals)
            
            # For a given center (cx, cy), required S = (2/sqrt(3)) * max_i max_j |dot(v_i - c, n_j)|
            # We want to minimize this over (cx, cy).
            # This is equivalent to finding the center that minimizes the max over all (vertex, normal) pairs.
            # 
            # Let d_ij = dot(v_i, n_j). We want to choose c to minimize max_ij |dot(v_i - c, n_j)|
            # = max_ij |d_ij - dot(c, n_j)|
            #
            # Let t_j = dot(c, n_j). We want to minimize max_ij |d_ij - t_j| over t_j (with constraint that t_j are consistent with some c).
            #
            # For 6 normals in 2D, the t_j are not independent. But for a first approximation, let's just
            # compute the required S for center = centroid.
            
            cx_try = np.mean(all_vertices[:, 0])
            cy_try = np.mean(all_vertices[:, 1])
            c = np.array([cx_try, cy_try])
            
            # Compute max projection
            # For each vertex and each normal, compute |dot(v - c, n)|
            # This is a (M, 6) matrix
            diffs = all_vertices - c  # (M, 2)
            projs = np.abs(diffs @ normals.T)  # (M, 6)
            max_proj = np.max(projs)
            S = 2.0 * max_proj / sqrt(3)
            
            if S < best_S:
                best_S = S
                best_cx, best_cy = cx_try, cy_try
                best_phi = phi
        
        return best_S, (best_cx, best_cy), best_phi * 180.0 / pi
    
    # Generate candidate arrangements
    
    # Strategy 1: Hexagonal grid arrangement
    # Place hexagons in a hexagonal lattice pattern
    
    def generate_hex_grid(r_max):
        """Generate hex grid positions for radius up to r_max."""
        positions = []
        for q in range(-r_max, r_max + 1):
            for r in range(-r_max, r_max + 1):
                if abs(q + r) <= r_max:
                    x, y = hex_to_xy_flat(q, r)
                    positions.append((x, y, 0.0))
        return positions
    
    # Try different grid sizes and pick the first N
    for r_max in range(0, 10):
        if time.time() > deadline:
            break
        positions = generate_hex_grid(r_max)
        if len(positions) < N:
            continue
        # Take first N positions (center out)
        # Sort by distance from center
        positions_sorted = sorted(positions, key=lambda p: p[0]**2 + p[1]**2)
        selected = positions_sorted[:N]
        
        S, center, angle = compute_bounding_hex_side(selected, selected)
        if S < best_side:
            best_side = S
            best_result = (selected, center, S, angle)
    
    # Strategy 2: Linear arrangements (1D, 2D grids)
    # Row of N hexagons
    for r_max in range(1, N + 1):
        if time.time() > deadline:
            break
        # Horizontal row
        positions = [(2.0 * sqrt(3) * k, 0.0, 0.0) for k in range(N)]
        S, center, angle = compute_bounding_hex_side(positions, positions)
        if S < best_side:
            best_side = S
            best_result = (positions, center, S, angle)
        
        # 2D grid
        for rows in range(1, N + 1):
            if time.time() > deadline:
                break
            cols = (N + rows - 1) // rows
            positions = []
            count = 0
            for r in range(rows):
                for c in range(cols):
                    if count >= N:
                        break
                    x = 2.0 * sqrt(3) * c + (sqrt(3) / 2) * r
                    y = 1.5 * r
                    positions.append((x, y, 0.0))
                    count += 1
            if len(positions) == N:
                S, center, angle = compute_bounding_hex_side(positions, positions)
                if S < best_side:
                    best_side = S
                    best_result = (positions, center, S, angle)
    
    # Strategy 3: Optimize positions with local search
    if best_result is not None and time.time() < deadline - 10:
        inner_positions = best_result[0]
        # Convert to numpy for optimization
        # Parameters: x, y, angle for each hexagon
        # Objective: minimize bounding hexagon side length
        
        def objective(params):
            if time.time() > deadline:
                return best_side
            positions = []
            for i in range(N):
                x = params[3*i]
                y = params[3*i+1]
                ang = params[3*i+2]
                positions.append((x, y, ang))
            S, _, _ = compute_bounding_hex_side(positions, positions)
            return S
        
        from scipy.optimize import minimize
        
        x0 = []
        for (x, y, ang) in inner_positions:
            x0.extend([x, y, ang])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 0.01, 'fatol': 0.01})
            if result.fun < best_side:
                new_positions = []
                for i in range(N):
                    new_positions.append((result.x[3*i], result.x[3*i+1], result.x[3*i+2]))
                S, center, angle = compute_bounding_hex_side(new_positions, new_positions)
                if S < best_side:
                    best_side = S
                    best_result = (new_positions, center, S, angle)
        except:
            pass
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [[2.0 * sqrt(3) * k, 0.0, 0.0] for k in range(N)]
        S, center, angle = compute_bounding_hex_side(inner, inner)
        return inner, center, S, angle
    
    inner, center, S, angle = best_result
    # Ensure inner is in the right format: list of [x, y, angle_degrees]
    inner_formatted = [[float(p[0]), float(p[1]), float(p[2])] for p in inner]
    return inner_formatted, (float(center[0]), float(center[1])), float(S), float(angle)
# EVOLVE-BLOCK-END


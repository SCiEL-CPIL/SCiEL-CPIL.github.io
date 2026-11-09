"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # For unit hexagons (side=1), the inradius is sqrt(3)/2, circumradius is 1
    # Center-to-center distance for non-overlapping hexagons: 2 * inradius = sqrt(3)
    # for flat orientation, or we can compute based on actual geometry
    
    # Initial guess: hexagonal lattice arrangement
    # For 12 hexagons, use a 3-4-3-2 or similar compact arrangement
    # Let's use a hexagonal grid: rows with varying counts
    
    # Hexagonal grid positions for 12 hexagons
    # Using axial coordinates converted to Cartesian
    # A good compact arrangement: center + 6 around + 5 more
    
    # Let's try a simple grid first, then optimize
    # 4 columns x 3 rows in hex grid
    def hex_grid_positions(n_rows, n_cols, spacing):
        positions = []
        for r in range(n_rows):
            for c in range(n_cols):
                x = spacing * (c + 0.5 * (r % 2))
                y = spacing * r * np.sqrt(3) / 2
                positions.append([x, y])
        return np.array(positions)
    
    # For 12 hexagons, try 4x3 hex grid
    spacing = np.sqrt(3)  # center-to-center distance for unit hexagons
    pos = hex_grid_positions(3, 4, spacing)
    
    # Center the positions
    pos -= pos.mean(axis=0)
    
    def compute_outer_hexagon(positions, angles):
        """Compute minimum side length of regular hexagon containing all unit hexagons."""
        # For each inner hexagon, compute its 6 vertices
        # Then find the minimum enclosing regular hexagon
        
        # A regular hexagon with center (cx, cy), side s, rotation theta
        # has vertices at distance s from center at angles theta + k*60 degrees
        
        # The constraint is that all vertices of all inner hexagons must be inside
        # the outer hexagon.
        
        # For a regular hexagon, a point (px, py) is inside if:
        # For each of the 6 edges, the point is on the correct side.
        
        # Alternative: use the support function approach
        # A regular hexagon with side s, center (cx,cy), rotation theta
        # The distance from center to each edge (inradius) = s * sqrt(3) / 2
        
        # Point (px,py) is inside iff for all 6 directions (theta + k*60):
        # (px-cx)*cos(dir) + (py-cy)*sin(dir) <= s * sqrt(3) / 2
        
        # So the minimum s is:
        # s = max over all points p and all 6 directions d of:
        #     ((p.x - cx)*cos(d) + (p.y - cy)*sin(d)) / (sqrt(3)/2)
        
        all_vertices = []
        for i, (px, py) in enumerate(positions):
            angle = np.radians(angles[i])
            # Vertices of unit hexagon (side=1) centered at (px,py) with rotation angle
            # For a regular hexagon with side s, vertices are at distance s from center
            # at angles: angle + k*60 degrees
            for k in range(6):
                v_angle = angle + k * np.pi / 3
                vx = px + np.cos(v_angle)
                vy = py + np.sin(v_angle)
                all_vertices.append([vx, vy])
        
        all_vertices = np.array(all_vertices)
        
        # Now find min enclosing regular hexagon
        # Parameterize by center (cx, cy) and rotation theta
        # s = max over vertices v and directions d_k = theta + k*60:
        #     (v.x - cx)*cos(d_k) + (v.y - cy)*sin(d_k) all divided by (sqrt(3)/2)
        # and we need all these to be <= s, and also >= -s (symmetric)
        # Actually for a regular hexagon centered at (cx,cy), the condition is:
        # |dot(v - c, u_k)| <= inradius = s*sqrt(3)/2 for all 6 unit vectors u_k
        
        # So s = max_k max_v |dot(v - c, u_k)| / (sqrt(3)/2)
        
        return all_vertices
    
    # Optimization variables: [x1,y1,theta1, ..., x12,y12,theta12, cx, cy, theta_outer]
    # That's 12*3 + 3 = 39 variables
    
    # Initial guess
    angles = np.zeros(N)
    cx, cy = 0.0, 0.0
    theta_outer = 0.0
    
    x0 = np.concatenate([pos.flatten(), angles, [cx, cy, theta_outer]])
    
    def cost(params):
        if time.time() > deadline:
            return 1e10
        pos_flat = params[:2*N]
        angles = params[2*N:3*N]
        cx, cy, theta_outer = params[3*N:3*N+3]
        
        positions = pos_flat.reshape(N, 2)
        
        all_vertices = compute_outer_hexagon(positions, angles)
        
        # Compute minimum enclosing regular hexagon
        # For direction theta_outer + k*60, compute max projection
        max_proj = 0.0
        for k in range(6):
            d = theta_outer + k * np.pi / 3
            dx, dy = np.cos(d), np.sin(d)
            projections = (all_vertices[:, 0] - cx) * dx + (all_vertices[:, 1] - cy) * dy
            max_proj = max(max_proj, np.max(np.abs(projections)))
        
        # inradius = s * sqrt(3) / 2, so s = max_proj / (sqrt(3)/2) = 2*max_proj/sqrt(3)
        s = 2.0 * max_proj / np.sqrt(3)
        
        # Add penalty for overlapping inner hexagons
        # Check pairwise distances
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # Minimum distance for non-overlapping unit hexagons
                # The minimum distance depends on relative orientation
                # Conservative: require dist >= sqrt(3) (for aligned hexagons)
                # More accurate: compute actual minimum distance based on orientations
                min_dist = np.sqrt(3)  # conservative
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return s + 100.0 * penalty
    
    # Use Nelder-Mead for this non-smooth problem
    result = minimize(cost, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if time.time() > deadline - 10:
        # Use initial guess if we're running out of time
        best_params = x0
    else:
        best_params = result.x
    
    pos_opt = best_params[:2*N].reshape(N, 2)
    angles_opt = best_params[2*N:3*N]
    cx_opt, cy_opt, theta_outer_opt = best_params[3*N:3*N+3]
    
    # Compute final side length
    all_vertices = compute_outer_hexagon(pos_opt, angles_opt)
    max_proj = 0.0
    for k in range(6):
        d = theta_outer_opt + k * np.pi / 3
        dx, dy = np.cos(d), np.sin(d)
        projections = (all_vertices[:, 0] - cx_opt) * dx + (all_vertices[:, 1] - cy_opt) * dy
        max_proj = max(max_proj, np.max(np.abs(projections)))
    
    side_length = 2.0 * max_proj / np.sqrt(3)
    
    # Build inner hexagons
    inner = []
    for i in range(N):
        x, y = pos_opt[i]
        angle_deg = np.degrees(angles_opt[i])
        inner.append([x, y, angle_deg])
    
    return inner, (cx_opt, cy_opt), side_length, np.degrees(theta_outer_opt)
# EVOLVE-BLOCK-END

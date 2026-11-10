# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the respective outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    deadline = time.time() + 620  # 10s safety margin
    
    # Initial seed: hexagonal grid arrangement
    # Unit hexagons have flat-top orientation (angle=0)
    # Distance between centers of adjacent hexagons in a grid:
    # For flat-top hexagons with side 1:
    #   horizontal neighbors: distance = 2 * (3/2) = 3 (wait, let me recalculate)
    #   Actually for regular hexagon with side s=1:
    #   width (flat-to-flat) = 2 * s * cos(30°) = 2 * (√3/2) = √3 ≈ 1.732
    #   height (point-to-point) = 2 * s = 2
    #   For flat-top orientation (vertex at top), the "width" is 2*s = 2, "height" is √3
    #   
    #   Let me use pointy-top orientation (angle=90°) for easier grid packing:
    #   Pointy-top: width = √3, height = 2
    #   Horizontal spacing: √3
    #   Vertical spacing: 1.5 (for adjacent rows, offset by √3/2)
    
    # Use pointy-top hexagons (angle = 90 degrees)
    # Grid layout for 12 hexagons:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    # Actually, let me use a more compact arrangement
    # For pointy-top hexagons:
    #   x-spacing = √3
    #   y-spacing = 1.5
    #   Odd rows offset by √3/2 in x
    
    # Let me try a 4-3-4-1 or similar arrangement
    # Better: use the natural hexagonal close packing
    
    # Initial positions (pointy-top, angle=90)
    sqrt3 = np.sqrt(3)
    
    # Hexagonal arrangement:
    # Center at (0,0)
    # 6 around it at distance √3 in directions 0, 60, 120, 180, 240, 300 degrees
    # Then 5 more in the next ring
    
    # For pointy-top hexagons, the distance between centers of touching hexagons is √3
    # Directions from center to neighbors (for pointy-top):
    # The 6 neighbors are at angles: 90°, 150°, 210°, 270°, 330°, 30° (i.e., 90+60k)
    # Wait, for pointy-top, the vertices are at 90, 150, 210, 270, 330, 30
    # The neighbor centers are in the directions of the edges' normals
    
    # Let me just compute: for two touching pointy-top hexagons, center-to-center distance = √3
    # The 6 neighbor directions (from center) are at angles: 0°, 60°, 120°, 180°, 240°, 300°
    # (since the hexagon is symmetric)
    
    # Actually, let me verify: a pointy-top hexagon has vertices at angles 90, 150, 210, 270, 330, 30
    # The edge normals (pointing outward) are at angles: 60, 120, 180, 240, 300, 0
    # So neighbor centers are in those directions at distance √3
    
    # Let me use flat-top orientation instead for simpler math
    # Flat-top: vertices at 0, 60, 120, 180, 240, 300
    # Edge normals at: 30, 90, 150, 210, 270, 330
    # Neighbor directions: 30, 90, 150, 210, 270, 330 at distance √3
    
    # I'll use flat-top (angle=0)
    
    # Positions for 12 hexagons in a hexagonal cluster:
    # Center: (0, 0)
    # Ring 1 (6 hexagons): at distance √3 in 6 directions
    # Ring 2 (5 hexagons): selected from the 12 positions at distance 2√3 or √3*√3
    
    # For a hexagonal number arrangement: 1 + 6 + 12 = 19 for full 3 rings
    # We need 12, so: 1 + 6 + 5 = 12
    
    # Ring 1 positions (flat-top, angle=0):
    # Directions: 30°, 90°, 150°, 210°, 270°, 330°
    ring1_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    ring1_positions = np.column_stack([np.sqrt(3) * np.cos(ring1_angles), 
                                        np.sqrt(3) * np.sin(ring1_angles)])
    
    # Ring 2: 12 positions at distance 2*sqrt(3) in the same 6 directions, 
    # plus 6 positions at distance sqrt(3)*sqrt(3)=3 at angles 0, 60, 120, 180, 240, 300
    # Actually, ring 2 of a hexagonal grid has 12 positions
    
    # Let me just generate a hexagonal grid and pick 12
    
    # Generate hex grid positions (flat-top)
    # Using axial coordinates
    def axial_to_cube(q, r):
        x = q
        z = r
        y = -x - z
        return x, y, z
    
    def axial_to_pixel(q, r, size=1):
        # Flat-top hexagon
        x = size * (3.0/2.0 * q)
        y = size * (np.sqrt(3)/2.0 * q + np.sqrt(3) * r)
        return x, y
    
    # Generate positions in a hexagonal pattern
    # For a hexagon of radius R (in axial coords), we have positions where max(|q|,|r|,|q+r|) <= R
    # R=0: 1 position
    # R=1: 7 positions  
    # R=2: 19 positions
    
    # We want 12 positions. Let's take R=1 (7 positions) + 5 from R=2
    
    # R=1 positions:
    r1_positions = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            if abs(q + r) <= 1:
                x, y = axial_to_pixel(q, r)
                r1_positions.append((x, y))
    
    # R=2 positions (only those with max(|q|,|r|,|q+r|) == 2):
    r2_positions = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2:
                x, y = axial_to_pixel(q, r)
                r2_positions.append((x, y))
    
    # Take all 7 from R=1 and 5 from R=2 (choose the first 5)
    selected_positions = r1_positions + r2_positions[:5]
    
    inner_x = np.array([p[0] for p in selected_positions])
    inner_y = np.array([p[1] for p in selected_positions])
    inner_angle = np.zeros(12)  # flat-top
    
    # Now we need to find the minimum enclosing regular hexagon
    # The outer hexagon should be centered at the centroid of the inner hexagons
    # and oriented to minimize its size
    
    # For a regular hexagon with side s, the distance from center to vertex is s
    # The distance from center to edge (apothem) is s * √3/2
    
    # An inner hexagon with center (cx, cy) and side 1 (flat-top) has vertices at:
    # (cx + cos(60k), cy + sin(60k)) for k=0..5
    
    # For the inner hexagon to fit inside the outer hexagon (centered at origin, side s, flat-top):
    # All vertices of the inner hexagon must satisfy the 6 half-plane inequalities
    # of the outer hexagon
    
    # Outer hexagon (flat-top, centered at origin, side s) has vertices at:
    # (s*cos(60k), s*sin(60k)) for k=0..5
    # The 6 edges have outward normals at angles 30, 90, 150, 210, 270, 330
    # Each edge: n · v <= apothem = s * √3/2
    
    # So for each inner vertex v, we need:
    # For each normal direction θ ∈ {30, 90, 150, 210, 270, 330}:
    #   v_x * cos(θ) + v_y * sin(θ) <= s * √3/2
    
    # Therefore: s >= max over all inner vertices and all normals of (v_x cos θ + v_y sin θ) / (√3/2)
    #            s >= 2/√3 * max(v_x cos θ + v_y sin θ)
    
    # We can also rotate the outer hexagon to minimize s
    # Let's try different outer hexagon orientations
    
    best_s = np.inf
    best_outer_angle = 0
    
    # Compute all inner vertices
    inner_vertices = []
    for i in range(12):
        for k in range(6):
            angle = np.radians(60 * k)
            vx = inner_x[i] + np.cos(angle)
            vy = inner_y[i] + np.sin(angle)
            inner_vertices.append((vx, vy))
    
    inner_vertices = np.array(inner_vertices)
    
    # Try different outer hexagon orientations
    for outer_angle_deg in range(0, 30, 1):  # 30° symmetry, so 0-30 is enough
        outer_angle = np.radians(outer_angle_deg)
        # Outer hexagon edge normals at angles: outer_angle + 30 + 60k
        normal_angles = [outer_angle + np.radians(30 + 60 * k) for k in range(6)]
        
        max_proj = 0
        for theta in normal_angles:
            proj = inner_vertices[:, 0] * np.cos(theta) + inner_vertices[:, 1] * np.sin(theta)
            max_proj = max(max_proj, np.max(proj))
        
        s = 2.0 / np.sqrt(3) * max_proj
        if s < best_s:
            best_s = s
            best_outer_angle = outer_angle_deg
    
    # Verify the packing is valid (no overlaps)
    # For flat-top hexagons, two hexagons overlap if their center distance < √3
    # (since they touch at distance √3)
    # Actually, for regular hexagons with side 1, the minimum center distance for 
    # non-overlap depends on orientation. For same orientation, it's √3.
    
    # Let's verify by checking all pairs
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            dx = inner_x[i] - inner_x[j]
            dy = inner_y[i] - inner_y[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_dist = min(min_dist, d)
    
    # For flat-top hexagons with same orientation, minimum non-overlapping distance is √3
    # If min_dist < √3, we have overlap
    
    # If there's overlap, we need to adjust
    if min_dist < np.sqrt(3) - 1e-8:
        # Fall back to a known valid packing
        # Use a simple grid that definitely works
        inner_x = np.array([0, -sqrt3, sqrt3, -sqrt3/2, sqrt3/2, -sqrt3/2, sqrt3/2,
                           -sqrt3*1.5, sqrt3*1.5, -sqrt3*1.5, sqrt3*1.5, 0])
        inner_y = np.array([0, 0, 0, 1.5, 1.5, -1.5, -1.5, 1.5, 1.5, -1.5, -1.5, -3])
        inner_angle = np.zeros(12)
        
        # Recompute best_s
        inner_vertices = []
        for i in range(12):
            for k in range(6):
                angle = np.radians(60 * k)
                vx = inner_x[i] + np.cos(angle)
                vy = inner_y[i] + np.sin(angle)
                inner_vertices.append((vx, vy))
        inner_vertices = np.array(inner_vertices)
        
        best_s = np.inf
        best_outer_angle = 0
        for outer_angle_deg in range(0, 30, 1):
            outer_angle = np.radians(outer_angle_deg)
            normal_angles = [outer_angle + np.radians(30 + 60 * k) for k in range(6)]
            
            max_proj = 0
            for theta in normal_angles:
                proj = inner_vertices[:, 0] * np.cos(theta) + inner_vertices[:, 1] * np.sin(theta)
                max_proj = max(max_proj, np.max(proj))
            
            s = 2.0 / np.sqrt(3) * max_proj
            if s < best_s:
                best_s = s
                best_outer_angle = outer_angle_deg
    
    # Try to optimize with local search if time permits
    if time.time() < deadline - 100:
        # Use scipy to optimize positions
        def objective(params):
            if time.time() > deadline - 10:
                return 1e10
            px = params[:12]
            py = params[12:24]
            
            # Check separation
            for i in range(12):
                for j in range(i+1, 12):
                    dx = px[i] - px[j]
                    dy = py[i] - py[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < np.sqrt(3) - 1e-6:
                        return 1e10 + (np.sqrt(3) - d)
            
            # Compute required outer hexagon size
            inner_vertices = []
            for i in range(12):
                for k in range(6):
                    angle = np.radians(60 * k)
                    vx = px[i] + np.cos(angle)
                    vy = py[i] + np.sin(angle)
                    inner_vertices.append((vx, vy))
            inner_vertices = np.array(inner_vertices)
            
            best_s_local = np.inf
            for outer_angle_deg in range(0, 30, 2):
                outer_angle = np.radians(outer_angle_deg)
                normal_angles = [outer_angle + np.radians(30 + 60 * k) for k in range(6)]
                
                max_proj = 0
                for theta in normal_angles:
                    proj = inner_vertices[:, 0] * np.cos(theta) + inner_vertices[:, 1] * np.sin(theta)
                    max_proj = max(max_proj, np.max(proj))
                
                s = 2.0 / np.sqrt(3) * max_proj
                if s < best_s_local:
                    best_s_local = s
            
            return best_s_local
        
        initial_params = np.concatenate([inner_x, inner_y])
        
        try:
            result = minimize(objective, initial_params, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.fun < best_s:
                inner_x = result.x[:12]
                inner_y = result.x[12:24]
                best_s = result.fun
        except Exception:
            pass
    
    # Final validation and compute outer hexagon
    inner_x = np.asarray(inner_x, dtype=float)
    inner_y = np.asarray(inner_y, dtype=float)
    
    # Recompute optimal outer angle
    inner_vertices = []
    for i in range(12):
        for k in range(6):
            angle = np.radians(60 * k)
            vx = inner_x[i] + np.cos(angle)
            vy = inner_y[i] + np.sin(angle)
            inner_vertices.append((vx, vy))
    inner_vertices = np.array(inner_vertices)
    
    best_s_final = np.inf
    best_outer_angle_final = 0
    for outer_angle_deg in np.arange(0, 30, 0.5):
        outer_angle = np.radians(outer_angle_deg)
        normal_angles = [outer_angle + np.radians(30 + 60 * k) for k in range(6)]
        
        max_proj = 0
        for theta in normal_angles:
            proj = inner_vertices[:, 0] * np.cos(theta) + inner_vertices[:, 1] * np.sin(theta)
            max_proj = max(max_proj, np.max(proj))
        
        s = 2.0 / np.sqrt(3) * max_proj
        if s < best_s_final:
            best_s_final = s
            best_outer_angle_final = outer_angle_deg
    
    # Center the outer hexagon at the centroid of the bounding box of inner vertices
    cx = (np.max(inner_vertices[:, 0]) + np.min(inner_vertices[:, 0])) / 2
    cy = (np.max(inner_vertices[:, 1]) + np.min(inner_vertices[:, 1])) / 2
    
    # Shift all inner hexagons so that the outer hexagon is centered at origin
    # Actually, we should center the outer hexagon at the centroid of the inner hexagon centers
    # for symmetry, but the optimal position might be different
    
    # Let's just center at origin for simplicity (the outer hexagon can be anywhere)
    # The inner hexagons' positions are relative to the outer hexagon center
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle_final])
    outer_hex_side_length = float(best_s_final)
    
    inner_hex_data = np.column_stack([inner_x, inner_y, inner_angle])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END

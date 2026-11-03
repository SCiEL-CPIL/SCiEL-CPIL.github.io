"""Packing 12 unit hexagons in a minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get 6 vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return np.array(verts)
    
    def min_distance_between_hexes(angle_diff_deg):
        """
        Minimum center-to-center distance for two unit hexagons with relative rotation
        angle_diff to just touch (not overlap).
        We use the fact that for regular hexagons, the minimum distance depends on
        the relative orientation. For aligned hexagons, it's 2 (side*2). For 30-degree
        offset, it's different.
        
        Actually, let's compute this properly using the support function / SAT.
        For two convex polygons, they don't overlap iff there exists a separating axis.
        The minimum distance is the minimum over all axes of the gap.
        
        For regular hexagons, the minimum center distance for non-overlap with relative
        rotation θ can be computed. Let me use a simpler approach: 
        The circumradius of a unit hexagon is 1 (distance from center to vertex).
        The inradius (apothem) is sqrt(3)/2.
        
        For two identical regular hexagons with relative rotation θ, the minimum
        center distance to avoid overlap is:
        - If θ = 0: 2 (they can tile, centers 2 apart)
        - General case: we need to check all 12 edge normals (6 from each, but they
          come in opposite pairs, so 6 unique directions from each = 12 total, but
          by symmetry we can reduce)
        
        Let me just compute it numerically for the angles we need.
        """
        # For regular hexagons, the minimum distance for non-overlap with relative
        # rotation θ is given by the maximum over all edge normals of the sum of
        # support functions.
        # 
        # A simpler approach: the minimum distance d(θ) such that two unit hexagons
        # with relative rotation θ don't overlap.
        # 
        # For θ = 0: d = 2 (they tile edge-to-edge)
        # For θ = 30°: the hexagons are rotated so that a vertex of one points at 
        #   the edge midpoint of the other. The distance would be different.
        #
        # Let me compute: the support function of a regular hexagon with side 1 in
        # direction φ is:
        # h(φ) = max over vertices of dot(vertex, direction)
        # 
        # For a unit hexagon centered at origin with angle 0:
        # vertices at angles 0, 60, 120, 180, 240, 300 degrees, radius 1
        # h(φ) = max(cos(φ), cos(φ-60°), cos(φ-120°), cos(φ+120°), cos(φ+60°), cos(φ+180°))
        #       = max over k of cos(φ - k*60°) for k=0..5
        #       = cos(min over k of |φ - k*60°| mod 360, but in [0, 30°])
        #       = cos(φ mod 60°) if φ mod 60° <= 30, else cos(60° - (φ mod 60°))
        #       = cos(min(φ mod 60°, 60° - φ mod 60°))
        #
        # So h(φ) = cos(φ mod 60°) for φ mod 60° in [0, 30°], and cos(60° - (φ mod 60°)) otherwise
        # But by symmetry, h(φ) = cos(min(φ mod 60°, 60° - (φ mod 60°)))
        # Which is just cos(φ) if we consider φ in [0, 30°] by symmetry.
        #
        # Actually, for a regular hexagon, the support function is:
        # h(φ) = cos(φ) for φ in [-30°, 30°] (mod 60° symmetry)
        # More precisely, h(φ) = cos(φ mod 60°) where we take the representative in [-30°, 30°]
        #
        # For two hexagons with relative rotation θ, the minimum center distance is:
        # d_min(θ) = max over all directions φ of [h_1(φ) + h_2(φ)]
        # where h_1 is support function of hex1 and h_2 of hex2 (reflected).
        # 
        # h_1(φ) = h(φ) for hexagon at angle 0
        # h_2(φ) = h(φ - θ) for hexagon at angle θ (but we need the support in direction -φ for the second hexagon to be on the other side)
        # 
        # Actually, for two convex bodies A and B, the minimum distance between them 
        # (when they don't overlap) is related to the support function of their Minkowski sum.
        # Two convex bodies A and B (centered at origins) don't overlap with centers at 
        # distance d apart in direction φ iff d >= h_A(φ) + h_B(-φ) for all φ.
        # So d_min = max_φ [h_A(φ) + h_B(-φ)]
        #
        # For regular hexagons: h(φ) has 6-fold symmetry.
        # h(φ) = cos(φ) for φ in [0, 30°] (taking the hexagon with flat side at top/bottom... 
        # wait, let me be careful about orientation)
        #
        # Let me just compute numerically for the specific angles.
        
        theta = math.radians(angle_diff_deg)
        # Check 12 directions (6 edge normals from each hexagon)
        # Hexagon 1 edge normals at angles: 90°, 150°, 210°, 270°, 330°, 30° (for flat-top)
        # Actually, for a hexagon with vertices at 0°, 60°, ..., the edge normals are at 30°, 90°, 150°, 210°, 270°, 330°
        # Hexagon 2 edge normals are rotated by θ
        
        max_dist = 0
        for i in range(6):
            # Edge normal of hex1
            phi1 = math.radians(30 + i * 60)
            # Edge normal of hex2  
            phi2 = math.radians(30 + i * 60) + theta
            
            # Support function in direction phi for unit hexagon
            def h(phi):
                # Reduce phi to [0, 60) and use symmetry
                phi_mod = phi % (math.pi / 3)  # mod 60 degrees
                if phi_mod > math.pi / 6:  # > 30 degrees
                    phi_mod = math.pi / 3 - phi_mod
                return math.cos(phi_mod)
            
            # For separating axis in direction phi1:
            # gap = h1(phi1) + h2(-phi1) where h2 is for the rotated hexagon
            # h2(-phi1) = h(-phi1 - theta) for the second hexagon
            d1 = h(phi1) + h(-phi1 - theta)
            
            # For separating axis in direction phi2:
            d2 = h(phi2) + h(-phi2 - theta)
            
            max_dist = max(max_dist, d1, d2)
        
        return max_dist
    
    # Precompute minimum distances for common angle differences
    # We'll cache this
    
    def compute_min_side(inner_positions, inner_angles, outer_center, outer_side, outer_angle):
        """Compute the minimum outer hexagon side length that contains all inner hexagons."""
        ox, oy = outer_center
        oa = math.radians(outer_angle)
        
        # The outer hexagon has vertices at:
        # ox + S*cos(oa + k*60°), oy + S*sin(oa + k*60°) for k=0..5
        # where S is the side length = circumradius
        
        # For a point to be inside the hexagon, its projection onto each of the 6 
        # edge normals must be <= the inradius = S * sqrt(3)/2
        # Edge normals are at angles oa + 30° + k*60° for k=0..5
        # The inradius (distance from center to each edge) is S * sqrt(3)/2
        
        inradius = outer_side * math.sqrt(3) / 2
        
        # For each inner hexagon, find the maximum projection of its vertices onto
        # each outer hexagon edge normal direction
        max_projection = 0
        
        for (cx, cy, angle_deg) in zip(inner_positions, inner_angles):
            verts = hex_vertices(cx, cy, angle_deg)
            for i in range(6):
                # Edge normal direction
                phi = oa + math.radians(30 + i * 60)
                # Project all vertices onto this direction
                projections = verts[:, 0] * math.cos(phi) + verts[:, 1] * math.sin(phi)
                # Subtract center projection
                center_proj = ox * math.cos(phi) + oy * math.sin(phi)
                max_proj = np.max(projections) - center_proj
                max_projection = max(max_projection, max_proj)
        
        # The required inradius is max_projection
        # Required side length = max_projection / (sqrt(3)/2) = max_projection * 2/sqrt(3)
        return max_projection * 2.0 / math.sqrt(3)
    
    def objective(params):
        """Objective: minimize the outer hexagon side length."""
        # params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, ox, oy, os, oa]
        # That's 12*3 + 4 = 40 parameters
        
        inner_positions = []
        inner_angles = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2] * 180.0 / math.pi  # convert to degrees
            inner_positions.append((x, y))
            inner_angles.append(a)
        
        ox = params[3*N]
        oy = params[3*N + 1]
        os = params[3*N + 2]
        oa = params[3*N + 3] * 180.0 / math.pi
        
        # Compute the minimum side length needed
        min_side = compute_min_side(inner_positions, inner_angles, (ox, oy), os, oa)
        
        # Add penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                xi, yi = inner_positions[i]
                xj, yj = inner_positions[j]
                ai = inner_angles[i]
                aj = inner_angles[j]
                
                # Distance between centers
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                
                # Minimum required distance
                angle_diff = (ai - aj) % 60.0  # exploit 60-degree symmetry
                angle_diff = min(angle_diff, 60.0 - angle_diff)
                
                # For aligned hexagons (angle_diff = 0), min distance is 2
                # For other angles, it's larger
                # Simple approximation: min_dist = 2.0 + 0.1 * angle_diff
                # More accurate: use the precomputed function
                min_dist = min_distance_between_hexes(angle_diff)
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    penalty += overlap * overlap
        
        # The objective is the actual side length needed + penalty
        # We want to minimize the side length, but the side length is determined by the positions
        # So we just return min_side + penalty
        return min_side + penalty
    
    # Initial guess: honeycomb arrangement
    # 12 hexagons in a 3-4-3-2 or similar arrangement
    # Let's try a 4x3 grid with some offset, or a hexagonal cluster
    
    # Honeycomb: rows of 3, 4, 3, 2 = 12? No, 3+4+3=10, need 2 more
    # Try: 2, 3, 4, 3 = 12
    # Or a 4x3 rectangular grid with hexagonal offsets
    
    # Let's use a simple 4x3 grid with hexagonal packing
    # Row 0: 4 hexagons at y=0
    # Row 1: 3 hexagons at y=sqrt(3) (offset by 1 in x)
    # Row 2: 4 hexagons at y=2*sqrt(3)
    # Row 3: 1 hexagon? No, that's 12 total: 4+3+4+1=12
    
    # Actually, let's do:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    # For hexagonal packing, adjacent rows are offset by 1 in x and sqrt(3) in y
    
    initial_params = []
    
    # Define the arrangement
    # Row 0 (y=0): 3 hexagons at x = -2, 0, 2
    # Row 1 (y=sqrt(3)): 4 hexagons at x = -3, -1, 1, 3
    # Row 2 (y=2*sqrt(3)): 3 hexagons at x = -2, 0, 2
    # Row 3 (y=3*sqrt(3)): 2 hexagons at x = -1, 1
    
    rows = [
        (0, [-2.0, 0.0, 2.0]),
        (math.sqrt(3), [-3.0, -1.0, 1.0, 3.0]),
        (2*math.sqrt(3), [-2.0, 0.0, 2.0]),
        (3*math.sqrt(3), [-1.0, 1.0])
    ]
    
    for (y, xs) in rows:
        for x in xs:
            initial_params.extend([x, y, 0.0])  # angle in radians
    
    # Outer hexagon: centered at origin, side ~8, angle 0
    initial_params.extend([0.0, 0.0, 8.0, 0.0])
    
    initial_params = np.array(initial_params)
    
    # Try optimization
    best_result = None
    best_side = float('inf')
    
    # Try a few different initial configurations
    configs = []
    
    # Config 1: the honeycomb above
    configs.append(initial_params.copy())
    
    # Config 2: 3x4 grid (more compact in one direction)
    params2 = []
    for i in range(3):
        for j in range(4):
            x = (j - 1.5) * 2.0
            y = (i - 1.0) * math.sqrt(3)
            if i % 2 == 1:
                x += 1.0
            params2.extend([x, y, 0.0])
    params2.extend([0.0, 0.0, 8.0, 0.0])
    configs.append(np.array(params2))
    
    # Config 3: circular arrangement
    params3 = []
    for i in range(N):
        angle = 2 * math.pi * i / N
        r = 3.0
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        params3.extend([x, y, 0.0])
    params3.extend([0.0, 0.0, 8.0, 0.0])
    configs.append(np.array(params3))
    
    for config in configs:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective, 
                config, 
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if result.success or result.fun < best_side:
                best_side = result.fun
                best_result = result.x
                
        except Exception as e:
            pass
    
    if best_result is None:
        # Fallback to initial configuration
        best_result = initial_params
    
    # Extract final positions
    inner = []
    for i in range(N):
        x = best_result[3*i]
        y = best_result[3*i + 1]
        a = best_result[3*i + 2] * 180.0 / math.pi
        inner.append([x, y, a])
    
    ox = best_result[3*N]
    oy = best_result[3*N + 1]
    # Compute the actual minimum side length
    inner_positions = [(p[0], p[1]) for p in inner]
    inner_angles = [p[2] for p in inner]
    oa = best_result[3*N + 3] * 180.0 / math.pi
    
    min_side = compute_min_side(inner_positions, inner_angles, (ox, oy), 1.0, oa)
    
    # Round to reasonable precision
    min_side = round(min_side, 6)
    
    return inner, (ox, oy), min_side, oa
# EVOLVE-BLOCK-END


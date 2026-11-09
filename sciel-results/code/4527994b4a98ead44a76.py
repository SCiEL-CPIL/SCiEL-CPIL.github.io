# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, side, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])


def _min_outer_side(inner_hex_data):
    """
    Find the minimum side length of a centered regular hexagon (at origin)
    that contains all inner hexagons.
    
    For a regular hexagon with side s centered at origin, the constraint is:
    For each vertex v of each inner hexagon, it must satisfy all 6 edge constraints.
    
    A regular hexagon with side s, centered at origin, angle 0 has edges:
    The hexagon is the intersection of 6 half-planes.
    For a hexagon with vertices at angles 0, 60, 120, 180, 240, 300 degrees,
    the edge between vertices at angles θ and θ+60 is at distance s*cos(30°) = s*√3/2
    from the center, in the direction of angle θ+30.
    
    Actually, let me think more carefully. A regular hexagon with side length s
    has vertices at distance s from center. The apothem (distance from center to
    midpoint of each edge) is s * √3 / 2.
    
    For a hexagon with one vertex pointing right (angle 0), the edges are:
    - Right edge: x = s (the vertex at angle 0 is at (s, 0), and the edge goes from 
      angle -30 to +30... wait, let me reconsider.
    
    Vertices of hexagon with side s, angle 0:
    v_i = (s*cos(i*60°), s*sin(i*60°)) for i=0..5
    
    v0 = (s, 0)
    v1 = (s/2, s*√3/2)
    v2 = (-s/2, s*√3/2)
    v3 = (-s, 0)
    v4 = (-s/2, -s*√3/2)
    v5 = (s/2, -s*√3/2)
    
    Edge from v0 to v1: midpoint is ((s+s/2)/2, (0+s√3/2)/2) = (3s/4, s√3/4)
    The normal to this edge points outward. The edge direction is v1-v0 = (-s/2, s√3/2)
    The outward normal is perpendicular to this, pointing away from center.
    Rotate edge direction by -90°: (s√3/2, s/2), normalized: (√3/2, 1/2)
    So the constraint is: √3/2 * x + 1/2 * y <= s * √3/2 (apothem)
    Wait, let me verify: at vertex v0 = (s, 0): √3/2*s + 1/2*0 = s√3/2. Yes!
    
    So the 6 constraints for a hexagon with side s, angle 0, centered at origin are:
    For each edge normal direction n_k (k=0..5), n_k · v <= s * √3 / 2
    
    The 6 outward normals (at angles 30°, 90°, 150°, 210°, 270°, 330°):
    n_k = (cos((60k+30)°), sin((60k+30)°))
    
    So for a point (x,y) to be inside:
    cos((60k+30)°)*x + sin((60k+30)°)*y <= s * √3/2 for all k
    
    Therefore: s >= max over all points and all k of [cos((60k+30)°)*x + sin((60k+30)°)*y] / (√3/2)
    
    Also need s >= 0 and the hexagon must actually contain the points.
    """
    # Get all vertices of all inner hexagons
    all_points = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, angle)
        all_points.append(verts)
    
    all_points = np.vstack(all_points)  # shape (72, 2)
    
    # The 6 outward normal directions for angle-0 hexagon
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    # For each normal, compute max projection of all points
    # projections: (72, 6)
    projections = all_points @ normals.T  # (72, 6)
    max_proj = np.max(projections, axis=0)  # (6,)
    
    # s must satisfy: n_k · v_max <= s * √3/2 for all k
    # So s >= max_k (n_k · v_max_k) / (√3/2)
    min_s = np.max(max_proj) / (np.sqrt(3) / 2)
    
    # Also ensure s is at least large enough (should be, but just in case)
    min_s = max(min_s, 0.0)
    
    return min_s


def _objective(params):
    """
    Objective function: minimize outer hexagon side length.
    params: array of shape (12, 2) with (x, y) for each inner hexagon.
    All inner hexagons have angle 0.
    """
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = params[:, 0]
    inner_hex_data[:, 1] = params[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    # Check for overlaps using separating axis theorem
    for i in range(12):
        for j in range(i + 1, 12):
            if _hexagons_overlap(inner_hex_data[i], inner_hex_data[j]):
                return 1e10  # penalty for overlap
    
    s = _min_outer_side(inner_hex_data)
    return s


def _hexagons_overlap(hex1, hex2):
    """
    Check if two unit hexagons overlap using SAT.
    hex1, hex2: (x, y, angle_deg)
    """
    verts1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    verts2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    # Get all edge normals from both hexagons
    def get_normals(verts):
        normals = []
        for i in range(6):
            v1 = verts[i]
            v2 = verts[(i + 1) % 6]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normals.append(normal / norm)
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return False  # separated
    
    return True  # overlapping


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    using optimization to minimize the outer hexagon side length.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    
    # Initial arrangement: hexagonal grid pattern
    # 12 hexagons in a compact arrangement
    # Center + 6 around + 5 more
    sqrt3 = np.sqrt(3)
    
    # Start with a reasonable initial configuration
    # Hexagonal close packing: center at origin, 6 around at distance 2*sqrt(3)/sqrt(3)=2... 
    # Actually for unit hexagons, adjacent centers are at distance 2*apothem = 2*sqrt(3)/2... 
    # Wait, apothem of unit hexagon is sqrt(3)/2. Two adjacent hexagons have centers 
    # at distance 2 * apothem = sqrt(3) apart if they share an edge.
    
    # Let's use a standard hex grid:
    # Row 0: 4 hexagons
    # Row 1: 3 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    # For a hex grid with unit hexagons, horizontal spacing is 1.5 (3/2 * side * 2 = 3)
    # Actually, for pointy-top hexagons with side 1:
    # Horizontal distance between adjacent columns: 3/2 * 2 = 3? No.
    # For flat-top hexagons (angle 0), the width is 2*side = 2, height is sqrt(3)*side = sqrt(3)
    # Adjacent in same row: distance = 2 * apothem_horizontal... 
    # Let me just use: for flat-top hexagons, adjacent centers horizontally are 3 apart
    # (since width is 2, and they share an edge of length 1, so center-to-center is 3/2*2=3? 
    # No: the distance between centers of two adjacent flat-top hexagons sharing a vertical edge
    # is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)? No that's not right either.
    
    # For a regular hexagon with side s, the distance from center to vertex is s.
    # The distance from center to edge midpoint (apothem) is s*sqrt(3)/2.
    # Two hexagons sharing an edge: center-to-center distance = 2 * apothem = s*sqrt(3).
    # For s=1: distance = sqrt(3) ≈ 1.732
    
    # For flat-top orientation (angle 0), the 6 neighbors are at:
    # (±sqrt(3), 0), (±sqrt(3)/2, ±3/2), (-sqrt(3)/2, ±3/2)
    # Wait, let me verify. The 6 vertices are at angles 0,60,120,180,240,300.
    # The 6 edge midpoints are at angles 30,90,150,210,270,330, at distance sqrt(3)/2.
    # So neighbor centers are at:
    # (sqrt(3)/2 * cos(30°), sqrt(3)/2 * sin(30°)) * 2 = (sqrt(3)/2*sqrt(3)/2, sqrt(3)/2*1/2)*2
    # = (3/4, sqrt(3)/4)*2 = (3/2, sqrt(3)/2)
    # Similarly for other angles.
    
    # So the 6 neighbor offsets are:
    # (3/2, sqrt(3)/2), (0, sqrt(3)), (-3/2, sqrt(3)/2), (-3/2, -sqrt(3)/2), (0, -sqrt(3)), (3/2, -sqrt(3)/2)
    
    h = np.sqrt(3) / 2
    w = 3 / 2
    
    # Build a hex grid of 12 hexagons
    # Let's use a compact arrangement:
    # Row 0 (y=0): positions x = -1.5, 0, 1.5, 3.0  (4 hexagons)
    # Row 1 (y=sqrt(3)): positions x = -0.75, 0.75, 2.25  (3 hexagons)
    # Row 2 (y=-sqrt(3)): positions x = -0.75, 0.75, 2.25  (3 hexagons)
    # Row 3 (y=2*sqrt(3)): positions x = 0, 1.5  (2 hexagons)
    # Total: 4+3+3+2 = 12
    
    sqrt3 = np.sqrt(3)
    
    initial_params = np.array([
        # Row 0
        [-1.5, 0],
        [0, 0],
        [1.5, 0],
        [3.0, 0],
        # Row 1 (y = sqrt(3))
        [-0.75, sqrt3],
        [0.75, sqrt3],
        [2.25, sqrt3],
        # Row 2 (y = -sqrt(3))
        [-0.75, -sqrt3],
        [0.75, -sqrt3],
        [2.25, -sqrt3],
        # Row 3 (y = 2*sqrt(3))
        [0, 2*sqrt3],
        [1.5, 2*sqrt3],
    ])
    
    # Try multiple initial configurations and local optimizations
    best_params = None
    best_s = float('inf')
    
    # Generate several initial configurations
    configs = [initial_params]
    
    # Alternative: more symmetric arrangement
    # 1 center, 6 around, 5 more
    config2 = np.array([
        [0, 0],  # center
        # 6 neighbors
        [w, h],
        [0, 2*h],
        [-w, h],
        [-w, -h],
        [0, -2*h],
        [w, -h],
        # 5 more: extend outward
        [2*w, 2*h],
        [0, 4*h],
        [-2*w, 2*h],
        [-2*w, -2*h],
        [2*w, -2*h],
    ])
    configs.append(config2)
    
    # Another: linear-ish
    config3 = np.array([
        [0, 0],
        [w, h],
        [-w, h],
        [-w, -h],
        [w, -h],
        [0, 2*h],
        [0, -2*h],
        [2*w, 2*h],
        [-2*w, 2*h],
        [2*w, -2*h],
        [-2*w, -2*h],
        [0, 4*h],
    ])
    configs.append(config3)
    
    for config in configs:
        if time.time() > deadline - 60:
            break
        
        # Evaluate initial
        s_init = _objective(config)
        if s_init < best_s:
            best_s = s_init
            best_params = config.copy()
        
        # Simple local optimization: coordinate descent with small steps
        params = config.copy()
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for step in step_sizes:
            if time.time() > deadline - 60:
                break
            
            improved = True
            while improved and time.time() < deadline - 60:
                improved = False
                current_s = _objective(params)
                
                for i in range(12):
                    for coord in range(2):
                        # Try positive and negative steps
                        for direction in [1, -1]:
                            new_params = params.copy()
                            new_params[i, coord] += direction * step
                            new_s = _objective(new_params)
                            if new_s < current_s - 1e-10:
                                params = new_params
                                current_s = new_s
                                improved = True
                                if current_s < best_s:
                                    best_s = current_s
                                    best_params = params.copy()
    
    # Final result
    if best_params is None:
        # Fallback to initial
        best_params = initial_params
        best_s = _objective(best_params)
    
    # Construct final hexagon data
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = best_params[:, 0]
    inner_hex_data[:, 1] = best_params[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END

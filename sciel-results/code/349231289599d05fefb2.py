import numpy as np

def construct():
    # Define the 11 hexagons
    # 1 center
    centers = [(0.0, 0.0)]
    # 6 neighbors at distance sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        centers.append((x, y))
    # 4 more at distance 3
    # Angles 30, 90, 150, 210
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        centers.append((x, y))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
        
    # The cluster is fixed.
    # We need to find the minimum side length S of a regular hexagon
    # that contains this cluster.
    # We can rotate the outer hexagon.
    # Due to symmetry of the cluster (almost), 30 degrees is likely optimal.
    # Let's calculate the required S for angle 30.
    
    angle_outer = 30.0
    # Vertices of a unit regular hexagon oriented at 30 degrees
    # (scaled by S)
    # We can compute the support function of the cluster in all directions
    # and require it to be <= support function of the outer hex.
    
    # Simpler: Check if the cluster fits in a hexagon of side S, angle 30.
    # The hexagon is defined by 6 half-planes.
    # For angle 30, the normals to the sides are at angles 0, 60, 120, 180, 240, 300.
    # Wait, if vertices are at 30, 90..., the sides are perpendicular to 0, 60...
    # Side 1 (right): Normal 0. Distance S * cos(0) * something?
    # Distance from center to side is apothem = S * sqrt(3) / 2.
    
    apothem = np.sqrt(3) / 2.0
    
    # Normals to the sides of the outer hexagon (angle 30)
    # The sides are vertical/horizontal?
    # Vertices at 30, 90...
    # Side between 30 and 90: Midpoint angle 60. Normal 60.
    # Side between 90 and 150: Midpoint angle 120. Normal 120.
    # ...
    # So normals are at 60, 120, 180, 240, 300, 0.
    
    normals = [np.deg2rad(60 + k * 60) for k in range(6)]
    
    # Calculate max projection of the cluster in each normal direction
    # Cluster is union of 11 hexagons.
    # We can just project all vertices of all 11 hexagons.
    
    all_vertices = []
    for cx, cy in centers:
        for i in range(6):
            v_angle = np.deg2rad(30 + i * 60)
            vx = cx + np.cos(v_angle)
            vy = cy + np.sin(v_angle)
            all_vertices.append((vx, vy))
            
    all_vertices = np.array(all_vertices)
    
    required_side = 0.0
    for n in normals:
        nx = np.cos(n)
        ny = np.sin(n)
        projections = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
        max_proj = np.max(projections)
        # The outer hexagon's support in direction n is S * apothem / cos(0)?
        # No, the distance from center to the side with normal n is S * sqrt(3) / 2.
        # So we need max_proj <= S * sqrt(3) / 2.
        # S >= max_proj / (sqrt(3) / 2)
        s_req = max_proj / (np.sqrt(3) / 2.0)
        if s_req > required_side:
            required_side = s_req
            
    # Add a tiny epsilon to ensure strict containment if needed
    side = required_side + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

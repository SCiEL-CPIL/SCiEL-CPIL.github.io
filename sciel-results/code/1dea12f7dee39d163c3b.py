import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define candidate configurations of center coordinates
    # 1. 2x6 Honeycomb Strip
    # 2. 3x4 Honeycomb Parallelogram
    # 3. 4x3 Honeycomb Parallelogram
    # 4. Compact Cluster (7 + 5)

    candidates = []

    # Config 1: 2x6 Strip
    centers_26 = []
    for i in range(6):
        centers_26.append((1.5 * i, 0.0))
    for i in range(6):
        centers_26.append((1.5 * i + 0.75, np.sqrt(3) / 2))
    candidates.append(np.array(centers_26))

    # Config 2: 3x4 Parallelogram
    centers_34 = []
    for i in range(3):
        for j in range(4):
            x = 1.5 * i - 1.5 * j
            y = (np.sqrt(3) / 2) * i + (np.sqrt(3) / 2) * j
            centers_34.append((x, y))
    candidates.append(np.array(centers_34))

    # Config 3: 4x3 Parallelogram (Rotated 3x4)
    centers_43 = []
    for i in range(4):
        for j in range(3):
            x = 1.5 * i - 1.5 * j
            y = (np.sqrt(3) / 2) * i + (np.sqrt(3) / 2) * j
            centers_43.append((x, y))
    candidates.append(np.array(centers_43))

    # Config 4: Compact Cluster
    # Center + 6 around + 5 more
    centers_compact = [(0.0, 0.0)]
    for k in range(6):
        angle = np.pi / 3 * k
        centers_compact.append((np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)))
    # Add 5 more to the outer ring, spaced out
    # To minimize max radius, add to the "sides" of the hexagon formed by the outer 6
    # The outer 6 form a hexagon of side sqrt(3).
    # Let's add them at the midpoints of the edges of the outer hexagon, extended outwards?
    # No, just attach them to the outer hexagons.
    # Let's attach 5 to the outer ring hexagons, one each for 5 of them.
    # This will break symmetry.
    # Let's try adding them in a way that keeps the shape roughly hexagonal.
    # Add to indices 0, 1, 2, 3, 4 of the outer ring.
    for k in range(5):
        angle = np.pi / 3 * k
        # Center of the k-th outer hexagon
        cx = np.sqrt(3) * np.cos(angle)
        cy = np.sqrt(3) * np.sin(angle)
        # Move further out by sqrt(3) in the same direction?
        # This creates a "star" shape.
        # Better to move perpendicular to the radius?
        # Let's just use the 3x4 grid, it's likely better for packing into a hexagon.
        pass
    
    # If we want a compact cluster, we can try a specific known packing.
    # But the 3x4 grid is a good candidate.

    best_S = float('inf')
    best_theta = 0.0
    best_centers = None

    # Sweep rotation of the outer hexagon
    # The inner hexagons are assumed to be aligned with the outer hexagon (angle 0)
    # for simplicity, as rotating them individually might not help much if the
    # configuration is symmetric.
    
    thetas = np.linspace(0, np.pi / 3, 100)
    
    for centers in candidates:
        # Center the configuration at (0,0)
        centroid = np.mean(centers, axis=0)
        centers_centered = centers - centroid
        
        for theta in thetas:
            # Normals to the sides of the outer hexagon
            # Side k is perpendicular to angle theta + k * 60 deg
            S_required = 0.0
            for k in range(6):
                normal_angle = theta + k * np.pi / 3
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                
                # Max projection of inner hexagons onto this normal
                # Projection of center c_i: c_i . n
                # Support of unit hexagon in direction n:
                # Since inner hexagons are aligned with outer (angle 0), 
                # and outer is rotated by theta, the relative angle is theta.
                # The support of a unit hexagon (flat-top, aligned with x-axis) 
                # in direction phi is:
                # if phi in [0, 30]: 0.5*cos(phi) + sqrt(3)/2*sin(phi)
                # ... it's the distance to the boundary.
                # Actually, for a unit hexagon with vertices at (1,0), etc,
                # the support function h(phi) is:
                # h(phi) = max( cos(phi), 0.5*cos(phi) + sqrt(3)/2*sin(phi), ... )
                # Due to symmetry, we can just compute the max dot product of vertices.
                
                # Vertices of a unit hexagon (flat-top):
                # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
                verts = np.array([
                    [1.0, 0.0],
                    [0.5, np.sqrt(3)/2],
                    [-0.5, np.sqrt(3)/2],
                    [-1.0, 0.0],
                    [-0.5, -np.sqrt(3)/2],
                    [0.5, -np.sqrt(3)/2]
                ])
                
                # Rotate vertices by -theta? No, the hexagon is fixed at angle 0.
                # The normal is rotated by theta.
                # So we project the fixed vertices onto the rotated normal.
                # proj = verts . n
                projs = verts @ np.array([nx, ny])
                h_support = np.max(projs)
                
                # Max projection of centers
                center_projs = centers_centered @ np.array([nx, ny])
                max_center_proj = np.max(center_projs)
                
                total_max_proj = max_center_proj + h_support
                S_required = max(S_required, total_max_proj)
            
            # The inradius r = S * sqrt(3) / 2
            # So S = r / (sqrt(3) / 2) = 2 * r / sqrt(3)
            S_val = 2.0 * S_required / np.sqrt(3)
            
            if S_val < best_S:
                best_S = S_val
                best_theta = theta
                best_centers = centers_centered

    # Construct the return value
    # inner: list of [x, y, angle_degrees]
    # center: (x, y)
    # side: S
    # angle: theta (in degrees? The interface says angle_degrees for inner, and rotation for outer)
    # "outer hexagon center (x, y), side length and rotation"
    # The rotation of the outer hexagon is best_theta.
    
    inner = []
    for c in best_centers:
        inner.append([c[0], c[1], 0.0]) # Inner hexagons aligned with outer (angle 0 relative to outer? 
                                        # If outer is rotated by theta, and inner are at 0, 
                                        # their absolute angle is theta. 
                                        # But the interface asks for angle_degrees of inner hexagons.
                                        # If we say 0.0, and outer is theta, they are aligned.
                                        # This is consistent.
    
    return inner, (0.0, 0.0), best_S, best_theta
# EVOLVE-BLOCK-END

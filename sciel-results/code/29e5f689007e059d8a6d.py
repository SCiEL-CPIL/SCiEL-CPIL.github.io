import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    # Define a compact packing of 11 unit hexagons
    # Using a "flower" of 7 and adding 4 more
    # Coordinates are centers of the hexagons
    # All hexagons are rotated by 0 degrees initially
    d = np.sqrt(3)
    
    # 7 center
    centers = np.array([
        [0.0, 0.0],
        [d, 0.0],
        [d/2, 1.5],
        [-d/2, 1.5],
        [-d, 0.0],
        [-d/2, -1.5],
        [d/2, -1.5]
    ])
    
    # Add 4 more to make 11
    # Let's try adding them to extend the shape slightly
    # Positions relative to the outer ring
    # This specific arrangement is a "thick" hexagon of side 2 with corners cut?
    # Let's just try a few candidate packings and pick the best container size
    
    candidates = []
    
    # Candidate 1: The 7-flower plus 4 attached to the sides
    # This is getting complicated to hardcode blindly.
    # Instead, let's use a known good packing for 11.
    # A regular hexagon of side 3 contains 19.
    # We can take the 11 central ones of a side 3 hexagon.
    # Side 3 hexagon centers in axial coords (q, r):
    # q in [-2, 2], r in [-2, 2], q+r in [-2, 2]
    # Let's generate all 19 and pick 11 that are central.
    
    axial_coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if -2 <= s <= 2:
                axial_coords.append((q, r))
                
    # Convert axial to cartesian
    def axial_to_cartesian(q, r):
        x = d * (q + r/2.0)
        y = 1.5 * r
        return [x, y]
        
    all_centers = np.array([axial_to_cartesian(q, r) for q, r in axial_coords])
    
    # We want 11. The center is (0,0).
    # Sort by distance from center
    dists = np.linalg.norm(all_centers, axis=1)
    sorted_indices = np.argsort(dists)
    centers_11 = all_centers[sorted_indices[:11]]
    
    # Now we have 11 centers.
    # We need to find the minimal side length S of a regular hexagon
    # that contains all vertices of these 11 hexagons.
    # The container hexagon can be rotated by angle alpha.
    
    # Function to compute minimal S for a given alpha
    def min_side_length(alpha):
        # Generate all vertices of the 11 inner hexagons
        # A unit hexagon centered at (cx, cy) with rotation 0 has vertices at:
        # (cx + cos(k*pi/3), cy + sin(k*pi/3)) for k=0..5
        # But the inner hexagons can also be rotated? 
        # The problem says "each inner hexagon may translate and rotate".
        # However, for a tight packing on a hexagonal lattice, they are usually aligned.
        # If we allow rotation, we might pack tighter, but it's much harder to solve.
        # Given the time limit, I will assume inner hexagons are aligned (angle=0).
        # If I allow inner rotation, I'd need to optimize 11 angles + 11 positions + 1 container angle.
        # That's 34 variables. Too many for a simple search.
        # So I stick to aligned inner hexagons.
        
        all_verts = []
        for cx, cy in centers_11:
            for k in range(6):
                angle = k * np.pi / 3.0
                vx = cx + np.cos(angle)
                vy = cy + np.sin(angle)
                all_verts.append([vx, vy])
        
        all_verts = np.array(all_verts)
        
        # Rotate vertices by -alpha to align with container axes
        # Container hexagon with side S, angle alpha.
        # In its local frame (angle 0), the hexagon is defined by:
        # |x| <= S/2 * sqrt(3) ? No.
        # A regular hexagon with side S, flat-top, centered at origin:
        # Vertices at (S*cos(k*pi/3), S*sin(k*pi/3)).
        # The constraints are:
        # |x| <= S * sqrt(3) / 2 ? No, that's for pointy-top?
        # Let's use the standard inequality for a regular hexagon centered at origin, side S, angle 0 (flat-top).
        # The distance from origin to side is apothem = S * sqrt(3) / 2.
        # The 6 sides are at angles 0, 60, 120, 180, 240, 300.
        # Normal vectors: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2).
        # For a point (x,y) to be inside, the projection onto each normal must be <= apothem.
        # Actually, if we rotate the container by alpha, we rotate the normals.
        # Alternatively, rotate the points by -alpha and check against the standard hexagon.
        
        cos_a = np.cos(-alpha)
        sin_a = np.sin(-alpha)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_verts = all_verts @ rot_mat.T
        
        # Standard hexagon (flat-top) constraints:
        # 1. |x| <= S * sqrt(3) / 2 ? No.
        # Let's derive S from the max projection.
        # The apothem A = S * sqrt(3) / 2.
        # The max projection of any point onto any of the 6 normal directions must be <= A.
        # Normals for flat-top hexagon (vertices at 0, 60, ...):
        # Wait, flat-top means vertices at 0, 60...
        # Normals are at 30, 90, 150...
        # Let's just use the vertices of the container.
        # A point is inside if it is inside the convex hull of the 6 container vertices.
        # But we want the minimal S.
        # S is determined by the maximum "hexagonal radius" of the points.
        # The "hexagonal radius" of a point (x,y) in a flat-top hexagon frame is:
        # r_hex = max( |x|, |y|/sqrt(3) + |x|/2, -|y|/sqrt(3) + |x|/2 ) ? No.
        # Let's use the projection method.
        # Normals for flat-top (side vertical? No, flat top means top side is horizontal).
        # If top side is horizontal, normals are at 90, 150, 210, 270, 330, 30.
        # Let's assume pointy-top (vertex at top). Normals at 0, 60, 120, 180, 240, 300.
        # Let's use pointy-top.
        # Normals:
        # n0 = (1, 0)
        # n1 = (0.5, sqrt(3)/2)
        # n2 = (-0.5, sqrt(3)/2)
        # n3 = (-1, 0)
        # n4 = (-0.5, -sqrt(3)/2)
        # n5 = (0.5, -sqrt(3)/2)
        
        normals = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        # Project rotated vertices onto normals
        projections = rotated_verts @ normals.T
        # The max absolute projection for each normal direction
        # Actually, the constraint is projection <= A.
        # Since the hexagon is centered at origin, we need max(proj) <= A and min(proj) >= -A.
        # So A >= max( max(projections), -min(projections) )
        
        max_proj = np.max(projections, axis=0)
        min_proj = np.min(projections, axis=0)
        
        required_apothems = np.maximum(max_proj, -min_proj)
        max_required_apothem = np.max(required_apothems)
        
        # Apothem A = S * sqrt(3) / 2
        S = 2.0 * max_required_apothem / np.sqrt(3)
        return S

    # Optimize alpha
    best_S = float('inf')
    best_alpha = 0.0
    
    # Try a range of angles
    for alpha in np.linspace(0, np.pi/3, 100):
        if time.time() > deadline:
            break
        S = min_side_length(alpha)
        if S < best_S:
            best_S = S
            best_alpha = alpha
            
    # Refine with golden section search around best_alpha
    if time.time() < deadline - 10:
        a = max(0, best_alpha - np.pi/60)
        b = min(np.pi/3, best_alpha + np.pi/60)
        gr = (np.sqrt(5) - 1) / 2
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = min_side_length(x1)
        f2 = min_side_length(x2)
        
        for _ in range(50):
            if time.time() > deadline:
                break
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = min_side_length(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = min_side_length(x2)
                
        best_alpha = (a + b) / 2
        best_S = min_side_length(best_alpha)
        
    # Construct the result
    inner = []
    for cx, cy in centers_11:
        inner.append([cx, cy, 0.0])
        
    # The container center is (0,0) because we centered the packing at (0,0)
    # But wait, the minimal enclosing hexagon might not be centered at (0,0)
    # We assumed the container is centered at (0,0).
    # Is it optimal to center the container at the centroid of the points?
    # Yes, for a regular polygon, the minimal enclosing instance is centered at the centroid of the points (or close to it).
    # Actually, the minimal enclosing circle is centered at the centroid of the convex hull's vertices? No.
    # For a regular polygon, the optimal center is the centroid of the points?
    # Let's assume (0,0) is good enough for now, as the packing is symmetric.
    
    return inner, (0.0, 0.0), best_S, best_alpha * 180.0 / np.pi


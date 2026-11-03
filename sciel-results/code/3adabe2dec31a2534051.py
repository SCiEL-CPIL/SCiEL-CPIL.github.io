import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # 1. Generate candidate configurations of 11 unit hexagons
    # We will generate a few dense clusters and pick the one that minimizes the container size.
    
    candidates = []
    
    # Candidate 1: The "Flower" (7 hexagons) + 4 more placed at the "corners" of a larger hexagonal grid
    # Flower centers (distance sqrt(3) apart)
    # Let's use a coordinate system where adjacent hex centers are distance d = sqrt(3) apart.
    # Actually, it's easier to use axial coordinates or just Cartesian offsets for a flat-top hex lattice.
    # Flat-top hex lattice vectors:
    # v1 = (sqrt(3), 0)  -> No, that's distance between centers if they touch?
    # Apothem a = sqrt(3)/2. Distance between centers of touching hexes = 2*a = sqrt(3).
    # If flat-top, the "horizontal" neighbor is at distance 2? No.
    # Let's visualize a flat-top hex.
    # Vertices at (1,0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1,0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2).
    # The flat sides are the vertical ones? No, the sides connecting (1/2, sqrt(3)/2) and (-1/2, sqrt(3)/2) is horizontal.
    # So "flat-top" means the top and bottom sides are horizontal.
    # The neighbor to the right shares the vertical side? No, the side from (1,0) to (1/2, sqrt(3)/2) is slanted.
    # The side from (1/2, sqrt(3)/2) to (1,0) is not vertical.
    # The vertical sides are at x=0.5? No.
    # Let's just use the standard: Distance between centers is sqrt(3).
    # Lattice vectors for flat-top:
    # u = (3/2, sqrt(3)/2) * 2? No.
    # Let's use the fact that the lattice is hexagonal.
    # Basis vectors e1, e2 with angle 60 deg, length sqrt(3).
    # e1 = (sqrt(3), 0)
    # e2 = (sqrt(3)/2, 3/2)
    # Check dist: sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3). Correct.
    
    e1 = np.array([np.sqrt(3), 0])
    e2 = np.array([np.sqrt(3)/2, 1.5])
    
    def generate_hex_lattice(r):
        """Generate centers for a hexagonal cluster of radius r (number of rings)"""
        points = []
        # Axial coordinates (q, r)
        for q in range(-r, r + 1):
            for rr in range(-r, r + 1):
                if abs(q + rr) <= r:
                    # Convert axial to cartesian
                    x = e1[0] * q + e2[0] * rr
                    y = e1[1] * q + e2[1] * rr
                    points.append([x, y])
        return np.array(points)

    # H2 has 7 points (radius 1 in axial, i.e., center + 1 ring)
    # Wait, my loop generates points where |q|<=r, |rr|<=r, |q+rr|<=r.
    # If r=1: 
    # q=0: rr in [-1, 1]. (0,-1), (0,0), (0,1). 3 pts.
    # q=1: rr in [-1, 0]. (1,-1), (1,0). 2 pts.
    # q=-1: rr in [0, 1]. (-1,0), (-1,1). 2 pts.
    # Total 7. Correct.
    
    # H3 has 19 points (radius 2)
    # We need 11.
    # Strategy: Take H3 (19 points), remove 8 points that are farthest from the center to minimize the hull?
    # Or take H2 (7 points) and add 4 points from H3 that are "inside" the convex hull of H3 but outside H2?
    # Actually, simply taking the 11 "most central" points of H3 is a good heuristic.
    
    points_h3 = generate_hex_lattice(2)
    # Calculate distance from origin for each point
    dists = np.linalg.norm(points_h3, axis=1)
    # Sort indices by distance
    sorted_indices = np.argsort(dists)
    # Take the first 11
    points_11_h3_subset = points_h3[sorted_indices[:11]]
    candidates.append(points_11_h3_subset)
    
    # Candidate 2: H2 (7 points) + 4 points from H3 that are adjacent to the H2 boundary?
    # H2 points are indices where dist < threshold?
    # Let's just use the 11 closest from H3. It's a very compact cluster.
    
    # Candidate 3: A rectangular grid?
    # 3x4 grid? 12 points. Remove 1.
    # Grid spacing: x-axis spacing sqrt(3), y-axis spacing 1.5?
    # For flat-top, vertical spacing between rows is 1.5 (3/2 * side? No, 3/4 * 2? No.
    # Height of hex is sqrt(3). Row offset is sqrt(3)/2 * 2? No.
    # Vertical distance between centers of stacked flat-top hexes: 3/2 * side?
    # Apothem is sqrt(3)/2. Height is sqrt(3).
    # If we stack them vertically, they don't interlock perfectly without offset.
    # Standard hex grid vertical step is 1.5 * side? No, 3/4 * height?
    # Let's skip the rectangular grid, the hex lattice subset is better.
    
    # 2. Evaluate the minimal container side length for each candidate
    best_S = float('inf')
    best_points = None
    best_angle = 0.0
    
    # Optimization: We can rotate the container. 
    # Since the inner hexagons are regular, we can fix their orientation to 0 
    # and rotate the container, OR rotate the points.
    # Rotating the points is equivalent to rotating the container's angle reference.
    # We will search for the optimal rotation of the container (theta).
    
    def calc_min_hex_side(points):
        """
        Calculate the minimum side length of a regular hexagon 
        (centered at origin) that contains all given points.
        We search for the optimal orientation of the hexagon.
        """
        # Center the points
        centroid = np.mean(points, axis=0)
        pts = points - centroid
        
        min_s = float('inf')
        best_theta = 0
        
        # Search theta from 0 to 60 degrees (symmetry of hexagon)
        # Step 0.5 degrees for precision
        thetas = np.linspace(0, np.pi/3, 181) # 0 to 60 deg
        
        for theta in thetas:
            # Normals of the hexagon sides
            # Angles of normals: theta + 0, 60, 120, 180, 240, 300
            # Actually, the distance from center to side is the apothem 'a'.
            # S = 2 * a / cos(30) = 4 * a / sqrt(3)
            # The constraint is that for all points p, 
            # p . n_k <= a for all 6 normals n_k.
            # So a_req = max_k (max_i p_i . n_k)
            
            a_req = 0
            for k in range(6):
                angle = theta + k * np.pi/3
                n = np.array([np.cos(angle), np.sin(angle)])
                # Project points onto normal
                proj = pts @ n
                max_proj = np.max(proj)
                if max_proj > a_req:
                    a_req = max_proj
            
            s = 4 * a_req / np.sqrt(3)
            if s < min_s:
                min_s = s
                best_theta = theta
                
        return min_s, centroid, best_theta

    for pts in candidates:
        s, centroid, theta = calc_min_hex_side(pts)
        if s < best_S:
            best_S = s
            best_points = pts
            best_centroid = centroid
            best_angle = theta
            
    # The best_points are relative to their own centroid (0,0) in the calc function?
    # No, calc_min_hex_side calculates centroid, then centers pts.
    # But it returns the original centroid.
    # We need to return the points in the global coordinate system.
    # The function returned 'centroid'. The 'pts' inside were centered.
    # So the global points are best_points + best_centroid? 
    # Wait, 'best_points' is the raw array from the loop.
    # 'best_centroid' is the mean of 'best_points'.
    # So the centered points are best_points - best_centroid.
    # The container is at (0,0) if we shift the container? 
    # The interface asks for inner (x,y,angle), center (x,y), side, angle.
    # We can set the outer center to (0,0).
    # Then the inner points should be at their calculated positions relative to (0,0).
    # In calc_min_hex_side, we found that if the container is at (0,0) and rotated by 'best_angle',
    # the points (centered at 0) fit.
    # So we should return the centered points as the inner positions.
    # And the outer center at (0,0).
    
    # However, the 'best_points' variable holds the *uncentered* points from the generator.
    # We need to return the points centered at (0,0) to match the container at (0,0).
    # Let's re-compute the centered points for the winner.
    
    final_pts = best_points - best_centroid
    
    # Construct the inner list
    # Angle of inner hexagons: 0.0 (assumed aligned with container? 
    # Actually, the optimization didn't rotate the inner hexagons, it rotated the container.
    # So relative to the container, the inner hexagons are at angle -best_angle?
    # Or if we fix inner hexagons at 0, the container is at best_angle.
    # The return value 'angle' for the outer hexagon is best_angle.
    # The 'angle' for inner hexagons is 0.0.
    
    inner = []
    for p in final_pts:
        inner.append([p[0], p[1], 0.0])
        
    return inner, (0.0, 0.0), best_S, best_angle

# EVOLVE-BLOCK-END

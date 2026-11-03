import math
import time
import numpy as np

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit

    # 1. Generate a dense, valid lattice packing for N=11 unit hexagons.
    # The optimal packing for 11 hexagons is generally a central hexagon
    # surrounded by 6 neighbors (total 7), plus 4 more.
    # A standard "flower" shape (7 hexagons) has a diameter of 3 units (vertex to vertex).
    # Adding 4 more hexagons to the perimeter creates a shape that is roughly
    # a larger hexagon or a distorted one.
    # 
    # Let's use the standard hexagonal lattice coordinates.
    # Basis vectors for unit hexagons (side 1):
    # v1 = (sqrt(3), 0)  -> distance between centers is sqrt(3)
    # v2 = (sqrt(3)/2, 1.5) -> distance is sqrt(3)
    # Wait, for side 1 hexagons:
    # Apothem a = sqrt(3)/2.
    # Distance between parallel sides = sqrt(3).
    # Centers of touching hexagons are separated by sqrt(3).
    # 
    # Lattice basis:
    # e1 = (sqrt(3), 0)
    # e2 = (sqrt(3)/2, 1.5)
    # Note: |e2| = sqrt(3/4 + 2.25) = sqrt(3) = sqrt(3). Correct.
    
    sqrt3 = math.sqrt(3)
    e1 = np.array([sqrt3, 0.0])
    e2 = np.array([sqrt3/2, 1.5])
    
    # Define the 11 positions in lattice coordinates (i, j) -> pos = i*e1 + j*e2
    # We want a compact cluster.
    # Center at (0,0).
    # Neighbors: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1) -- these are the 6 neighbors.
    # Total 7.
    # We need 4 more.
    # Let's add (2,0), (2,-1), (-2,0), (-2,1)? Or (1,1), (-1,-1), etc.
    # A common shape for 11 is a hexagon of side 2 (which has 1+6+12=19 hexagons) 
    # minus 8 corner ones? 
    # Or simply take the 7 center+ring, and add 4 from the next ring.
    
    # Let's try a specific symmetric configuration:
    # Ring 1 (6 hexes):
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Center: (0,0)
    # This is 7 hexes.
    # Add 4 more to make 11.
    # To keep it compact, we should fill in the "gaps" or extend symmetrically.
    # The convex hull of the 7 hexes is a hexagon with side length 2 (in terms of hex units? No).
    # The outer boundary of the 7-hex cluster is a hexagon with side length 2 (vertices at distance 2 from center? No).
    # Vertices of the 7-hex cluster:
    # The farthest vertices are at the tips of the outer hexagons.
    # Tip of (1,0) hex in direction 0 deg is at (1,0) + (1,0) = (2,0) relative to center hex center?
    # No, center hex is at (0,0). Its vertices are at radius 1.
    # Hex at (sqrt3, 0). Its rightmost vertex is at sqrt3 + 1.
    # So the cluster extends to x = sqrt3 + 1.
    
    # Let's just pick a set of 11 lattice points that form a convex, compact shape.
    # Points:
    pts = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), # 7
        (2, 0), (2, -1), (-2, 0), (-2, 1) # 4 more, extending the "points" of the hexagon
    ]
    
    # Let's verify this shape.
    # It looks like a hexagon with side length 2, but missing the corners?
    # A full hexagon of side 2 (radius 2 in lattice steps) has 1 + 6 + 12 = 19 hexes.
    # The "side 2" hexagon vertices in lattice coords are (2,0), (1,1), (-1,2), (-2,0), (-1,-1), (1,-2).
    # My set:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (2,-1), (-2,0), (-2,1)
    # This is not perfectly symmetric. Let's try a more standard "hexagon minus corners" or "diamond".
    # Actually, for N=11, the optimal shape is often cited as a hexagon of side 2 with 8 corners removed?
    # No, 19 - 8 = 11.
    # The corners of a side-2 hexagon are at (2,0), (1,1), (-1,2), (-2,0), (-1,-1), (1,-2).
    # If we remove the 6 corners, we have 13.
    # If we remove 6 corners and 2 others, we get 11.
    
    # Let's try a different compact set:
    # Center (0,0)
    # 6 neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 4 more: (1,1), (-1,-1), (2,-1), (-2,1)?
    # Let's just use a known good packing for 11.
    # According to literature, the optimal packing of 11 unit circles/hexagons in a hexagon is quite specific.
    # However, we don't need the *global* optimal packing of the inner hexagons to be perfect, 
    # we just need a *valid* packing that is *dense*. The grid search in the current program was struggling to find *any* good one reliably.
    # A lattice packing is guaranteed valid.
    
    # Let's use this set which is very compact:
    lattice_points = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]
    
    # Convert to Cartesian
    positions = []
    for i, j in lattice_points:
        x = i * e1[0] + j * e2[0]
        y = i * e1[1] + j * e2[1]
        positions.append((x, y))
        
    # 2. Optimize the Outer Hexagon
    # We want to find the minimum side length S and rotation angle A, and center (cx, cy)
    # such that all 11 inner hexagons (aligned with axes, i.e., angle 0) are inside.
    
    # Inner hexagon vertices (relative to center):
    inner_verts = []
    for k in range(6):
        theta = k * math.pi / 3
        inner_verts.append((math.cos(theta), math.sin(theta)))
    
    # All absolute vertices of all inner hexagons
    all_verts = []
    for (px, py) in positions:
        for (vx, vy) in inner_verts:
            all_verts.append((px + vx, py + vy))
            
    # Function to calculate the required side length S for a given outer hexagon
    # defined by center (cx, cy) and rotation angle A.
    # The condition for a point P to be inside a regular hexagon with side S, center C, angle A is:
    # Let P' = P - C rotated by -A.
    # P' must satisfy:
    # |x'| <= S * sqrt(3) / 2 ? No.
    # The apothem is a = S * sqrt(3) / 2.
    # The hexagon is the intersection of 3 strips:
    # |x' * cos(30) + y' * sin(30)| <= a
    # |x' * cos(90) + y' * sin(90)| <= a
    # |x' * cos(150) + y' * sin(150)| <= a
    
    def min_side_for_params(cx, cy, angle_deg):
        a_rad = math.radians(angle_deg)
        cos_a = math.cos(a_rad)
        sin_a = math.sin(a_rad)
        
        max_apothem_needed = 0.0
        
        for (px, py) in all_verts:
            # Translate
            dx = px - cx
            dy = py - cy
            # Rotate back to local frame of outer hex
            # x' = dx * cos(a) + dy * sin(a)
            # y' = -dx * sin(a) + dy * cos(a)
            # Wait, rotation by -A:
            # x' = dx * cos(A) + dy * sin(A)
            # y' = -dx * sin(A) + dy * cos(A)
            
            x_loc = dx * cos_a + dy * sin_a
            y_loc = -dx * sin_a + dy * cos_a
            
            # Calculate projections on the 3 normal axes (30, 90, 150 deg)
            # Normal 1: 30 deg
            proj1 = abs(x_loc * math.cos(math.radians(30)) + y_loc * math.sin(math.radians(30)))
            # Normal 2: 90 deg
            proj2 = abs(y_loc)
            # Normal 3: 150 deg
            proj3 = abs(x_loc * math.cos(math.radians(150)) + y_loc * math.sin(math.radians(150)))
            
            apothem_needed = max(proj1, proj2, proj3)
            if apothem_needed > max_apothem_needed:
                max_apothem_needed = apothem_needed
                
        # Side S = 2 * apothem / sqrt(3)
        return 2.0 * max_apothem_needed / math.sqrt(3)

    # 3. Search for optimal parameters
    # The center (cx, cy) should be near the centroid of the inner hexagons.
    # The angle A can be searched. Due to symmetry, 0 to 60 is sufficient.
    
    cx0 = sum(p[0] for p in positions) / N
    cy0 = sum(p[1] for p in positions) / N
    
    best_side = float('inf')
    best_cx = cx0
    best_cy = cy0
    best_angle = 0.0
    
    # Coarse search over angle
    # Since the shape is roughly hexagonal, the optimal angle will likely align with the symmetry axes.
    # We'll scan 0 to 60 degrees with a step of 1 degree, then refine.
    
    for angle_deg in np.arange(0, 60.1, 1.0):
        # For a given angle, the optimal center (cx, cy) minimizes the max projection.
        # This is equivalent to finding the minimum enclosing hexagon for the set of vertices.
        # We can approximate the optimal center by trying a small grid around the centroid.
        
        # A better approach: The optimal center for a fixed angle is the center of the minimum bounding box in the rotated frame?
        # Not exactly, it's the center of the minimum enclosing hexagon.
        # However, for convex shapes, the optimal center is often the centroid or close to it.
        # Let's do a local search on cx, cy for each angle.
        
        # Initial guess: Centroid
        cur_cx = cx0
        cur_cy = cy0
        
        # Local search step size
        step_size = 0.05
        # We can do a few iterations of gradient descent or simple coordinate descent.
        # Since N is small, we can just do a fine grid search around the centroid for the center.
        # But that's 60 angles * (20x20 grid) * 66 vertices * 3 projections = ~ 2.4M ops. Fast enough.
        
        # Let's refine the center using a simple local search.
        # We'll just use the centroid for now, and if we have time, refine.
        # Actually, let's do a small grid search for center for each angle.
        
        # Range for center search: +/- 0.5 from centroid
        # This should cover any reasonable shift.
        
        found_better = False
        for dx in np.arange(-0.5, 0.51, 0.1):
            for dy in np.arange(-0.5, 0.51, 0.1):
                test_cx = cx0 + dx
                test_cy = cy0 + dy
                side = min_side_for_params(test_cx, test_cy, angle_deg)
                if side < best_side:
                    best_side = side
                    best_cx = test_cx
                    best_cy = test_cy
                    best_angle = angle_deg
                    found_better = True
                    
        if time.time() > deadline:
            break
            
    # Refine the best angle and center
    if best_side < float('inf'):
        # Refine angle around best_angle
        for angle_deg in np.arange(best_angle - 1, best_angle + 1.1, 0.1):
            # Refine center around best_cx, best_cy
            for dx in np.arange(-0.1, 0.11, 0.02):
                for dy in np.arange(-0.1, 0.11, 0.02):
                    test_cx = best_cx + dx
                    test_cy = best_cy + dy
                    side = min_side_for_params(test_cx, test_cy, angle_deg)
                    if side < best_side:
                        best_side = side
                        best_cx = test_cx
                        best_cy = test_cy
                        best_angle = angle_deg
                        
        # Final polish
        for angle_deg in np.arange(best_angle - 0.1, best_angle + 0.11, 0.01):
            for dx in np.arange(-0.02, 0.021, 0.005):
                for dy in np.arange(-0.02, 0.021, 0.005):
                    test_cx = best_cx + dx
                    test_cy = best_cy + dy
                    side = min_side_for_params(test_cx, test_cy, angle_deg)
                    if side < best_side:
                        best_side = side
                        best_cx = test_cx
                        best_cy = test_cy
                        best_angle = angle_deg

    # 4. Format Output
    # inner: list of [x, y, angle_degrees]
    # center: [cx, cy]
    # side: best_side
    # angle: best_angle
    
    inner_rows = []
    for (x, y) in positions:
        inner_rows.append([x, y, 0.0]) # All inner hexagons are aligned (angle 0)
        
    return inner_rows, [best_cx, best_cy], best_side, best_angle

if __name__ == "__main__":
    r = construct()
    print(r)

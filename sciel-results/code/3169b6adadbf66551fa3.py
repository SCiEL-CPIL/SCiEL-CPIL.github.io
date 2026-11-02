import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # 1. Generate hexagonal lattice points
    # We want to pack unit hexagons.
    # In a hexagonal packing of unit hexagons (side 1), the distance between centers
    # of adjacent hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    # Let d = sqrt(3).
    # Lattice basis vectors:
    # u = (d, 0)
    # v = (d/2, d*sqrt(3)/2) = (sqrt(3)/2, 3/2)
    
    d = np.sqrt(3)
    u = np.array([d, 0])
    v = np.array([d/2, d*np.sqrt(3)/2])
    
    # We will search for a "hexagonal" subset of the lattice centered at the origin.
    # A hexagonal number arrangement with radius R (number of rings) contains 3R(R+1)+1 hexagons.
    # For R=1: 7 hexagons.
    # For R=2: 19 hexagons.
    # For R=3: 37 hexagons.
    # The seed program used N=11 in a line. Let's try to match or exceed that count with a 2D shape.
    # R=2 gives 19, which is > 11.
    # R=1 gives 7, which is < 11.
    # The task says "n unit hexagons" but doesn't specify n in the prompt text explicitly, 
    # except the seed uses N=11. 
    # Wait, the seed program creates `inner` with 11 entries. 
    # "Place disjoint regular hexagons of side length 1 inside a regular hexagon... minimizing the outer side length"
    # Usually these tasks fix N. The seed code has `N = 11`.
    # I should probably stick to N=11 or find the optimal arrangement for N=11.
    # A hexagonal lattice subset of size 11?
    # Center + 6 neighbors + 4 more?
    # Or maybe the task allows ANY N? "inner rows ... of n unit hexagons".
    # If N is not fixed, we could pack more hexagons? No, "minimizing the outer side length" usually implies a fixed N or maximizing N for a fixed size.
    # Given the seed returns 11 hexagons, it's highly likely N=11 is the target.
    
    # Let's assume N=11.
    # We need to place 11 unit hexagons.
    # A perfect hexagon of side 2 (19 hexagons) is too many.
    # A hexagon of side 1 (7 hexagons) is too few.
    # We need to remove 8 from the 19-hexagon cluster? Or add 4 to the 7-hexagon cluster?
    
    # Let's generate coordinates for a large hexagonal cluster and pick the best 11.
    # Or, simply try specific configurations.
    
    # Configuration 1: Center + 6 neighbors + 4 more from the next ring.
    # To minimize the bounding hexagon, we want the shape to be as "hexagonal" as possible.
    
    # Let's generate points for R=2 (19 points).
    # Indices (i, j) such that max(|i|, |j|, |i+j|) <= R ? No, that's for axial coordinates in a hex grid.
    # Let's use axial coordinates (q, r).
    # Distance from center is max(|q|, |r|, |q+r|) in some metric?
    # Standard hex distance: (|q| + |r| + |q+r|) / 2.
    
    points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= 2: # This generates the 19-point hexagon
                # Convert axial to Cartesian
                # x = d * (q + r/2)
                # y = d * (sqrt(3)/2) * r
                x = d * (q + r/2.0)
                y = d * (np.sqrt(3)/2.0) * r
                points.append((q, r, x, y))
    
    # We have 19 points. We need 11.
    # The "core" 7 points are those with max(|q|,|r|,|s|) <= 1.
    # The outer ring has 12 points.
    # We need to pick 4 from the outer ring to add to the core 7.
    # To minimize the bounding hexagon, we should pick points that are "symmetric" or fill the gaps.
    # However, any 4 points from the outer ring will extend the bounding box.
    # The outer ring points are at distance 2 in hex-metric.
    # The bounding hexagon of the core 7 is side length 2 (in units of center-to-vertex distance? No).
    
    # Let's compute the outer hexagon side length for a given set of centers.
    # The inner hexagons have side 1.
    # The outer hexagon must contain all inner hexagons.
    # This is equivalent to finding the smallest regular hexagon (with same orientation? No, rotation allowed)
    # that contains the union of the inner hexagons.
    # Actually, the outer hexagon can be rotated.
    # But for a hexagonal lattice, the optimal outer hexagon is likely aligned with the lattice.
    
    # If we align the outer hexagon with the lattice (angle 0 or 30 degrees?),
    # The "width" of a unit hexagon in the x-direction is 2 (side 1, flat-to-flat is sqrt(3), point-to-point is 2).
    # Wait, regular hexagon side 1:
    # Width (flat-to-flat) = sqrt(3).
    # Width (point-to-point) = 2.
    
    # If we place centers on a lattice with spacing sqrt(3),
    # The distance between centers is sqrt(3).
    # The hexagons touch.
    
    # Let's define the outer hexagon by its side length L and center (0,0).
    # A point (x,y) is inside the outer hexagon (oriented with flat sides horizontal? or vertices?)
    # Let's assume the outer hexagon is oriented such that its sides are parallel to the inner ones.
    # Inner hexagons are at angle 0.
    # Outer hexagon at angle 0.
    
    # Condition for a point (x,y) to be inside a regular hexagon of side L centered at origin, angle 0:
    # The hexagon is the intersection of 6 half-planes.
    # Normal vectors at 0, 60, 120, 180, 240, 300 degrees.
    # Distance from center to side (apothem) is L * sqrt(3) / 2.
    # So, for each angle theta in [0, 60, ...], |x cos(theta) + y sin(theta)| <= L * sqrt(3) / 2.
    
    # We need to find the minimum L such that for ALL inner hexagons, ALL their vertices are inside.
    # Since the inner hexagons are convex and the outer is convex, it suffices to check the vertices of the inner hexagons.
    # Or, even simpler, for each inner hexagon center (cx, cy) and rotation (0),
    # the maximum extent in direction theta is the center projection + the inner hexagon's extent in that direction.
    # Inner hexagon (side 1, angle 0) extent in direction theta:
    # If theta is 0 (x-axis), extent is 1 (vertex at 1).
    # If theta is 30 (normal to side), extent is sqrt(3)/2 (apothem).
    # General formula for support function of unit hexagon (angle 0) in direction theta:
    # h(theta) = max( cos(theta), cos(theta-60), cos(theta-120), ... ) * 1 ? No.
    # The vertices are at angles 0, 60, 120, 180, 240, 300.
    # h(theta) = max( cos(theta - 0), cos(theta - 60), ..., cos(theta - 300) ).
    
    # So for an inner hexagon at (cx, cy), the constraint on L is:
    # L * sqrt(3)/2 >= cx * cos(theta) + cy * sin(theta) + h_inner(theta)
    # for all 6 theta values (0, 60, 120, 180, 240, 300).
    
    # We can compute the required L for a specific set of centers.
    
    def calc_L(centers):
        max_L = 0.0
        # Inner hexagon support function
        # Vertices at 0, 60, 120, 180, 240, 300
        # h(theta) = max(cos(theta - k*60) for k in 0..5)
        # Note: cos is even, so we can just check absolute values or specific ranges.
        # Actually, since we check all 6 outer normals, and the inner is symmetric,
        # we can just compute the max over the 6 outer directions.
        
        thetas = [0.0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
        
        for cx, cy in centers:
            for theta in thetas:
                # Projection of center
                proj_center = cx * np.cos(theta) + cy * np.sin(theta)
                # Support of inner hexagon in this direction
                # The inner hexagon has vertices at angles 0, 60, ...
                # The support in direction theta is the max dot product with unit vectors at vertex angles.
                # v_k = (cos(k*pi/3), sin(k*pi/3))
                # h = max( cos(theta - k*pi/3) )
                h = 0.0
                for k in range(6):
                    val = np.cos(theta - k * np.pi / 3.0)
                    if val > h:
                        h = val
                # Constraint: L * sqrt(3)/2 >= proj_center + h
                # Note: proj_center + h can be negative, but L must be positive.
                # We take the max over all constraints.
                required_L = (proj_center + h) * 2.0 / np.sqrt(3.0)
                if required_L > max_L:
                    max_L = required_L
        return max_L
    
    # Now, let's find the best subset of 11 points from the 19-point lattice.
    # Or, we can just try the "most compact" shapes.
    # The 19-point hexagon has L=?
    # Let's calculate L for the full 19-point set (R=2).
    centers_19 = [(p[2], p[3]) for p in points]
    L_19 = calc_L(centers_19)
    
    # We need 11 points.
    # Let's try removing the 8 "corner" points of the 19-point hexagon?
    # The 19-point hexagon has 6 corners (dist 2 in axial, e.g., (2,0), (0,2), etc.) and 12 edge midpoints?
    # No, R=2 hexagon:
    # Ring 0: 1
    # Ring 1: 6
    # Ring 2: 12
    # Total 19.
    # To get 11, we keep Ring 0 (1) + Ring 1 (6) = 7.
    # We need 4 more from Ring 2.
    # To minimize L, we should pick points that are "inside" the convex hull of the others?
    # No, we want to minimize the bounding box.
    # The bounding box is determined by the extreme points.
    # If we pick 4 points from Ring 2, they will likely be the extremes.
    # To minimize the max extent, we should pick points that are "balanced".
    # However, any point in Ring 2 has distance 2 in axial metric.
    # The "width" of the shape will be determined by the furthest points.
    
    # Let's brute force select 4 points from the 12 in Ring 2.
    # 12 choose 4 is 495 combinations. Very fast.
    
    ring2_points = [p for p in points if max(abs(p[0]), abs(p[1]), abs(-p[0]-p[1])) == 2]
    # There are 12 such points.
    
    best_L = float('inf')
    best_centers = None
    
    import itertools
    
    # The base 7 points (Ring 0 and 1)
    base_points = [p for p in points if max(abs(p[0]), abs(p[1]), abs(-p[0]-p[1])) <= 1]
    base_centers = [(p[2], p[3]) for p in base_points]
    
    # Iterate over combinations of 4 from ring2
    for combo in itertools.combinations(range(len(ring2_points)), 4):
        if time.time() > deadline:
            break
        extra_centers = [(ring2_points[i][2], ring2_points[i][3]) for i in combo]
        all_centers = base_centers + extra_centers
        L = calc_L(all_centers)
        if L < best_L:
            best_L = L
            best_centers = all_centers
    
    # If we didn't find a better one, or if we want to try other shapes?
    # What if we don't use the lattice?
    # The lattice is likely optimal for hexagons.
    
    # Let's also check if a different rotation of the outer hexagon helps.
    # The current calc_L assumes outer hexagon angle 0.
    # If we rotate the outer hexagon by 15 degrees, the support function changes.
    # But for a hexagonal lattice, the symmetry suggests 0 or 30 is optimal.
    # Let's stick to angle 0 for now, as it aligns with the lattice.
    
    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    # center: [x, y] of outer hexagon
    # side: L
    # angle: 0.0
    
    inner_list = []
    for cx, cy in best_centers:
        inner_list.append([cx, cy, 0.0])
    
    return inner_list, [0.0, 0.0], best_L, 0.0
# EVOLVE-BLOCK-END

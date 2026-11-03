"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard (though this solution is deterministic and instant)
    deadline = time.time() + 140.0
    
    # Strategy: Use a hexagonal lattice arrangement.
    # For N=12, a good configuration is a central hexagon surrounded by a ring of 6,
    # and another ring of 5? No, rings are 1, 6, 12...
    # Let's try a compact cluster.
    # A regular hexagon of side length k contains 1 + 3k(k-1) hexagons?
    # k=1: 1
    # k=2: 7
    # k=3: 19
    # We have 12. So we can fit a "k=2" cluster (7 hexes) plus 5 more.
    # Or we can use a more rectangular/linear hex packing.
    
    # Let's generate a standard hex grid of unit hexagons (side 1).
    # Distance between centers in a hex grid is sqrt(3) * side? No.
    # For regular hexagons of side s, distance between centers in a dense packing is sqrt(3)*s.
    # Let s=1. Distance d = sqrt(3).
    
    # Generate candidate positions on a hex grid.
    # Axial coordinates (q, r).
    # x = d * (q + r/2)
    # y = d * (sqrt(3)/2 * r) -- wait, standard hex grid spacing.
    # Let's use the standard "pointy-top" or "flat-top" coordinates.
    # Flat-top hexagons:
    # Width = 2*s, Height = sqrt(3)*s.
    # Horizontal distance between centers: 3/2 * s = 1.5
    # Vertical distance between centers: sqrt(3) * s = sqrt(3)
    # Offset for odd columns: sqrt(3)/2 * s = sqrt(3)/2
    
    s = 1.0
    dx = 1.5 * s
    dy = math.sqrt(3) * s
    
    # Generate a list of potential center positions in a grid around origin.
    # We need 12 positions.
    # Let's pick a cluster that minimizes the bounding box.
    # A 3x4 rectangular grid of hexes?
    # Or a hexagonal shape of radius 1 (7 hexes) + 5 more?
    
    # Let's try to generate a compact set of 12 points from a hex lattice.
    # We'll iterate through a range of axial coordinates and pick the 12 closest to origin.
    
    candidates = []
    max_range = 10
    for q in range(-max_range, max_range + 1):
        for r in range(-max_range, max_range + 1):
            # Convert axial to pixel (flat-top)
            x = dx * (q + r / 2.0)
            y = dy * r
            candidates.append((q, r, x, y))
            
    # Sort by distance from origin to get the most compact cluster
    candidates.sort(key=lambda p: p[2]**2 + p[3]**2)
    
    # Take the first 12
    selected = candidates[:N]
    
    # Extract positions
    positions = [(p[2], p[3]) for p in selected]
    
    # Calculate the bounding hexagon.
    # The inner hexagons are axis-aligned (angle 0).
    # We need to find the smallest regular hexagon (side L, angle A, center C) that contains all these unit hexagons.
    # Since the inner hexagons are fixed at angle 0, and the outer one can rotate,
    # we can assume WLOG that the outer hexagon is aligned with the inner ones (angle 0) 
    # because the set of unit hexagons is not rotationally symmetric in a way that would benefit 
    # from a rotated container unless the cluster itself is rotated. 
    # However, rotating the cluster might help if the bounding shape is rectangular.
    # For a hex lattice cluster, the natural bounding shape is a hexagon.
    # Let's assume outer angle = 0 and center = centroid of the cluster.
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    
    # Center the cluster at origin to simplify
    cx_mean = sum(xs) / len(xs)
    cy_mean = sum(ys) / len(ys)
    
    centered_positions = [(x - cx_mean, y - cy_mean) for x, y in positions]
    
    # Now find the minimal side length L of a regular hexagon centered at (0,0) with angle 0
    # that contains all centered_positions (which are centers of unit hexagons of side 1).
    
    # A point (px, py) is inside a regular hexagon of side L centered at origin (flat-top, angle 0)
    # if:
    # |px| <= L
    # |py| <= L * sqrt(3) / 2
    # AND |sqrt(3)*px + py| <= L * sqrt(3)  ? No, that's for pointy-top or different orientation.
    
    # Let's define the hexagon vertices for flat-top, side L, angle 0:
    # (L, 0), (L/2, L*sqrt(3)/2), (-L/2, L*sqrt(3)/2), (-L, 0), (-L/2, -L*sqrt(3)/2), (L/2, -L*sqrt(3)/2)
    
    # The constraints for a point (x,y) to be inside are:
    # 1. x <= L
    # 2. x >= -L
    # 3. y <= L * sqrt(3) / 2
    # 4. y >= -L * sqrt(3) / 2
    # 5. sqrt(3)*x + y <= L * sqrt(3)  (Top-Right edge) -> y <= sqrt(3)(L - x)
    # 6. sqrt(3)*x - y >= -L * sqrt(3) (Bottom-Right edge) -> y >= sqrt(3)(x - L)
    
    # Wait, the unit hexagons have radius (distance to vertex) = 1.
    # The "center" of the unit hexagon is at (px, py).
    # The unit hexagon extends 1 unit from its center in the direction of its vertices.
    # So, effectively, we need the Minkowski sum of the cluster of points and a unit hexagon to fit inside the outer hexagon.
    # This is equivalent to finding the smallest outer hexagon that contains the "expanded" cluster.
    # Expanding the cluster by a unit hexagon (side 1) is the same as checking if the distance from the center of the outer hexagon
    # to the "supporting lines" of the outer hexagon is sufficient to cover the unit hexagon's extent.
    
    # Simpler approach:
    # For a given L, is the configuration valid?
    # The condition that a unit hexagon (center cx, cy, angle 0) is inside a large hexagon (center 0,0, angle 0, side L)
    # is equivalent to:
    # For each of the 6 inequalities defining the large hexagon, the maximum value of the linear form over the unit hexagon
    # must be <= the RHS of the inequality.
    
    # Inequalities for Large Hexagon (flat-top, side L, center 0,0):
    # h1(x,y) = x <= L
    # h2(x,y) = -x <= L  => x >= -L
    # h3(x,y) = y <= L*sqrt(3)/2
    # h4(x,y) = -y <= L*sqrt(3)/2 => y >= -L*sqrt(3)/2
    # h5(x,y) = sqrt(3)*x + y <= L*sqrt(3)  (Wait, let's verify vertices)
    # Vertices: (L,0), (L/2, L*sqrt(3)/2).
    # Line through (L,0) and (L/2, L*sqrt(3)/2):
    # Slope = (L*sqrt(3)/2 - 0) / (L/2 - L) = (L*sqrt(3)/2) / (-L/2) = -sqrt(3).
    # Eq: y - 0 = -sqrt(3)(x - L) => y = -sqrt(3)x + L*sqrt(3) => sqrt(3)x + y = L*sqrt(3).
    # So constraint: sqrt(3)*x + y <= L*sqrt(3).
    
    # h6(x,y) = -sqrt(3)*x + y <= L*sqrt(3) ?
    # Line through (L/2, L*sqrt(3)/2) and (0, L*sqrt(3))? No, vertex is (-L/2, L*sqrt(3)/2).
    # Line through (L/2, L*sqrt(3)/2) and (-L/2, L*sqrt(3)/2) is y = L*sqrt(3)/2. That's h3.
    # Line through (-L/2, L*sqrt(3)/2) and (-L, 0):
    # Slope = (0 - L*sqrt(3)/2) / (-L - (-L/2)) = (-L*sqrt(3)/2) / (-L/2) = sqrt(3).
    # Eq: y - 0 = sqrt(3)(x + L) => y = sqrt(3)x + L*sqrt(3) => -sqrt(3)x + y = L*sqrt(3).
    # So constraint: -sqrt(3)*x + y <= L*sqrt(3).
    
    # Now, for a unit hexagon centered at (cx, cy), what is the max of these linear forms?
    # The unit hexagon is also flat-top, side 1.
    # Its support function in direction (a,b) is the max of a*x + b*y over the hexagon.
    # For a regular hexagon of side s centered at origin, the support function is:
    # s * max( |a|/2 + |b|/(2*sqrt(3)) * something? )
    # Actually, it's easier to note that the unit hexagon is the Minkowski sum of 3 line segments?
    # Or just calculate the max over the 6 vertices of the unit hexagon.
    # Vertices of unit hexagon (side 1, angle 0) relative to its center (0,0):
    # v0: (1, 0)
    # v1: (0.5, sqrt(3)/2)
    # v2: (-0.5, sqrt(3)/2)
    # v3: (-1, 0)
    # v4: (-0.5, -sqrt(3)/2)
    # v5: (0.5, -sqrt(3)/2)
    
    # For a linear form f(x,y) = ax + by, the max over the hexagon is max over vertices.
    # So for each constraint i and each inner hexagon j, we check:
    # a_i * (cx_j + vx_k) + b_i * (cy_j + vy_k) <= RHS_i
    # where (vx_k, vy_k) are the vertices of the unit hexagon.
    
    # This is a linear programming problem or just a simple max calculation.
    # Since we want to minimize L, we can binary search on L or calculate the required L directly.
    # L must be at least:
    # L >= max_j max_k ( a_i * cx_j + b_i * cy_j + a_i * vx_k + b_i * vy_k ) / (coeff of L in RHS_i)
    
    # Let's compute the required L for each constraint i.
    
    # Coefficients (a, b) and RHS coeff (c) for L:
    # 1. x <= L       -> a=1, b=0, c=1
    # 2. -x <= L      -> a=-1, b=0, c=1
    # 3. y <= L*sqrt(3)/2 -> a=0, b=1, c=sqrt(3)/2
    # 4. -y <= L*sqrt(3)/2 -> a=0, b=-1, c=sqrt(3)/2
    # 5. sqrt(3)x + y <= L*sqrt(3) -> a=sqrt(3), b=1, c=sqrt(3)
    # 6. -sqrt(3)x + y <= L*sqrt(3) -> a=-sqrt(3), b=1, c=sqrt(3)
    
    constraints = [
        (1, 0, 1),
        (-1, 0, 1),
        (0, 1, math.sqrt(3)/2),
        (0, -1, math.sqrt(3)/2),
        (math.sqrt(3), 1, math.sqrt(3)),
        (-math.sqrt(3), 1, math.sqrt(3))
    ]
    
    unit_vertices = [
        (1, 0),
        (0.5, math.sqrt(3)/2),
        (-0.5, math.sqrt(3)/2),
        (-1, 0),
        (-0.5, -math.sqrt(3)/2),
        (0.5, -math.sqrt(3)/2)
    ]
    
    required_L = 0.0
    
    for a, b, c in constraints:
        max_val = -1e9
        for cx, cy in centered_positions:
            for vx, vy in unit_vertices:
                val = a * (cx + vx) + b * (cy + vy)
                if val > max_val:
                    max_val = val
        # max_val <= c * L  =>  L >= max_val / c
        if c > 0:
            L_req = max_val / c
        else:
            # Should not happen with this setup
            L_req = 0
        if L_req > required_L:
            required_L = L_req
            
    # The outer hexagon side length is required_L.
    # Center is (cx_mean, cy_mean).
    # Angle is 0.
    
    # Construct the inner list: [x, y, angle]
    inner = []
    for x, y in positions:
        inner.append([x, y, 0.0])
        
    outer_center = (cx_mean, cy_mean)
    outer_side = required_L
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END

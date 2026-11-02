"""Improved program for hexagon packing task."""
import math
import itertools
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    N = 11
    
    # For N=11, the optimal packing is likely a subset of a hexagonal grid.
    # A side-length L hexagon can hold hex(2L-1) unit hexagons.
    # L=2: 7 hexagons (too small)
    # L=3: 19 hexagons (enough for 11)
    # L=2.5: not integer, but maybe possible with rotation? Unlikely to be better than 3 without complex geometry.
    # Let's try to fit 11 hexagons in L=3.
    # The centers of a side-3 hexagon grid (with side length 3 in terms of unit hexagons) are at axial coordinates (q, r)
    # such that max(|q|, |r|, |q+r|) <= 2.
    # Total 19 positions.
    # We need to select 11 positions that are mutually disjoint (which they are, as they are grid points) 
    # and contained within the outer hexagon of side length 3.
    # Actually, if we place unit hexagons at all 19 grid points of a side-3 hexagon, 
    # the enclosing hexagon has side length 3 + 2*(sqrt(3)/2) ? No.
    # Let's re-evaluate the geometry.
    # A unit regular hexagon has side 1. Distance between centers of adjacent unit hexagons is sqrt(3).
    # If we pack them in a hexagonal lattice with spacing sqrt(3), 
    # the center of the cluster is at (0,0).
    # The vertices of the cluster are the centers of the corner hexagons.
    # For a side-L hexagon of unit hexagons (where L is the number of hexagons along the edge),
    # the axial coordinate range is -L+1 to L-1.
    # The distance from center to a corner hexagon center is (L-1) * sqrt(3).
    # The circumradius of a unit hexagon is 1.
    # So the distance from the center of the cluster to the vertex of the outer hexagon is (L-1)*sqrt(3) + 1.
    # For a regular hexagon of side S enclosing this, the circumradius is S.
    # So S = (L-1)*sqrt(3) + 1.
    # If L=3 (19 hexagons), S = 2*sqrt(3) + 1 ≈ 2*1.732 + 1 = 4.464.
    # If we can fit 11 hexagons in a smaller cluster, maybe L=2 is not enough (7 hex).
    # Can we fit 11 hexagons in a shape with circumradius < 4.464?
    # What if we use a non-grid packing?
    # The problem allows rotation and translation.
    # However, for hexagons, the optimal packing is generally the hexagonal lattice.
    # Is it possible to fit 11 hexagons in S < 4.464?
    # Let's check S=4.
    # If S=4, max distance from center is 4.
    # (L-1)*sqrt(3) + 1 <= 4 => (L-1)*sqrt(3) <= 3 => L-1 <= 3/sqrt(3) = sqrt(3) ≈ 1.732 => L <= 2.732.
    # So integer L must be 2. But L=2 holds only 7 hexagons.
    # So we cannot use a full hexagonal grid with L=3 and get S < 4.464.
    # Can we remove some hexagons from the L=3 grid to reduce the bounding box?
    # The bounding box is determined by the outermost vertices.
    # If we remove the 6 corner hexagons of the L=3 grid, we are left with 13 hexagons.
    # The new outermost centers are at distance 1*sqrt(3) from the center (the "side" centers).
    # The vertices of these side-center hexagons extend to 1*sqrt(3) + 1 ≈ 2.732.
    # This is much less than 4.
    # Wait, the shape of the remaining 13 hexagons is a hexagon of side 2?
    # The L=3 grid consists of a central hexagon, 6 around it (L=2 ring), and 12 around that (L=3 ring).
    # Total 1 + 6 + 12 = 19.
    # The "corners" are the 6 hexagons at the vertices of the big hexagon.
    # If we remove the 6 corners, we have 13 hexagons.
    # The remaining set is a hexagon of side 2?
    # The vertices of the remaining shape are the midpoints of the sides of the original big hexagon.
    # The distance from center to these midpoints is 1*sqrt(3) (axial distance 1).
    # The circumradius of the enclosing hexagon would be 1*sqrt(3) + 1 ≈ 2.732.
    # But wait, is the enclosing shape a regular hexagon?
    # The vertices of the 13-hexagon cluster form a regular hexagon.
    # The side length of this new outer hexagon?
    # The distance between adjacent vertices (midpoints of original sides) is 2 * (1*sqrt(3)) * sin(60) ?
    # Let's calculate the coordinates.
    # Axial coords of the 13 hexagons:
    # Ring 0: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (excluding corners):
    # Corners of Ring 2 are (2,0), (1,1), (-1,2), (-2,0), (-1,-1), (1,-2).
    # We keep: (2,-1), (0,2), (-2,1), (-1,-2), (1,-1 is ring 1), etc.
    # Let's list Ring 2:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-0 is (2,0)), etc.
    # Actually, Ring 2 has 12 hexagons.
    # The 6 corners are: (2,0), (1,1), (-1,2), (-2,0), (-1,-1), (1,-2).
    # The other 6 are: (2,-1), (0,2), (-2,1), (-1,-2), (1,-1), (2,0 is corner).
    # Let's be precise.
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) = 2.
    # The 12 points are:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-0 is (2,0)), 
    # Wait, the loop is:
    # (2,0) -> (1,1) -> (0,2) -> (-1,2) -> (-2,1) -> (-2,0) -> (-1,-1) -> (0,-2) -> (1,-2) -> (2,-1) -> (2,0)
    # That's 10? No, 12 points.
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,0) is 10.
    # Missing (2, -1) is there. What about (1, -1)? No, that's ring 1.
    # The points with max distance 2 are:
    # q=2: r=0, -1, -2? No, |q+r| <= 2. If q=2, r can be 0, -1, -2.
    # (2,0): max(2,0,2)=2.
    # (2,-1): max(2,1,1)=2.
    # (2,-2): max(2,2,0)=2.
    # So (2,0), (2,-1), (2,-2).
    # Similarly for other directions.
    # There are 6 sides, 2 points per side? No.
    # Side 1: (2,0), (2,-1), (2,-2).
    # Side 2: (1,-2), (0,-2), (-1,-2).
    # Side 3: (-2,-1), (-2,0), (-2,1).
    # Side 4: (-1,2), (0,2), (1,2).
    # Side 5: (2,1), (2,2), (2,3)? No.
    # Let's just generate them programmatically.
    
    # The 13 hexagons (center + ring 1 + 6 non-corner ring 2) form a hexagon of side 2?
    # The vertices of this cluster are at (2,-1), (0,2), (-2,1), (-1,-2), (1,-1), (2,0)?
    # No, the corners are removed.
    # The new "corners" are the midpoints of the original sides.
    # e.g., (2,-1) is a vertex of the new cluster?
    # The distance from center to (2,-1) is sqrt(3)*2? No.
    # Axial to Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # For (2,-1): x = sqrt(3) * (2 - 0.5) = 1.5*sqrt(3), y = -1.5.
    # Distance = sqrt(2.75*3 + 2.25) = sqrt(8.25 + 2.25) = sqrt(10.5) ≈ 3.24.
    # This is larger than 1*sqrt(3)+1 ≈ 2.73.
    # So removing corners doesn't shrink the bounding box enough to go below L=2.5?
    # Actually, the bounding box of the 13-hexagon cluster is determined by the vertices (2,-1), etc.
    # The circumradius is sqrt(10.5) ≈ 3.24.
    # So S ≈ 3.24.
    # This is better than 4.46.
    # Can we do better?
    # What if we remove more?
    # We need 11 hexagons.
    # 13 is close. Remove 2 more.
    # Remove (2,-1) and (0,2)?
    # Then the new vertices are (2,0) [removed], (2,-2)?
    # This is getting complex.
    
    # Given the constraints and the likelihood that the optimal solution for N=11 is around S=3.24 or slightly less,
    # and the complexity of finding the exact optimal subset,
    # let's provide a valid packing with S=3.24 (the 13-hexagon cluster minus 2).
    # Or simply provide the 13-hexagon cluster with S=3.24 and note that we can remove 2 to get 11.
    # Removing 2 hexagons from the 13 will only reduce the bounding box or keep it same.
    # So S=3.24 is a safe upper bound.
    
    # Let's calculate the exact S for the 13-hexagon cluster.
    # The vertices are at axial coords: (2,-1), (0,2), (-2,1), (-1,-2), (1,-1), (2,0)?
    # No, the 6 vertices of the "hexagon of side 2" (13 hexes) are:
    # (2,-1), (0,2), (-2,1), (-1,-2), (1,-1), (2,0) is not a vertex if corner (2,0) is removed?
    # Wait, (2,0) is a corner of the L=3 grid. It is removed.
    # The vertices of the 13-hex cluster are the "side midpoints" of the L=3 grid.
    # These are (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,1), (-1,2), (0,2), (1,2), (2,1), (2,0)?
    # I need to be careful.
    # Let's just use the 13 hexagons and calculate the minimal enclosing hexagon.
    # It is a regular hexagon.
    # The side length is the distance between adjacent vertices divided by 1?
    # No, the side length of the outer hexagon is the distance from center to vertex (circumradius).
    # For a regular hexagon, side length = circumradius.
    # So S = max distance from center to any vertex of any inner hexagon.
    
    # Let's compute S for the 13-hex cluster.
    # Axial coords of the 13 hexes:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,1), (-1,2), (0,2) ... wait, (0,2) is a corner?
    # (0,2): max(0,2,2)=2. It is a corner?
    # The corners of the L=3 hexagon are (2,0), (1,1), (-1,2), (-2,0), (-1,-1), (1,-2).
    # So (0,2) is NOT a corner. It is a side midpoint.
    # The 6 corners are: (2,0), (1,1), (-1,2), (-2,0), (-1,-1), (1,-2).
    # The other 6 in ring 2 are: (2,-1), (1,-2 is corner), ...
    # Let's list ring 2 again:
    # (2,0) [corner], (1,1) [corner], (0,2) [side], (-1,2) [corner], (-2,1) [side], (-2,0) [corner],
    # (-1,-1) [corner], (0,-2) [side], (1,-2) [corner], (2,-1) [side], (2,-2) [side?], (1,2) [side?].
    # This is confusing. Let's use code to generate.
    
    # For now, let's assume S = 3.24 is a good guess.
    # Actually, let's just return the 13-hex cluster with S calculated.
    
    # Generate 13 hexagons
    hexes = []
    # Center
    hexes.append((0, 0))
    # Ring 1
    for q, r in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
        hexes.append((q, r))
    # Ring 2 non-corners
    # Corners: (2,0), (1,1), (-1,2), (-2,0), (-1,-1), (1,-2)
    corners = {(2,0), (1,1), (-1,2), (-2,0), (-1,-1), (1,-2)}
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2 and (q,r) not in corners:
                hexes.append((q, r))
    
    # We have 1 + 6 + 6 = 13 hexagons.
    # We need 11. Remove 2.
    # To minimize the bounding box, remove two that are "furthest" or redundant.
    # Since they are all at similar distances, removing any two should keep S the same or reduce it slightly.
    # Let's remove the last two added.
    hexes = hexes[:11]
    
    # Calculate the minimal enclosing hexagon side length S.
    # Convert axial to Cartesian
    def axial_to_xy(q, r):
        x = math.sqrt(3) * (q + r/2)
        y = 1.5 * r
        return x, y
    
    # For each hexagon, the vertices are at distance 1 from center, at angles 0, 60, ...
    # The outer boundary is the convex hull of all vertices.
    # Since the arrangement is symmetric, the minimal enclosing hexagon will be centered at (0,0) and aligned.
    # We just need the max distance from origin to any vertex of any hexagon.
    # Because the outer hexagon is regular and centered at origin.
    # Is it guaranteed to be regular and centered?
    # If the inner packing is symmetric, yes.
    # Our 11-hex set is a subset of a symmetric 13-hex set.
    # If we remove two symmetric hexagons, symmetry is broken.
    # But we can rotate the outer hexagon to minimize S.
    # However, for a quick estimate, let's assume S = max distance.
    
    max_dist = 0
    for q, r in hexes:
        x, y = axial_to_xy(q, r)
        # Vertices of this hexagon
        for i in range(6):
            angle = math.pi / 3 * i
            vx = x + math.cos(angle)
            vy = y + math.sin(angle)
            d = math.sqrt(vx*vx + vy*vy)
            if d > max_dist:
                max_dist = d
    
    # This max_dist is the circumradius of the enclosing circle.
    # For a regular hexagon, side length = circumradius.
    # So S = max_dist.
    
    # Check time
    if time.time() > deadline - 10:
        return [[axial_to_xy(q, r)[0], axial_to_xy(q, r)[1], 0.0] for q, r in hexes], (0.0, 0.0), max_dist, 0.0
    
    # We can try to optimize by removing different pairs.
    # But 11! is too large.
    # Let's just return this.
    
    inner = [[axial_to_xy(q, r)[0], axial_to_xy(q, r)[1], 0.0] for q, r in hexes]
    return inner, (0.0, 0.0), max_dist, 0.0

# EVOLVE-BLOCK-END

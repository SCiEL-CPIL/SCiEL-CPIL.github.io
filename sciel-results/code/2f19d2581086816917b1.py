import math
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # The current program uses a randomized local search which can be slow and unstable.
    # This strategy uses a deterministic geometric construction.
    # We know that 11 unit hexagons can be packed in a "3-3-3-2" cluster.
    # This cluster forms a shape that is roughly hexagonal itself.
    # By aligning the outer hexagon with the lattice, we can compute the exact minimal size.
    
    # 1. Define the lattice positions for the 11 hexagons.
    # The distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # Let d = sqrt(3).
    # We use axial coordinates (q, r) for the triangular lattice.
    # The 11 hexagons in a 3-3-3-2 cluster (centered as much as possible) can be defined.
    # Let's center the cluster around the origin.
    # A 3x3 block of hexagons has 9 hexagons. We add 2 more to make 11.
    # Actually, the densest packing of 11 hexagons is often a "flower" shape (1 center, 6 ring, 4 more).
    # Let's try the 1-6-4 configuration (center + 1st ring + part of 2nd ring).
    
    d = math.sqrt(3)
    eps = 1e-9
    
    # Positions in Cartesian coordinates
    # Center at (0,0)
    positions = [(0.0, 0.0)]
    
    # 1st ring (6 hexagons)
    for i in range(6):
        angle = math.pi / 3 * i
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        positions.append((x, y))
    
    # 2nd ring (we need 4 more to make 11 total: 1 + 6 + 4 = 11)
    # The 2nd ring hexagons are at distance 2*d from center? No, distance is d * sqrt(1^2 + 1^2 + 1^2)?
    # In axial coordinates, distance from center is max(|q|, |r|, |q+r|).
    # 1st ring: distance 1.
    # 2nd ring: distance 2.
    # Let's just place 4 hexagons in the 2nd ring symmetrically.
    # The 2nd ring has 12 positions. We pick 4 that are spaced out.
    # Indices in 2nd ring (angle 0 to 330, step 30 deg? No, 360/12 = 30 deg).
    # Wait, the 2nd ring hexagons are not all at the same distance from origin in Cartesian?
    # In a triangular lattice, the "distance" is graph distance.
    # Let's use the explicit Cartesian coordinates for a standard hex grid.
    
    # Let's restart the coordinate generation to be precise.
    # Hexagon side length s = 1.
    # Width (flat-to-flat) w = sqrt(3).
    # Height (point-to-point) h = 2.
    
    # Let's place them on a grid where x-axis is horizontal.
    # Flat-top hexagons.
    # Center (0,0).
    # Neighbors:
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    
    pos = []
    pos.append((0.0, 0.0)) # 1
    
    # Ring 1
    pos.append((d, 0.0)) # 2
    pos.append((d/2, 1.5)) # 3
    pos.append((-d/2, 1.5)) # 4
    pos.append((-d, 0.0)) # 5
    pos.append((-d/2, -1.5)) # 6
    pos.append((d/2, -1.5)) # 7
    
    # Ring 2 (need 4 more)
    # Let's pick 4 from the next shell.
    # The next shell positions (distance 2 in graph steps):
    # (2d, 0), (3d/2, 1.5), (d, 3), (-d/2, 3), (-3d/2, 1.5), (-2d, 0), ...
    # Let's pick (2d, 0), (d, 3), (-d, 3), (-2d, 0) ? No, that's not symmetric enough or might be sparse.
    # Let's pick (3d/2, 1.5), (3d/2, -1.5), (-3d/2, 1.5), (-3d/2, -1.5).
    # These are "corner" extensions.
    
    pos.append((1.5*d, 1.5)) # 8
    pos.append((1.5*d, -1.5)) # 9
    pos.append((-1.5*d, 1.5)) # 10
    pos.append((-1.5*d, -1.5)) # 11
    
    # This forms a roughly rectangular/hexagonal shape.
    # Let's verify disjointness.
    # All pairs must be at least d apart.
    # (0,0) to (1.5d, 1.5): dist^2 = 2.25*3 + 2.25 = 6.75 + 2.25 = 9. dist=3 > d. OK.
    # (d,0) to (1.5d, 1.5): dist^2 = (0.5d)^2 + 1.5^2 = 0.75 + 2.25 = 3. dist=sqrt(3)=d. OK (touching).
    # (d/2, 1.5) to (1.5d, 1.5): dist = d. OK (touching).
    # (d/2, 1.5) to (1.5d, -1.5): dist^2 = (d)^2 + 3^2 = 3 + 9 = 12. dist=sqrt(12) > d. OK.
    # This configuration is valid.
    
    # 2. Calculate the minimal bounding hexagon for this set of points.
    # We assume the outer hexagon is also flat-top (angle 0).
    # The outer hexagon is defined by 6 half-planes.
    # Normal vectors for flat-top hexagon (pointing outwards):
    # Angles: 0, 60, 120, 180, 240, 300 degrees?
    # For a flat-top hexagon centered at origin:
    # Vertices are at angles 0, 60, 120, 180, 240, 300.
    # The sides are perpendicular to these vectors? No.
    # The sides are horizontal and slanted.
    # Let's define the constraints by the apothems.
    # The distance from center to each side (apothem) is R_apo.
    # Side length S = R_apo / (sqrt(3)/2) = 2*R_apo/sqrt(3).
    
    # For a flat-top hexagon, the 6 outward normals are at angles:
    # 30, 90, 150, 210, 270, 330 degrees.
    # Wait, if vertices are at 0, 60..., then the normal to the side between 0 and 60 is at 30.
    # Yes.
    
    normals = []
    for i in range(6):
        angle = math.radians(30 + 60 * i)
        normals.append((math.cos(angle), math.sin(angle)))
    
    max_proj = 0.0
    for nx, ny in normals:
        for px, py in pos:
            proj = px * nx + py * ny
            if proj > max_proj:
                max_proj = proj
    
    # max_proj is the required apothem of the outer hexagon.
    R_apo = max_proj
    S = 2 * R_apo / math.sqrt(3)
    
    # 3. Construct the inner hexagons.
    # All inner hexagons are flat-top (angle 0).
    inner = []
    for px, py in pos:
        inner.append([px, py, 0.0])
        
    center = (0.0, 0.0)
    angle = 0.0
    
    # 4. Refinement: 
    # The problem allows rotation. The above assumes all hexagons are aligned.
    # Is it possible to do better by rotating?
    # For hexagons, the densest packing is generally on the lattice.
    # However, the bounding box might be smaller if we rotate the whole cluster or individual hexagons.
    # Given the symmetry of the 3-3-3-2 cluster, rotating the whole cluster by 30 degrees (pointy-top) might yield a different bounding box.
    # Let's test the pointy-top orientation (angle 30 for outer, and rotate inner hexagons by 30).
    
    # Actually, the "width" of the cluster in the flat-top orientation:
    # Max x is 1.5*d + 0.5 (radius of hex) ? No, we used centers.
    # The hexagon extends 0.5 from center in x direction? No, radius is 1.
    # Flat-top hexagon:
    # x-extent: [-1, 1]
    # y-extent: [-sqrt(3)/2, sqrt(3)/2]
    
    # Let's re-evaluate the bounding box more carefully.
    # The projection of a single hexagon onto a normal vector n.
    # The max projection of a hexagon centered at c is c.n + R_apo_inner * (n . n_local_max)?
    # Actually, the support function of a regular hexagon.
    # For a flat-top hexagon (angle 0), the support function h(n) is:
    # If n is at angle 30, 90, etc (normal to side), h(n) = apothem = sqrt(3)/2.
    # If n is at angle 0, 60, etc (normal to vertex), h(n) = radius = 1.
    # Generally h(n) = apothem / cos(delta) where delta is angle between n and closest side normal.
    
    # In my previous calculation, I only projected the CENTERS.
    # I need to project the EXTREMES of the hexagons.
    
    # Let's redo the S calculation.
    # For each of the 6 outer normals (30, 90, 150, 210, 270, 330):
    # Find the max projection of any inner hexagon onto that normal.
    # The max projection of an inner hexagon (flat-top) onto a normal at angle theta_outer:
    # The inner hexagon's support function in direction theta_outer.
    # Let theta_rel = theta_outer mod 60.
    # The support distance from center is:
    # d_support = (sqrt(3)/2) / cos(min(theta_rel, 60-theta_rel))
    # Wait, the normal to the side is at 30. The vertex is at 0.
    # If outer normal is at 30 (aligned with inner side normal), support is sqrt(3)/2.
    # If outer normal is at 0 (aligned with inner vertex), support is 1.
    # Formula: h(theta) = (sqrt(3)/2) / cos(theta) for theta in [-30, 30] relative to side normal?
    # Let's check. At theta=0 (side normal), cos(0)=1, h=sqrt(3)/2. Correct.
    # At theta=30 (vertex normal), cos(30)=sqrt(3)/2, h = (sqrt(3)/2) / (sqrt(3)/2) = 1. Correct.
    
    max_proj_total = 0.0
    for i in range(6):
        angle_outer = math.radians(30 + 60 * i)
        nx, ny = math.cos(angle_outer), math.sin(angle_outer)
        
        for px, py in pos:
            # Projection of center
            c_proj = px * nx + py * ny
            
            # Support of the hexagon in direction (nx, ny)
            # Angle of normal in local frame of inner hex (which is flat-top, angle 0)
            # The inner hex has side normals at 30, 90, 150...
            # We need the angle between (nx, ny) and the closest side normal of the inner hex.
            # Side normals are at 30 + 60k.
            # Let alpha = angle_outer.
            # Find delta = min_k |alpha - (30 + 60k)|
            # Since alpha is 30 + 60i, and inner is aligned, the relative angle is 0.
            # So the support is exactly sqrt(3)/2.
            
            h_inner = math.sqrt(3) / 2
            
            total_proj = c_proj + h_inner
            
            if total_proj > max_proj_total:
                max_proj_total = total_proj
                
    R_apo_outer = max_proj_total
    S = 2 * R_apo_outer / math.sqrt(3)
    
    # Let's check if rotating the inner hexagons by 30 degrees (pointy-top) helps.
    # If inner hexagons are pointy-top (angle 30), their side normals are at 0, 60, 120...
    # If outer is flat-top (normals 30, 90...), the relative angle is 30.
    # Support h_inner would be 1.0 (vertex).
    # This is larger than sqrt(3)/2 (~0.866).
    # So aligning orientations (both flat-top or both pointy-top) is better for the bounding box of a single hex.
    # However, the lattice spacing might change? No, lattice spacing is determined by disjointness.
    # For flat-top hexagons, the center distance to touch is sqrt(3) (horizontal) or 2*1.5=3 (vertical)?
    # Wait.
    # Flat-top hex:
    # Horizontal neighbors: distance sqrt(3).
    # Vertical neighbors: distance 3? No.
    # Let's re-verify the lattice.
    # Flat-top hex at (0,0).
    # Right neighbor: (sqrt(3), 0). Distance sqrt(3).
    # Top-right neighbor: (sqrt(3)/2, 1.5). Distance sqrt(3/4 + 2.25) = sqrt(3) = 1.732.
    # Yes, all neighbors in the triangular lattice are at distance sqrt(3).
    
    # So the positions `pos` are correct for flat-top hexagons.
    # And the support calculation is correct for aligned orientations.
    
    # Let's print S to see what we got.
    # pos x range: -1.5d to 1.5d. d=sqrt(3)~1.732. 1.5*1.732 = 2.598.
    # pos y range: -1.5 to 1.5.
    # Max x projection (normal 0 deg? No, outer normals are 30, 90...)
    # Outer normal at 0 deg? No, flat-top outer has normals at 30, 90, 150, 210, 270, 330.
    # Wait, if outer is flat-top, its vertices are at 0, 60...
    # Its sides are perpendicular to 30, 90...
    # So the constraints are on projections onto 30, 90, 150, 210, 270, 330.
    
    # Let's calculate max_proj for normal 30 deg (0.866, 0.5).
    # Point (1.5d, 1.5) = (2.598, 1.5).
    # Proj = 2.598 * 0.866 + 1.5 * 0.5 = 2.25 + 0.75 = 3.0.
    # Plus h_inner (0.866) = 3.866.
    # R_apo = 3.866.
    # S = 2 * 3.866 / 1.732 = 4.46.
    
    # Is this optimal?
    # The area of 11 unit hexagons is 11 * (3*sqrt(3)/2) = 11 * 2.598 = 28.58.
    # Area of outer hex with side S is (3*sqrt(3)/2) * S^2.
    # 2.598 * S^2 >= 28.58 => S^2 >= 11 => S >= sqrt(11) = 3.316.
    # Our S is 4.46. This is much larger. Why?
    # Because the cluster is not a perfect hexagon. It's a "3-3-3-2" shape.
    # A perfect hexagonal cluster of side length k (in units of hexagons) has 3k(k-1)+1 hexagons.
    # k=1: 1 hex.
    # k=2: 7 hex.
    # k=3: 19 hex.
    # We have 11 hex. We are between k=2 and k=3.
    # The minimal bounding hexagon for 11 hexagons is likely determined by the "corners" of the cluster.
    
    # Can we do better?
    # Maybe rotate the cluster?
    # Or use a different packing?
    # The "flower" packing (1 center, 6 ring, 4 more) is what I used.
    # What if I use a linear chain? No, that's worse.
    # What if I use a 3-4-4 packing?
    # Let's stick with the deterministic construction for now. It's valid and fast.
    # The score is 1.0 for the seed. This will likely be > 1.0 if S is smaller than the seed's S.
    # The seed's S is unknown but likely similar or worse due to random search.
    
    return inner, center, S, angle

# EVOLVE-BLOCK-END

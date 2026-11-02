"""Packing unit regular hexagons inside a regular hexagon via non-convex optimization."""
import numpy as np
from scipy.optimize import minimize
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    
    Returns:
        inner: list of [x, y, angle_degrees] for each inner hexagon
        center: (x, y) of outer hexagon
        side: outer hexagon side length
        angle: outer hexagon rotation in degrees
    """
    # For N=11, we can use a known optimal or near-optimal configuration.
    # The seed places them in a line with side 40.0 which is very loose.
    # 
    # A unit regular hexagon has inradius = sqrt(3)/2 * side = sqrt(3)/2
    # and circumradius = side = 1.
    # 
    # For N=11, a good packing might be a 3-4-3 arrangement or similar.
    # Let's try to find a compact arrangement.
    # 
    # Actually, let's use a more systematic approach:
    # The minimum outer side length for packing N unit hexagons in a hexagon.
    # For small N, known results exist. For N=11, let's estimate.
    # 
    # A hexagonal lattice packing: place hexagons on a triangular grid.
    # The distance between centers of adjacent hexagons in a hexagonal lattice
    # is sqrt(3) (for unit side length hexagons, the center-to-center distance
    # when they share an edge is 2 * inradius = sqrt(3)).
    # 
    # Let's try a 3-4-3 arrangement:
    # Row 0 (top): 3 hexagons
    # Row 1 (middle): 4 hexagons  
    # Row 2 (bottom): 3 hexagons
    # 
    # In a hexagonal lattice, rows are offset. The vertical distance between
    # rows is 1.5 (for unit hexagons, the vertical spacing is 3/2 * side = 1.5
    # when measured center-to-center in the standard orientation).
    # 
    # Actually, for regular hexagons with side length 1:
    # - Width (flat-to-flat) = sqrt(3)
    # - Height (point-to-point) = 2
    # - In a hexagonal tiling, horizontal spacing = sqrt(3), vertical spacing = 1.5
    # 
    # Let's place them in a compact hexagonal cluster shape.
    # For 11 hexagons, one arrangement is a central hexagon surrounded by
    # 6, then another ring of 4... that doesn't work cleanly.
    # 
    # Let's just use a simple grid-like arrangement and estimate the bounding hexagon.
    # 
    # Alternative: use the fact that for a hexagonal arrangement, the outer
    # hexagon side length can be computed from the extent of the inner hexagons.
    # 
    # Let me try a specific known-good configuration for N=11.
    # A 11-hexagon cluster: think of it as a larger hexagon of side 2 (which has
    # 1+3+5+3+1 = 13 positions) minus 2 corner positions, or some other shape.
    # 
    # Actually, let's just compute a reasonable upper bound and return it.
    # The seed returns 40.0 which is way too large. Even a linear arrangement
    # of 11 hexagons would need much less.
    # 
    # Linear arrangement: 11 hexagons in a row. Each hexagon has width sqrt(3)
    # in the horizontal direction (if oriented with flat sides horizontal).
    # Center-to-center distance = sqrt(3). Total span = 10 * sqrt(3) + sqrt(3) = 11*sqrt(3).
    # The outer hexagon needs to contain this. A hexagon with side s has a certain
    # capacity. This is getting complex.
    # 
    # Let's just try a compact arrangement and compute the required outer side.
    
    # Hexagonal lattice positions for 11 hexagons (3-4-3 arrangement):
    # Using axial coordinates or just Cartesian.
    # 
    # Let's use a simple approach: place hexagons in rows.
    # Row y=0: 4 hexagons at x = -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3)
    # Row y=1.5: 3 hexagons at x = -sqrt(3), 0, sqrt(3)
    # Row y=-1.5: 3 hexagons at x = -sqrt(3), 0, sqrt(3)
    # 
    # Wait, that's 10. Let me recount: 4+3+3 = 10. Need 11.
    # 
    # Try 4-4-3 or 3-5-3 or 2-4-2-3 etc.
    # 
    # Let's try:
    # Row 0 (y=0): 3 hexagons
    # Row 1 (y=1.5): 4 hexagons
    # Row 2 (y=-1.5): 4 hexagons
    # Total: 11
    # 
    # Positions:
    # Row y=0: x = -sqrt(3), 0, sqrt(3)
    # Row y=1.5: x = -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3)
    # Row y=-1.5: x = -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3)
    
    sqrt3 = np.sqrt(3)
    
    # Generate positions
    positions = []
    # Row y=0: 3 hexagons
    for x in [-sqrt3, 0.0, sqrt3]:
        positions.append([x, 0.0, 0.0])
    # Row y=1.5: 4 hexagons
    for x in [-1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3]:
        positions.append([x, 1.5, 0.0])
    # Row y=-1.5: 4 hexagons
    for x in [-1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3]:
        positions.append([x, -1.5, 0.0])
    
    inner = positions
    
    # Now compute the minimum outer hexagon side length that contains all these.
    # The outer hexagon is centered at (0,0) with some side length s and rotation 0.
    # 
    # A regular hexagon with side s and rotation 0 (flat top) has vertices at:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # 
    # Actually, for a hexagon centered at origin with side s, the inradius (apothem) is s*sqrt(3)/2.
    # The hexagon can be described as the intersection of 6 half-planes.
    # 
    # For a hexagon with flat top (rotation 0), the constraints are:
    # |x| <= s
    # |x/2 + y*sqrt(3)/2| <= s*sqrt(3)/2  ... wait, let me think more carefully.
    # 
    # A regular hexagon centered at origin with circumradius s (side length s) and
    # one vertex pointing right has vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    # 
    # The hexagon is the set of points (x,y) such that for each of the 6 directions,
    # the projection onto that direction is <= s.
    # 
    # More precisely, for a hexagon with circumradius s, the constraint in direction
    # theta is: x*cos(theta) + y*sin(theta) <= s, for theta = 0, 60, 120, 180, 240, 300.
    # 
    # So for a point (px, py) to be inside, we need:
    # max over theta in {0,60,120,180,240,300} of (px*cos(theta) + py*sin(theta)) <= s
    # 
    # But we also need to account for the fact that the inner hexagons have size.
    # Each inner hexagon has vertices that must be inside the outer hexagon.
    # 
    # For a unit hexagon at position (px, py) with rotation angle alpha, its vertices
    # are at (px + cos(alpha + k*60), py + sin(alpha + k*60)) for k=0..5.
    # 
    # The maximum over all vertices and all 6 directions gives the required s.
    # 
    # Let's compute this for our configuration.
    
    def required_side(inner_hexagons, outer_center=(0.0, 0.0), outer_rotation=0.0):
        """Compute minimum outer side length to contain all inner hexagons."""
        outer_rot_rad = np.radians(outer_rotation)
        max_s = 0.0
        for (px, py, alpha_deg) in inner_hexagons:
            alpha_rad = np.radians(alpha_deg) + outer_rot_rad
            # Vertices of this inner hexagon
            for k in range(6):
                vx = px + np.cos(alpha_rad + k * np.pi / 3)
                vy = py + np.sin(alpha_rad + k * np.pi / 3)
                # Relative to outer center
                rx = vx - outer_center[0]
                ry = vy - outer_center[1]
                # Check against 6 directions of outer hexagon
                for m in range(6):
                    theta = m * np.pi / 3
                    proj = rx * np.cos(theta) + ry * np.sin(theta)
                    if proj > max_s:
                        max_s = proj
        return max_s
    
    side = required_side(inner)
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END

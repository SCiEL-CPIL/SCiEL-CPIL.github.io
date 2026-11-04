import numpy as np
import time

N = 11
DEADLINE = 140.0

def construct():
    """
    Constructs a packing of 11 unit regular hexagons inside a regular hexagon.
    Strategy: Deterministic "flower" packing.
    - 1 central hexagon.
    - 6 hexagons in the first ring (touching center).
    - 4 hexagons in the second ring (touching ring 1 and center? No, touching ring 1).
    All inner hexagons are rotated 30 degrees (flat-top orientation).
    Outer hexagon is also rotated 30 degrees.
    """
    # Epsilon to ensure strict disjointness (interiors do not overlap)
    # Minimum center distance for touching unit hexagons is sqrt(3).
    eps = 1e-9
    
    centers = []
    
    # Ring 0: Center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt(3) + eps
    # Angles: 0, 60, 120, 180, 240, 300
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 4 hexagons at distance 3 + 2*eps
    # These fill the gaps between Ring 1 hexagons.
    # Angles: 30, 90, 150, 270
    # Note: The hexagon at 270 is directly below the center.
    # The hexagons at 30, 150, 270, 90 are placed such that they touch the adjacent Ring 1 hexagons.
    # Distance from Ring 1 (angle 0) to Ring 2 (angle 30):
    # R1: (sqrt(3), 0)
    # R2: (3*cos(30), 3*sin(30)) = (3*sqrt(3)/2, 3/2)
    # dx = 3*sqrt(3)/2 - sqrt(3) = sqrt(3)/2
    # dy = 3/2
    # dist = sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3). Correct.
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # All inner hexagons are rotated 30 degrees
    inner_angle = 30.0
    
    inner = []
    for x, y in centers:
        inner.append([x, y, inner_angle])
        
    # Calculate required outer side length
    # Inner hexagons are flat-top (angle 30).
    # Vertices of a unit hexagon with angle 30 deg:
    # The "radius" (center to vertex) is 1.0.
    # The "apothem" (center to side midpoint) is sqrt(3)/2.
    
    # Max extent in X and Y directions for a single flat-top hexagon:
    # Max X offset: cos(30) = sqrt(3)/2 (vertices at 30, 330 deg)
    # Min X offset: -sqrt(3)/2
    # Max Y offset: sin(90) = 1.0 (vertex at 90 deg)
    # Min Y offset: -1.0
    
    # We need to find the max extent of the entire cluster.
    # Due to symmetry, we just check the max positive X and Y.
    max_x = 0.0
    max_y = 0.0
    
    for x, y in centers:
        # Max x of this hexagon: x + sqrt(3)/2
        max_x = max(max_x, x + np.sqrt(3)/2)
        # Max y of this hexagon: y + 1.0
        max_y = max(max_y, y + 1.0)
        
    # Outer hexagon is also flat-top (angle 30).
    # For a regular hexagon of side S rotated 30 degrees:
    # The width (distance between vertical sides) is sqrt(3) * S.
    # The height (distance between horizontal sides) is 2 * S.
    
    # Constraints for containment:
    # 1. The cluster must fit within the width: 2 * max_x <= sqrt(3) * S
    # 2. The cluster must fit within the height: 2 * max_y <= 2 * S  => max_y <= S
    
    s_from_x = 2 * max_x / np.sqrt(3)
    s_from_y = max_y
    
    side = max(s_from_x, s_from_y)
    
    # Add a tiny margin to side to ensure containment validity
    # The eps in center placement creates a gap of eps between touching hexagons.
    # The max extent calculation already includes the full geometry.
    # The side length calculated is exact for the centers placed.
    # However, due to floating point, adding a tiny margin is safer.
    side += 1e-9
    
    outer_center = (0.0, 0.0)
    outer_angle = 30.0
    
    return inner, outer_center, side, outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

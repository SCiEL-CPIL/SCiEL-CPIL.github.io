import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate 12 hexagons in a hexagonal grid pattern
    # Ring 0: 1 hex
    # Ring 1: 6 hexes
    # Ring 2: 12 hexes (we take 5)
    
    inner = []
    
    # Helper to convert axial coordinates to cartesian
    # Axial (q, r) for hex grid
    # x = 1.5 * q
    # y = sqrt(3) / 2 * q + sqrt(3) * r
    # This assumes pointy-top or flat-top?
    # Standard flat-top:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    import math
    sqrt3 = math.sqrt(3)
    
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return [x, y, 0.0]
        
    # Add Ring 0
    inner.append(axial_to_xy(0, 0))
    
    # Add Ring 1
    ring1_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    for dq, dr in ring1_dirs:
        inner.append(axial_to_xy(dq, dr))
        
    # Add 5 from Ring 2
    # Ring 2 directions from center:
    # (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1) ... wait, need 12.
    # Let's just iterate over a range and pick the first 5
    ring2_candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Check if it's in Ring 2
            # Distance in axial is max(|q|, |r|, |q+r|) ? No.
            # Simple check: generate all neighbors of Ring 1
            pass
            
    # Simpler: Hardcode 5 positions in Ring 2
    # (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1) is not correct logic.
    # Let's use the directions from center for Ring 2:
    # The 12 directions for Ring 2 are:
    # (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)
    # Let's pick 5 of these.
    ring2_pts = [
        (2, 0),
        (2, -1),
        (1, -2),
        (0, -2),
        (-1, -1)
    ]
    
    for q, r in ring2_pts:
        inner.append(axial_to_xy(q, r))
        
    # We have 1 + 6 + 5 = 12 hexagons.
    
    # The bounding box of this configuration (side-3 grid subset)
    # Max distance from center is for (2,0) -> x=3, y=sqrt(3). Dist = sqrt(9+3)=sqrt(12)=3.46?
    # Wait, (2,0) in axial: x=3, y=sqrt(3)*1 = 1.732.
    # Distance = sqrt(9 + 3) = sqrt(12) = 3.464.
    # But the vertex of the hexagon at (2,0) extends further.
    # The hexagon at (2,0) has center (3, 1.732).
    # Its farthest vertex is at distance 3.464 + 1 = 4.464.
    # So we need outer side S >= 4.464.
    # Let's set S = 4.5 to be safe.
    
    outer_center = [0.0, 0.0]
    outer_side = 4.5
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END

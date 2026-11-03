import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    centers = []
    
    # 1 central
    centers.append((0.0, 0.0))
    
    # 6 first ring
    d1 = np.sqrt(3)
    for i in range(6):
        a = np.pi / 3 * i
        centers.append((d1 * np.cos(a), d1 * np.sin(a)))
        
    # 5 second ring
    # Place them attached to the first ring.
    # The first ring hexes are at angles 0, 60, ...
    # The gaps are at 30, 90, ...
    # We can place hexes in the gaps, further out.
    # Distance from center to a hex in the gap:
    # The hex in the gap touches two first-ring hexes.
    # Its center is at distance 2*d1? No.
    # Let's just place them at distance 2*sqrt(3) from center, at angles 30, 90, 150, 210, 270.
    d2 = 2 * np.sqrt(3)
    for i in range(5):
        a = np.pi / 6 + np.pi / 3 * i # 30, 90, 150, 210, 270
        centers.append((d2 * np.cos(a), d2 * np.sin(a)))
        
    # Check overlap? 
    # Distance between first ring (d1) and second ring (d2) at adjacent angles.
    # First ring at 0 deg: (d1, 0).
    # Second ring at 30 deg: (d2*cos30, d2*sin30) = (d2*sqrt(3)/2, d2/2).
    # d2 = 2*d1.
    # x2 = d1*sqrt(3). y2 = d1.
    # x1 = d1. y1 = 0.
    # Dist^2 = (d1*sqrt(3)-d1)^2 + d1^2 = d1^2 (3 - 2sqrt(3) + 1 + 1) = d1^2 (5 - 2sqrt(3)).
    # 5 - 2*1.732 = 1.536.
    # Dist = d1 * sqrt(1.536) = 1.732 * 1.24 = 2.15.
    # Min dist for touch is sqrt(3) = 1.732.
    # So they are disjoint.
    
    # Center the cluster
    x_coords = [c[0] for c in centers]
    y_coords = [c[1] for c in centers]
    cx = (min(x_coords) + max(x_coords)) / 2
    cy = (min(y_coords) + max(y_coords)) / 2
    
    inner = []
    for x, y in centers:
        inner.append([x - cx, y - cy, 0.0])
        
    # Estimate S.
    # Max radius is d2 + 1 (vertex) = 2*sqrt(3) + 1 = 3.464 + 1 = 4.464.
    # Container S must be >= 4.464?
    # If aligned, S >= 4.464.
    # Let's use S = 4.5.
    return inner, (0.0, 0.0), 4.5, 0.0
# EVOLVE-BLOCK-END

import numpy as np

def construct():
    # Use a small epsilon to ensure disjointness
    # 1e-7 is safe for floating point precision while minimizing side length
    eps = 1e-7
    
    # Centers
    centers = []
    # Origin
    centers.append((0.0, 0.0))
    
    # 6 neighbors at dist sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
        
    # 4 more at dist 3 + 2*eps
    # Angles 30, 90, 150, 210
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
        
    # Calculate required outer side length
    # Max X extent comes from ring 2 hexagons at 30 and 150 degrees
    # Total Max X = (2+eps)*sqrt(3)
    # Max Y extent comes from ring 2 hexagon at 90 deg
    # Total Max Y = 4+2eps
    
    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    
    side = max(s_x, s_y)
    
    # No extra margin needed, eps ensures disjointness and geometry ensures containment
    # side += 0.01 # Removed
    
    return inner, (0.0, 0.0), side, 30.0

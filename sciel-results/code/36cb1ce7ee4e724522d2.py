import numpy as np

def construct():
    # Strategy: Use an asymmetric 11-hexagon cluster to reduce the bounding box size.
    # The symmetric 13-hexagon cluster requires S=4.0.
    # An asymmetric 11-hexagon cluster can be shifted to reduce the span in the limiting dimension.
    
    eps = 1e-5
    
    centers = []
    # Ring 0: center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 4 hexagons at distance 3 + 2*eps
    # Angles 30, 90, 150, 210 (Asymmetric selection to allow shifting)
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # Calculate the bounding box of the cluster
    # Inner hexagons are flat-top (angle 30).
    # Max X offset: sqrt(3)/2
    # Max Y offset: 1.0
    
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    
    for x, y in centers:
        min_x = min(min_x, x - np.sqrt(3)/2)
        max_x = max(max_x, x + np.sqrt(3)/2)
        min_y = min(min_y, y - 1.0)
        max_y = max(max_y, y + 1.0)
        
    # Center the cluster to minimize the maximum distance from origin
    # The outer hexagon is centered at (0,0).
    # We can shift the inner cluster by (-center_x, -center_y)
    # to center its bounding box at the origin.
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    # Shifted extents
    width = max_x - min_x
    height = max_y - min_y
    
    # Outer hexagon (pointy-top, angle 30)
    # Width = sqrt(3) * S
    # Height = 2 * S
    
    # We need:
    # sqrt(3) * S >= width
    # 2 * S >= height
    
    s_x = width / np.sqrt(3)
    s_y = height / 2.0
    
    side = max(s_x, s_y)
    
    # Apply the shift to the centers
    inner = []
    for x, y in centers:
        inner.append([x - center_x, y - center_y, 30.0])
        
    # Outer hexagon center is (0,0)
    return inner, (0.0, 0.0), side, 30.0

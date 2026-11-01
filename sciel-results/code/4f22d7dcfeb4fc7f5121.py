import numpy as np

N = 11

def construct():
    # Side length of unit hexagon is 1.
    # Distance between centers in tiling is sqrt(3).
    a = np.sqrt(3)
    
    # Centers of the 7 central hexagons
    centers = [
        (0.0, 0.0),
        (a, 0.0),
        (a/2, 1.5),
        (-a/2, 1.5),
        (-a, 0.0),
        (-a/2, -1.5),
        (a/2, -1.5)
    ]
    
    # Add 4 extensions to make 11
    # Directions: 0, 60, 120, 180
    # This results in max width 4a in x, 3a in others.
    # S = 1 + 4 = 5.
    
    extensions = [
        (2*a, 0.0),       # Ext of (a, 0)
        (a, 3.0),         # Ext of (a/2, 1.5)
        (-a, 3.0),        # Ext of (-a/2, 1.5)
        (-2*a, 0.0)       # Ext of (-a, 0)
    ]
    
    all_centers = centers + extensions
    
    inner = []
    for x, y in all_centers:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), 5.0, 0.0

import numpy as np

def construct():
    # Generate 19-hex grid (radius 2)
    centers = []
    u = np.array([np.sqrt(3), 0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) <= 2:
                # Axial to Cartesian
                x = u[0]*q + v[0]*r
                y = u[1]*q + v[1]*r
                centers.append([x, y])
                
    # Remove 6 corners
    corners = [(2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)]
    centers_to_remove = []
    for c in corners:
        x = u[0]*c[0] + v[0]*c[1]
        y = u[1]*c[0] + v[1]*c[1]
        centers_to_remove.append([x, y])
        
    centers = [c for c in centers if c not in centers_to_remove]
    
    # Remove 2 more edge hexes to get 11
    # (1,1) and (-1,-1)
    extra_remove = [(1,1), (-1,-1)]
    extra_remove_coords = []
    for c in extra_remove:
        x = u[0]*c[0] + v[0]*c[1]
        y = u[1]*c[0] + v[1]*c[1]
        extra_remove_coords.append([x, y])
        
    centers = [c for c in centers if c not in extra_remove_coords]
    
    # Now we have 11 centers
    # Optimize outer hexagon
    # ...

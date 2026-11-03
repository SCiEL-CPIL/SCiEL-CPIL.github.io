import numpy as np
import math

N = 12

def construct():
    # Define centers for 12 hexagons in a hexagonal lattice
    # Using axial coordinates (q, r)
    # Convert to Cartesian: x = sqrt(3) * (q + r/2), y = 1.5 * r
    
    centers_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (-2, 0), (0, -2)
    ]
    
    centers = []
    for q, r in centers_axial:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    centers = np.array(centers)
    
    # Function to calculate minimum enclosing hexagon side for a given rotation phi
    def min_side_for_phi(phi):
        # Normals of the outer hexagon
        # The edges are at angles phi, phi+60, ...
        # The normals are at phi, phi+60, ...
        # Actually, the distance from center to edge is apothem.
        # The support of a point p in direction n is p dot n.
        # We need S * apothem_factor >= max_support + inner_apothem
        # apothem_outer = S * sqrt(3) / 2
        # inner_apothem = 1 * sqrt(3) / 2
        
        max_s = 0.0
        for k in range(6):
            angle = phi + k * math.pi / 3.0
            n = np.array([math.cos(angle), math.sin(angle)])
            # Project all centers onto this normal
            projections = centers @ n
            max_proj = np.max(projections)
            # The condition is:
            # max_proj + inner_apothem <= S * sqrt(3) / 2
            # S >= (max_proj + sqrt(3)/2) * 2 / sqrt(3)
            required_s = (max_proj + math.sqrt(3)/2.0) * 2.0 / math.sqrt(3.0)
            if required_s > max_s:
                max_s = required_s
        return max_s

    # Search for optimal phi
    # The function is periodic with period pi/3 (60 degrees)
    # We can search in [0, pi/3]
    best_s = float('inf')
    best_phi = 0.0
    
    # Coarse search
    for i in range(600):
        phi = i * math.pi / 3.0 / 600.0
        s = min_side_for_phi(phi)
        if s < best_s:
            best_s = s
            best_phi = phi
            
    # Fine search around best_phi
    step = math.pi / 3.0 / 600.0
    for i in range(-100, 100):
        phi = best_phi + i * step / 100.0
        s = min_side_for_phi(phi)
        if s < best_s:
            best_s = s
            best_phi = phi

    # The outer hexagon side length
    L = best_s
    
    # The inner hexagons
    # We assume they are aligned with the outer hexagon for optimal packing
    # So their angle should be the same as the outer hexagon's angle
    # The outer hexagon angle is best_phi
    # The inner hexagons should also be at best_phi
    
    inner = [[x, y, math.degrees(best_phi)] for x, y in centers]
    
    # Outer hexagon center is (0,0)
    # Side length is L
    # Rotation is best_phi
    return inner, (0.0, 0.0), L, best_phi


"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Unit regular hexagons packed on a triangular lattice
    # Lattice vectors for hexagonal close-packing
    v1 = np.array([np.sqrt(3), 0.0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate candidate lattice positions near origin
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            pos = i * v1 + j * v2
            dist = np.linalg.norm(pos)
            candidates.append((dist, i, j, pos))
    
    # Sort by distance from origin and select N closest
    candidates.sort(key=lambda x: x[0])
    selected = candidates[:N]
    centers = [pos for _, _, _, pos in selected]
    
    # Compute centroid of selected centers
    cx = float(np.mean([c[0] for c in centers]))
    cy = float(np.mean([c[1] for c in centers]))
    center = np.array([cx, cy])
    
    # Hexagonal distance: for flat-top hexagon, max of projections onto 3 axes
    def hex_dist(p, c):
        d = p - c
        proj0 = abs(d[0])
        proj60 = abs(d[0] * 0.5 + d[1] * (np.sqrt(3)/2))
        proj120 = abs(d[0] * (-0.5) + d[1] * (np.sqrt(3)/2))
        return max(proj0, proj60, proj120)
    
    # Maximum hexagonal distance from center to any inner hexagon center
    max_hd = max(hex_dist(c, center) for c in centers)
    
    # Outer hexagon inradius must accommodate the farthest inner hexagon
    # Unit hexagon inradius is sqrt(3)/2
    unit_inradius = np.sqrt(3) / 2
    outer_inradius = max_hd + unit_inradius
    outer_side = outer_inradius * 2.0 / np.sqrt(3)
    
    inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
    return inner, (cx, cy), float(outer_side), 0.0
# EVOLVE-BLOCK-END

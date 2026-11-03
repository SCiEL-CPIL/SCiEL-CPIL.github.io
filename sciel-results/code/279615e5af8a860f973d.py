"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct():
    """
    Place 12 unit regular hexagons inside a regular hexagon of minimal side length.
    Uses a hexagonal lattice packing strategy.
    
    Returns:
        inner: list of [x, y, angle_degrees] for each inner hexagon
        center: (cx, cy) of outer hexagon
        side: side length of outer hexagon
        angle: rotation angle of outer hexagon (0.0 for flat-top)
    """
    # For unit hexagons on a hexagonal lattice, the center-to-center distance
    # between adjacent hexagons is sqrt(3) ≈ 1.732.
    # 
    # We use a compact packing of 12 hexagons. A good configuration is
    # a central hexagon surrounded by 6, plus 5 more from the next ring
    # arranged to minimize the bounding box.
    #
    # Axial coordinates (q, r) for the 12 hexagons:
    # Center: (0, 0)
    # Ring 1 (6 hexagons): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (5 hexagons): (2,0), (1,1), (0,2), (-1,2), (-2,1)
    #
    # Convert axial to Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    sqrt3 = np.sqrt(3.0)
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    inner = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
    
    # Compute the required outer side length S such that all hexagons are contained.
    # For a flat-top outer hexagon centered at origin with side S, the distance
    # from center to any vertex is S (the circumradius).
    # 
    # Each inner hexagon has "radius" 1 (distance from center to vertex).
    # For containment, we need: distance from outer center to inner hexagon center + 1 <= S
    # But this is only valid if the inner hexagon is fully inside, which requires
    # checking all vertices of the inner hexagon are inside the outer hexagon.
    #
    # A simpler bound: the maximum distance from origin to any point in the packing.
    # The furthest point from origin will be a vertex of one of the outermost inner hexagons.
    # 
    # For each inner hexagon centered at (cx, cy), its vertices are at:
    # (cx + cos(theta_k), cy + sin(theta_k)) for k=0..5, where theta_k = 60*k degrees
    # (for flat-top orientation, angle 0).
    #
    # Actually, for a regular hexagon with side length 1, the distance from center to
    # vertex is 1. The vertices are at angles 0, 60, 120, 180, 240, 300 degrees.
    
    max_dist = 0.0
    for x, y, _ in inner:
        for k in range(6):
            theta = np.pi / 3.0 * k
            vx = x + np.cos(theta)
            vy = y + np.sin(theta)
            d = np.sqrt(vx**2 + vy**2)
            max_dist = max(max_dist, d)
    
    # The outer hexagon must contain all these points.
    # For a flat-top hexagon with side S centered at origin, a point (px, py) is inside
    # if:
    #   |py| <= sqrt(3)/2 * S
    #   |px| <= S - |py|/sqrt(3)   (when |py| <= sqrt(3)/2 * S)
    #
    # Equivalently, the inradius is sqrt(3)/2 * S, and the condition is:
    #   |px| + |py|/sqrt(3) <= S
    #
    # So we need S >= max over all vertices of (|vx| + |vy|/sqrt(3)).
    
    max_S = 0.0
    for x, y, _ in inner:
        for k in range(6):
            theta = np.pi / 3.0 * k
            vx = x + np.cos(theta)
            vy = y + np.sin(theta)
            val = abs(vx) + abs(vy) / sqrt3
            max_S = max(max_S, val)
    
    # Add a small epsilon to ensure strict containment
    S = max_S + 1e-9
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END

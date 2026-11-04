"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define lattice vectors for a honeycomb packing of unit hexagons
    # Flat-top orientation:
    # v1: horizontal step
    # v2: diagonal step
    v1 = np.array([np.sqrt(3), 0.0])
    v2 = np.array([np.sqrt(3) / 2.0, 1.5])
    
    centers = []
    
    # 1. Central hexagon
    centers.append(np.array([0.0, 0.0]))
    
    # 2-7. First ring (6 neighbors)
    ring1_vectors = [v1, v2, v2 - v1, -v1, -v2, v1 - v2]
    for v in ring1_vectors:
        centers.append(v.copy())
        
    # 8-12. Second ring (5 more hexagons)
    # We select 5 positions from the second ring that are closer to the center (distance 3.0)
    # rather than the ones at distance 2*sqrt(3) ~ 3.46
    # Candidates at distance 3.0:
    # v1 + v2
    # -v1 - v2
    # 2*v1 - v2
    # -2*v1 + v2
    # v1 - 2*v2
    # -v1 + 2*v2
    # 2*v2 - v1
    # -2*v2 + v1
    
    # We pick 5 of these. To maintain some symmetry, we can pick:
    # v1 + v2
    # -v1 - v2
    # 2*v1 - v2
    # -2*v1 + v2
    # v1 - 2*v2
    
    ring2_candidates = [
        v1 + v2,
        -v1 - v2,
        2 * v1 - v2,
        -2 * v1 + v2,
        v1 - 2 * v2
    ]
    
    for v in ring2_candidates:
        centers.append(v.copy())
        
    # Ensure we have exactly N hexagons
    if len(centers) > N:
        centers = centers[:N]
        
    inner = []
    for c in centers:
        # All hexagons have the same orientation (0 degrees) for simplicity
        # This allows them to pack in a honeycomb lattice without rotation
        inner.append([c[0], c[1], 0.0])
        
    # The outer hexagon
    # The max distance from origin to any vertex of the inner hexagons is approx 4.0
    # A regular hexagon of side S has circumradius S.
    # So S = 4.0 should contain them if centered at origin.
    # We can try to optimize S slightly, but 4.0 is a safe upper bound.
    # Let's calculate the exact max distance to be precise.
    
    max_dist = 0.0
    for c in centers:
        # Check vertices of the hexagon at c
        # Vertices are at angles 0, 60, 120, 180, 240, 300
        for k in range(6):
            angle = np.radians(60 * k)
            vx = c[0] + np.cos(angle)
            vy = c[1] + np.sin(angle)
            dist = np.sqrt(vx**2 + vy**2)
            if dist > max_dist:
                max_dist = dist
                
    side = max_dist
    center = (0.0, 0.0)
    angle = 0.0 # Outer hexagon orientation can be 0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END

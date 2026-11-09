import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # We want to pack N unit hexagons into a minimal outer hexagon.
    # Optimal packing for hexagons is on a triangular lattice.
    # The centers of the hexagons are spaced by sqrt(3).
    
    # Generate a grid of potential centers in axial coordinates (q, r)
    # Convert axial to Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    # We need to find a compact set of N centers.
    # Strategy: Generate centers in increasing order of distance from origin.
    
    centers = []
    # Search radius in axial coordinates to ensure we get at least N points
    # A hexagonal number H_k = 3k(k-1) + 1. 
    # For N=11, k=2 gives 7, k=3 gives 19. So radius 3 is enough.
    # To be safe for any N, we can estimate k ~ sqrt(N/3) + 2.
    max_r = int(np.sqrt(N / 3.0)) + 3
    
    candidates = []
    for q in range(-max_r, max_r + 1):
        for r in range(-max_r, max_r + 1):
            # Check if this axial coordinate is valid within a hexagon of radius max_r
            # Distance in axial coords: max(abs(q), abs(r), abs(q+r)) <= max_r
            if max(abs(q), abs(r), abs(q + r)) <= max_r:
                # Convert to Cartesian
                x = np.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                dist = np.sqrt(x**2 + y**2)
                candidates.append((dist, x, y))
    
    # Sort by distance from origin to keep the cluster compact
    candidates.sort(key=lambda item: item[0])
    
    # Take the first N centers
    selected_centers = [ (x, y) for _, x, y in candidates[:N] ]
    
    # Calculate the required outer hexagon size
    # The outer hexagon must contain all inner hexagons.
    # The farthest point of an inner hexagon at center (x,y) from the origin is 
    # distance_to_center + circumradius_of_inner_hex.
    # Circumradius of a unit hexagon is 1.0.
    # The outer hexagon has circumradius R. It must satisfy R >= max_dist + 1.
    # The side length of a regular hexagon is equal to its circumradius.
    
    max_dist = 0.0
    for x, y in selected_centers:
        d = np.sqrt(x**2 + y**2)
        if d > max_dist:
            max_dist = d
            
    side_length = max_dist + 1.0
    
    # Construct the inner list: [x, y, angle]
    # All inner hexagons are aligned with the outer one (angle 0)
    inner = [[x, y, 0.0] for x, y in selected_centers]
    
    # Outer hexagon center is (0,0), side is side_length, angle is 0.
    # The problem asks for (inner, center, side, angle).
    # center is (x, y) of outer hexagon.
    # side is side length.
    # angle is rotation of outer hexagon.
    
    return inner, (0.0, 0.0), side_length, 0.0
# EVOLVE-BLOCK-END

import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Unit hexagon side length
    s = 1.0
    sqrt3 = math.sqrt(3)
    
    # Honeycomb lattice vectors
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 3/2)
    
    centers = []
    
    # Center hexagon
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = math.radians(60 * k)
        x = sqrt3 * math.cos(angle)
        y = 1.5 * math.sin(angle) * 2 # Wait, distance is sqrt(3).
        # Vector to neighbor:
        # k=0: (sqrt(3), 0)
        # k=1: (sqrt(3)/2, 3/2)
        # General: (sqrt(3) cos(60k), 1.5 sin(60k) * 2? No.
        # y = sqrt(3) * sin(60k) * (1/2) * 2?
        # Let's just use the vectors directly.
        pass
        
    # Explicit coordinates for Ring 1
    ring1 = [
        (sqrt3, 0.0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (-sqrt3, 0.0),
        (-sqrt3/2, -1.5),
        (sqrt3/2, -1.5)
    ]
    centers.extend(ring1)
    
    # Ring 2 Midpoints: 5 hexagons
    # Midpoints are at distance 3 from center, at angles 30, 90, 150, 210, 270, 330
    # We pick 5 of them. Let's pick all except the one at 30 degrees (index 0 in 30+60k).
    # Actually, any 5 works for the bounding box if we assume symmetry, but we must ensure they are disjoint.
    # They are on the lattice, so they are disjoint.
    # The bounding box size is determined by the max projection.
    # If we remove one, the max projection in that specific direction might decrease, but the outer hexagon
    # is regular, so its side length is determined by the max over all 6 directions.
    # The other 5 midpoints will still project to the same max value in their respective directions.
    # So S remains 2*sqrt(3) + 1.
    
    ring2_midpoints = []
    for k in range(6):
        angle = math.radians(30 + 60 * k)
        x = 3.0 * math.cos(angle)
        y = 3.0 * math.sin(angle)
        ring2_midpoints.append((x, y))
        
    # Pick 5
    for k in range(5):
        centers.append(ring2_midpoints[k])
        
    # Ensure we have exactly 12
    if len(centers) != 12:
        # Fallback to simple grid if logic fails (should not happen)
        centers = [[4.0 * k, 0.0] for k in range(-(N // 2), N - N // 2)]
        # Convert to (x,y)
        centers = [(c[0], c[1]) for c in centers]
        
    # Calculate S
    # S = 2*sqrt(3) + 1
    S = 2.0 * sqrt3 + 1.0
    
    # Format inner: [[x, y, angle], ...]
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Outer center
    center = (0.0, 0.0)
    
    # Outer angle
    angle = 0.0
    
    return inner, center, S, angle
# EVOLVE-BLOCK-END

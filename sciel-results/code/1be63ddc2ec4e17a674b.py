import itertools
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Axial coordinates for a hex grid of radius 3
    # |q| <= 3, |r| <= 3, |q+r| <= 3
    points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q + r) <= 3:
                points.append((q, r))
                
    # We need to choose 11 points from these 19
    # To ensure we find a compact cluster, we might want to bias towards the center, 
    # but 75k combinations is small enough to just check all.
    
    best_side = float('inf')
    best_config = None
    
    # Precompute unit vectors for projection
    # Hexagon axes are at 0, 60, 120 degrees.
    # Projections onto these axes must be <= apothem = S * sqrt(3) / 2
    # So S >= 2 * max_proj / sqrt(3)
    
    # Cartesian conversion:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # Vertices of a unit hexagon centered at (cx, cy) with rotation 0:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    sqrt3 = math.sqrt(3)
    
    # Directions for projection (unit vectors)
    # Axis 1: (1, 0)
    # Axis 2: (cos(60), sin(60)) = (0.5, sqrt(3)/2)
    # Axis 3: (cos(120), sin(120)) = (-0.5, sqrt(3)/2)
    
    u1 = (1.0, 0.0)
    u2 = (0.5, sqrt3/2.0)
    u3 = (-0.5, sqrt3/2.0)
    
    # Hexagon vertices relative to center
    hex_verts = [
        (1.0, 0.0),
        (0.5, sqrt3/2.0),
        (-0.5, sqrt3/2.0),
        (-1.0, 0.0),
        (-0.5, -sqrt3/2.0),
        (0.5, -sqrt3/2.0)
    ]
    
    count = 0
    for combo in itertools.combinations(points, N):
        if time.time() > deadline:
            break
            
        # Convert axial to cartesian centers
        centers = []
        for q, r in combo:
            cx = sqrt3 * (q + r/2.0)
            cy = 1.5 * r
            centers.append((cx, cy))
            
        # Calculate max projection
        max_proj = 0.0
        
        for cx, cy in centers:
            for vx, vy in hex_verts:
                px = cx + vx
                py = cy + vy
                
                # Project onto 3 axes
                p1 = abs(px * u1[0] + py * u1[1])
                p2 = abs(px * u2[0] + py * u2[1])
                p3 = abs(px * u3[0] + py * u3[1])
                
                local_max = max(p1, p2, p3)
                if local_max > max_proj:
                    max_proj = local_max
        
        # Calculate side length S
        # Apothem A = S * sqrt(3) / 2
        # A >= max_proj
        # S >= 2 * max_proj / sqrt(3)
        S = 2.0 * max_proj / sqrt3
        
        if S < best_side:
            best_side = S
            best_config = centers
            # Optional: break early if we find a perfect packing?
            # Theoretical min for 11? 
            # 7 hexes fit in S=2? 
            # 7-hex cluster: max dist from center is 1 (center) + sqrt(3) (neighbor center) + 1 (vertex)?
            # No, neighbor vertex is at dist sqrt(3)+1 from center?
            # Let's just let the search find the min.
            
        count += 1
        
    # If no config found (should not happen), fallback to simple line
    if best_config is None:
        best_config = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_side = 40.0
    else:
        # Format output: inner rows (x, y, angle_degrees)
        inner = [[cx, cy, 0.0] for cx, cy in best_config]
        center = (0.0, 0.0)
        side = best_side
        angle = 0.0
        return inner, center, side, angle

# EVOLVE-BLOCK-END

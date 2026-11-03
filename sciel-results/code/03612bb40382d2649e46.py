import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Define helper to calculate min side length S for a set of centers
    def calc_S(centers):
        # Centers are (x, y)
        # We need min S such that all centers are inside hexagon of side S-1
        # Effective hexagon side S_eff = S - 1
        # Condition: max hex_dist(centers) <= S_eff
        # hex_dist(p) = max over 6 normals of (p . n) / (sqrt(3)/2) ?
        # Actually, S_eff = max( max(x), max(x/sqrt(3) + y), ... ) * 2/sqrt(3) ?
        
        # Let's use the projection method.
        # Normals at 0, 60, 120, 180, 240, 300
        # For a point p, the required S_eff is max( p_x, p_x/sqrt(3) + p_y, -p_x/sqrt(3) + p_y, -p_x, -p_x/sqrt(3) - p_y, p_x/sqrt(3) - p_y ) * 2/sqrt(3) ?
        # No, the inradius is S_eff * sqrt(3)/2.
        # So S_eff = max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3).
        
        max_proj = 0.0
        for cx, cy in centers:
            # 6 directions
            p1 = cx
            p2 = cx/np.sqrt(3) + cy
            p3 = -cx/np.sqrt(3) + cy
            p4 = -cx
            p5 = -cx/np.sqrt(3) - cy
            p6 = cx/np.sqrt(3) - cy
            local_max = max(abs(p1), abs(p2), abs(p3), abs(p4), abs(p5), abs(p6))
            if local_max > max_proj:
                max_proj = local_max
        
        S_eff = 2.0 * max_proj / np.sqrt(3)
        return S_eff + 1.0

    # Configuration 1: Hexagonal Cluster (1+6+4)
    # Try different sets of 4 corners
    centers1 = [
        (0, 0),
        (np.sqrt(3), 0),
        (np.sqrt(3)/2, 1.5),
        (-np.sqrt(3)/2, 1.5),
        (-np.sqrt(3), 0),
        (-np.sqrt(3)/2, -1.5),
        (np.sqrt(3)/2, -1.5)
    ]
    # Corners at 30, 90, 150, 210
    corners = [
        (3*np.sqrt(3)/2, 1.5),
        (0, 3),
        (-3*np.sqrt(3)/2, 1.5),
        (-3*np.sqrt(3)/2, -1.5)
    ]
    config1 = centers1 + corners
    S1 = calc_S(config1)

    # Configuration 2: 3x4 Parallelogram
    # Rows of 3, 3, 3, 2
    # dx = sqrt(3), dy = 1.5
    centers2 = []
    # Row 0 (y=0): 3
    centers2.append((-np.sqrt(3), 0))
    centers2.append((0, 0))
    centers2.append((np.sqrt(3), 0))
    # Row 1 (y=1.5): 3
    centers2.append((-np.sqrt(3)/2, 1.5))
    centers2.append((np.sqrt(3)/2, 1.5))
    centers2.append((3*np.sqrt(3)/2, 1.5))
    # Row 2 (y=3.0): 3
    centers2.append((-np.sqrt(3), 3.0))
    centers2.append((0, 3.0))
    centers2.append((np.sqrt(3), 3.0))
    # Row 3 (y=4.5): 2
    centers2.append((-np.sqrt(3)/2, 4.5))
    centers2.append((np.sqrt(3)/2, 4.5))
    S2 = calc_S(centers2)

    # Configuration 3: 2x6 Parallelogram
    # Rows of 6, 5?
    centers3 = []
    # Row 0: 6
    for i in range(-3, 3):
        centers3.append((i*np.sqrt(3), 0))
    # Row 1: 5
    for i in range(-2, 3):
        centers3.append((i*np.sqrt(3) + np.sqrt(3)/2, 1.5))
    S3 = calc_S(centers3)

    # Pick best
    best_S = min(S1, S2, S3)
    if best_S == S1:
        centers = config1
    elif best_S == S2:
        centers = centers2
    else:
        centers = centers3

    # Construct inner hexagons
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
    
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END

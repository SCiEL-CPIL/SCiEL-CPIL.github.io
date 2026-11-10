import numpy as np
import time
import itertools

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for n=54 points and m=27 lines.
    Uses a 7x8 grid construction with diagonal lines, removing 2 corners
    to fit the point count, and selecting the 27 best lines.
    """
    start_time = time.time()
    deadline = 600.0 # 600s limit, keep safety margin
    
    # Grid dimensions
    W, H = 7, 8
    
    # All points in the 7x8 grid
    all_points = np.array([(x, y) for x in range(W) for y in range(H)], dtype=np.int64)
    
    # Corners of the grid
    corners = [
        (0, 0),
        (W-1, 0),
        (0, H-1),
        (W-1, H-1)
    ]
    
    # Candidate lines: all diagonals (slope 1) and anti-diagonals (slope -1)
    # Line equation: A*x + B*y + C = 0
    # Slope 1: y = x + c  =>  x - y + c = 0  =>  (1, -1, c)
    # Slope -1: y = -x + c => x + y + c = 0 =>  (1, 1, c)
    
    candidate_lines = []
    
    # Generate slope 1 lines
    # c = y - x
    # Min c: 0 - 6 = -6
    # Max c: 7 - 0 = 7
    for c in range(-6, 8):
        candidate_lines.append((1, -1, c))
        
    # Generate slope -1 lines
    # c = x + y
    # Min c: 0 + 0 = 0
    # Max c: 6 + 7 = 13
    for c in range(0, 14):
        candidate_lines.append((1, 1, c))
        
    candidate_lines = np.array(candidate_lines, dtype=np.int64)
    
    best_incidence_count = -1
    best_points = None
    best_lines = None
    
    # Try all pairs of corners to remove
    for c1, c2 in itertools.combinations(corners, 2):
        # Create point set by removing c1 and c2
        # Mask points that are NOT c1 and NOT c2
        mask = ~((all_points[:, 0] == c1[0] & all_points[:, 1] == c1[1]) | 
                 (all_points[:, 0] == c2[0] & all_points[:, 1] == c2[1]))
        points = all_points[mask]
        
        # Check if we have exactly 54 points
        if len(points) != 54:
            continue
            
        # Calculate incidence count for each candidate line
        # points shape: (54, 2)
        # lines shape: (28, 3)
        # A*x + B*y + C = 0
        # We can compute this for all pairs efficiently
        # But 54*28 is small, so even a loop is fine, but vectorized is better
        
        # Expand points to (54, 1, 2) and lines to (1, 28, 3)
        # Actually, just dot product won't work directly with C.
        # val = A*x + B*y + C
        # pts_x = points[:, 0][:, np.newaxis] # (54, 1)
        # pts_y = points[:, 1][:, np.newaxis] # (54, 1)
        # lines_A = lines[:, 0][np.newaxis, :] # (1, 28)
        # lines_B = lines[:, 1][np.newaxis, :] # (1, 28)
        # lines_C = lines[:, 2][np.newaxis, :] # (1, 28)
        
        pts_x = points[:, 0][:, np.newaxis]
        pts_y = points[:, 1][:, np.newaxis]
        A = lines[:, 0][np.newaxis, :]
        B = lines[:, 1][np.newaxis, :]
        C = lines[:, 2][np.newaxis, :]
        
        # (54, 1) * (1, 28) + (54, 1) * (1, 28) + (1, 28) = (54, 28)
        vals = A * pts_x + B * pts_y + C
        
        # Count zeros
        incidences = np.sum(vals == 0, axis=0) # shape (28,)
        
        # Sort incidences descending and sum top 27
        sorted_incidences = np.sort(incidences)[::-1]
        total_inc = np.sum(sorted_incidences[:27])
        
        if total_inc > best_incidence_count:
            best_incidence_count = total_inc
            best_points = points.copy()
            # Get the indices of the top 27 lines
            top_27_indices = np.argsort(incidences)[::-1][:27]
            best_lines = lines[top_27_indices].copy()
            
            # Check time
            if time.time() > deadline - 5:
                break

    # If no valid configuration found (should not happen), fallback to baseline
    if best_points is None:
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)

    return best_points, best_lines

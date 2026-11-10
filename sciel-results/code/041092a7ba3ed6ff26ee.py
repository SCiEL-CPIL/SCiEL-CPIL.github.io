# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
from itertools import combinations
import time

def normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) for uniqueness."""
    # Handle zero case (should not happen for valid lines from points)
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    
    # Compute GCD
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A, B, C = A // g, B // g, C // g
    
    # Make first non-zero entry positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B == 0 and C < 0:
        A, B, C = -A, -B, -C
        
    return (A, B, C)

def count_incidences(points, lines):
    """Calculate total incidences between points and lines."""
    # points: (N, 2) array
    # lines: (M, 3) array
    # Compute A*x + B*y + C for all pairs
    # Use broadcasting
    # points[:, 0] -> (N, 1)
    # points[:, 1] -> (N, 1)
    # lines[:, 0] -> (1, M)
    # etc.
    px = points[:, 0].astype(np.int64)
    py = points[:, 1].astype(np.int64)
    A = lines[:, 0].astype(np.int64)
    B = lines[:, 1].astype(np.int64)
    C = lines[:, 2].astype(np.int64)
    
    # (N, 1) + (N, 1) + (1, M) -> (N, M)
    # Actually: A[i]*x[j] + B[i]*y[j] + C[i]
    # A is (M,), x is (N,)
    # We want result (N, M) or (M, N)
    # val = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    val = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    
    # Count zeros
    return np.sum(val == 0)

def find_best_configuration(points):
    """Find 27 lines maximizing incidences for given points."""
    n_points = len(points)
    target_lines = 27
    
    # Generate all lines passing through at least 2 points
    lines_dict = {}
    
    # For each pair of points, determine the line
    # Optimization: We can just iterate and add to dict
    for i in range(n_points):
        x1, y1 = points[i]
        for j in range(i + 1, n_points):
            x2, y2 = points[j]
            
            # Line through (x1, y1) and (x2, y2)
            # A = y1 - y2
            # B = x2 - x1
            # C = x1*y2 - x2*y1
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            norm = normalize_line(A, B, C)
            
            if norm not in lines_dict:
                lines_dict[norm] = 0
            # We don't count here yet, we'll count later for efficiency
            # or we can count incrementally? 
            # Incremental counting is hard because a line with k points is generated k(k-1)/2 times.
            # Better to store the set of points or just the line and count later.
            # Since we only have ~1000 lines, counting later is fine.
            
    # Convert to list
    unique_lines = list(lines_dict.keys())
    
    # Count incidences for each unique line
    line_counts = []
    for line in unique_lines:
        A, B, C = line
        count = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                count += 1
        line_counts.append((count, line))
        
    # Sort by count descending
    line_counts.sort(key=lambda x: x[0], reverse=True)
    
    # Greedy selection
    selected_lines = []
    selected_set = set()
    total_inc = 0
    
    for count, line in line_counts:
        if len(selected_lines) == target_lines:
            break
        if line not in selected_set:
            selected_lines.append(line)
            selected_set.add(line)
            total_inc += count
            
    # Convert to array
    lines_arr = np.array(selected_lines, dtype=np.int64)
    
    # Verify shape
    if lines_arr.shape != (27, 3):
        # If we didn't find enough lines (shouldn't happen for 54 points), pad with dummy? 
        # But task requires 27 lines.
        # With 54 points, there are many lines.
        pass
        
    return points, lines_arr, total_inc

def generate_grid_points(kx, ky):
    """Generate a grid of points."""
    xs = np.arange(1, kx + 1)
    ys = np.arange(1, ky + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep margin
    
    best_score = -1
    best_points = None
    best_lines = None
    
    # Candidate point configurations
    # 1. 6x9 Grid
    # 2. 3x18 Grid
    # 3. 2x27 Grid
    # 4. 9x6 Grid (same as 1)
    # 5. Maybe a "diamond" or other shape?
    # Let's stick to grids first as they are structured.
    
    configs = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
        (1, 54),
        (54, 1),
    ]
    
    for kx, ky in configs:
        if time.time() > deadline - 10:
            break
            
        points = generate_grid_points(kx, ky)
        
        # Check for duplicates in points (should be none for grids)
        if len(np.unique(points, axis=0)) != 54:
            continue
            
        # Find best lines
        pts, lns, inc = find_best_configuration(points)
        
        if inc > best_score:
            best_score = inc
            best_points = pts
            best_lines = lns
            
    # If no config found (should not happen), fallback to baseline
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

# EVOLVE-BLOCK-END

import numpy as np
from math import gcd
import time
from itertools import combinations

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _evaluate_configuration(points):
    n = len(points)
    # Generate all unique lines
    lines_set = set()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            lines_set.add(_normalize_line(A, B, C))
    
    # Count incidences for each line
    line_list = list(lines_set)
    if not line_list:
        return 0, [], points
        
    # Convert to numpy for faster dot product if possible, 
    # but loop over lines is fine for ~few hundred lines
    points_arr = np.array(points, dtype=np.int64)
    incidences = []
    for A, B, C in line_list:
        # A*x + B*y + C == 0
        vals = A * points_arr[:, 0] + B * points_arr[:, 1] + C
        count = np.sum(vals == 0)
        incidences.append(count)
        
    # Sort and sum top 27
    incidences.sort(reverse=True)
    top_27 = incidences[:27]
    total_inc = sum(top_27)
    
    # Retrieve the lines corresponding to top 27 counts
    # We need to map counts back to lines
    # Since we sorted a copy, we need to be careful if there are ties.
    # But we just need the set of lines that gives the max sum.
    # Actually, if there are ties, any selection of 27 lines with the highest counts works.
    # To be deterministic and return valid lines, we pick the first 27 from the sorted list of (count, line)
    
    # Re-do to keep track of lines
    line_counts = []
    for i, (A, B, C) in enumerate(line_list):
        line_counts.append((incidences[i], (A, B, C)))
        
    line_counts.sort(key=lambda x: x[0], reverse=True)
    selected_lines = [lc[1] for lc in line_counts[:27]]
    
    return total_inc, selected_lines, points

def szemeredi_trotter_incidences():
    start_time = time.time()
    best_score = 0
    best_points = None
    best_lines = None
    
    # Candidate 1: 7x8 grid (56 points), remove 2
    xs = np.arange(1, 8)
    ys = np.arange(1, 9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    grid_7x8 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Iterate over all pairs to remove
    n_points = len(grid_7x8)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if time.time() - start_time > 550:
                break
            remove_indices = [i, j]
            points = np.delete(grid_7x8, remove_indices, axis=0)
            score, lines, pts = _evaluate_configuration(points)
            if score > best_score:
                best_score = score
                best_points = pts
                best_lines = lines
                
        if time.time() - start_time > 550:
            break
            
    # Candidate 2: 6x9 grid (54 points), no removal
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    grid_6x9 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    score, lines, pts = _evaluate_configuration(grid_6x9)
    if score > best_score:
        best_score = score
        best_points = pts
        best_lines = lines
        
    # Candidate 3: 8x7 grid (56 points), remove 2
    # Symmetric to 7x8, but let's check if grid orientation matters for line selection
    # The set of lines is the same up to rotation/reflection, incidence counts should be identical.
    # So we can skip if 7x8 was fully explored.
    
    # Candidate 4: 5x11 grid (55 points), remove 1
    xs = np.arange(1, 6)
    ys = np.arange(1, 12)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    grid_5x11 = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    for i in range(len(grid_5x11)):
        if time.time() - start_time > 550:
            break
        points = np.delete(grid_5x11, i, axis=0)
        score, lines, pts = _evaluate_configuration(points)
        if score > best_score:
            best_score = score
            best_points = pts
            best_lines = lines
            
    # Fallback if no lines found (should not happen)
    if best_lines is None:
        # Default to simple grid
        xs = np.arange(1, 8)
        ys = np.arange(1, 9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        pts = np.delete(pts, [0, 55], axis=0)
        # Generate lines
        lines_set = set()
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                A = y2 - y1
                B = -(x2 - x1)
                C = (x2 - x1) * y1 - (y2 - y1) * x2
                lines_set.add(_normalize_line(A, B, C))
        line_list = list(lines_set)
        incidences = []
        pts_arr = np.array(pts, dtype=np.int64)
        for A, B, C in line_list:
            vals = A * pts_arr[:, 0] + B * pts_arr[:, 1] + C
            count = np.sum(vals == 0)
            incidences.append(count)
        line_counts = list(zip(incidences, line_list))
        line_counts.sort(key=lambda x: x[0], reverse=True)
        selected_lines = [lc[1] for lc in line_counts[:27]]
        best_points = pts
        best_lines = selected_lines

    lines_arr = np.array(best_lines, dtype=np.int64)
    return best_points, lines_arr

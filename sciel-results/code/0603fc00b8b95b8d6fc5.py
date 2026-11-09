import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients (A,B,C) so that gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count (point, line) incidences exactly using int64."""
    # points: (N,2), lines: (M,3)
    N = points.shape[0]
    M = lines.shape[0]
    count = 0
    for m in range(M):
        A, B, C = lines[m]
        vals = A * points[:, 0] + B * points[:, 1] + C
        count += int(np.sum(vals == 0))
    return count

def _evaluate_configuration(points):
    """
    Given exactly 54 points, compute the best possible set of 27 lines
    that maximizes total incidences. Returns (score, top_lines).
    """
    n = points.shape[0]
    lines_dict = {}
    
    for i in range(n):
        x1, y1 = points[i, 0], points[i, 1]
        for j in range(i + 1, n):
            x2, y2 = points[j, 0], points[j, 1]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            key = _normalize_line(A, B, C)
            if key is not None:
                lines_dict[key] = 0
    
    # Count incidences for each line
    for key in lines_dict:
        A, B, C = key
        vals = A * points[:, 0] + B * points[:, 1] + C
        lines_dict[key] = int(np.sum(vals == 0))
    
    # Select top 27 lines
    sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
    selected = [l for l, c in sorted_lines[:27]]
    score = sum(c for _, c in sorted_lines[:27])
    
    # Pad if necessary (shouldn't happen with 54 points, but be safe)
    if len(selected) < 27:
        all_keys = set(lines_dict.keys())
        sel_keys = set(selected)
        pad = []
        for k in range(1000000, 1000000 + (27 - len(selected))):
            pad.append((1, 0, k))
        selected.extend(pad[:27 - len(selected)])
    
    return score, np.array(selected, dtype=np.int64)

def _generate_grid(a, b):
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _evaluate_removal(base_points, num_remove):
    """Try removing `num_remove` points from base_points to improve top-27 sum."""
    n = base_points.shape[0]
    if n < num_remove:
        return -1, None
    
    # Get all lines and their incidence counts
    lines_dict = {}
    point_lines = [set() for _ in range(n)]
    
    for i in range(n):
        x1, y1 = base_points[i, 0], base_points[i, 1]
        for j in range(i + 1, n):
            x2, y2 = base_points[j, 0], base_points[j, 1]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            key = _normalize_line(A, B, C)
            if key is not None:
                lines_dict[key] = 0
                point_lines[i].add(key)
                point_lines[j].add(key)
    
    for key in lines_dict:
        A, B, C = key
        vals = A * base_points[:, 0] + B * base_points[:, 1] + C
        lines_dict[key] = int(np.sum(vals == 0))
    
    def top_k_sum(counts, k=27):
        if len(counts) < k:
            return sum(counts.values())
        sorted_vals = sorted(counts.values(), reverse=True)
        return sum(sorted_vals[:k])
    
    best_score = -1
    best_mask = None
    
    if num_remove == 0:
        score = top_k_sum(lines_dict)
        return score, base_points
        
    elif num_remove == 1:
        for i in range(n):
            temp_counts = dict(lines_dict)
            for key in point_lines[i]:
                temp_counts[key] -= 1
            score = top_k_sum(temp_counts)
            if score > best_score:
                best_score = score
                best_mask = np.ones(n, dtype=bool)
                best_mask[i] = False
                
    elif num_remove == 2:
        for i in range(n):
            for j in range(i + 1, n):
                temp_counts = dict(lines_dict)
                for key in point_lines[i]:
                    temp_counts[key] -= 1
                for key in point_lines[j]:
                    temp_counts[key] -= 1
                score = top_k_sum(temp_counts)
                if score > best_score:
                    best_score = score
                    best_mask = np.ones(n, dtype=bool)
                    best_mask[i] = False
                    best_mask[j] = False
                    
    if best_mask is None:
        return -1, None
    return best_score, base_points[best_mask]

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, score):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            _, lines = _evaluate_configuration(pts)
            best_lines = lines
            
    # Try various grid configurations
    grid_candidates = [
        (6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2),
        (1, 54), (54, 1), (7, 8), (8, 7), (5, 11), (11, 5),
        (4, 14), (14, 4), (2, 27), (3, 18), (18, 3)
    ]
    
    # Exact grids (54 points)
    for a, b in [(6,9), (9,6), (3,18), (18,3), (2,27), (27,2), (1,54), (54,1)]:
        if time.time() > deadline - 30:
            break
        pts = _generate_grid(a, b)
        score, _ = _evaluate_configuration(pts)
        update_best(pts, score)
        
    # Grids with removals
    removal_candidates = [(7,8,2), (8,7,2), (5,11,1), (11,5,1), (4,14,2), (14,4,2)]
    for a, b, rem in removal_candidates:
        if time.time() > deadline - 30:
            break
        pts_base = _generate_grid(a, b)
        score, pts_best = _evaluate_removal(pts_base, rem)
        if pts_best is not None and len(pts_best) == 54:
            real_score, _ = _evaluate_configuration(pts_best)
            update_best(pts_best, real_score)
            
    # Grids with additions
    def try_addition(base_a, base_b, add_points):
        if time.time() > deadline - 30:
            return
        base_pts = _generate_grid(base_a, base_b)
        add_arr = np.array(add_points, dtype=np.int64)
        pts = np.vstack([base_pts, add_arr])
        if len(pts) != 54:
            return
        # Check duplicates
        if len(set(map(tuple, pts))) != 54:
            return
        score, _ = _evaluate_configuration(pts)
        update_best(pts, score)
        
    # 5x10 + 4 points
    try_addition(5, 10, [(1,11),(2,11),(3,11),(4,11)])
    try_addition(5, 10, [(6,1),(6,2),(6,3),(6,4)])
    try_addition(10, 5, [(11,1),(11,2),(11,3),(11,4)])
    try_addition(10, 5, [(1,6),(2,6),(3,6),(4,6)])
    
    # 4x12 + 6 points
    try_addition(4, 12, [(1,13),(2,13),(3,13),(4,13),(5,13),(6,13)])
    try_addition(4, 12, [(5,1),(5,2),(5,3),(5,4),(5,5),(5,6)])
    
    # 6x8 + 6 points
    try_addition(6, 8, [(1,9),(2,9),(3,9),(4,9),(5,9),(6,9)])
    try_addition(6, 8, [(7,1),(7,2),(7,3),(7,4),(7,5),(7,6)])
    
    # 7x7 + 5 points
    try_addition(7, 7, [(1,8),(2,8),(3,8),(4,8),(5,8)])
    try_addition(7, 7, [(8,1),(8,2),(8,3),(8,4),(8,5)])
    
    # 8x6 + 6 points
    try_addition(8, 6, [(1,7),(2,7),(3,7),(4,7),(5,7),(6,7)])
    try_addition(8, 6, [(7,1),(7,2),(7,3),(7,4),(7,5),(7,6)])
    
    # 9x5 + 9 points
    try_addition(9, 5, [(1,6),(2,6),(3,6),(4,6),(5,6),(6,6),(7,6),(8,6),(9,6)])
    try_addition(9, 5, [(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(6,8),(6,9)])
    
    # Fallback
    if best_points is None:
        pts = _generate_grid(6, 9)
        score, lines = _evaluate_configuration(pts)
        best_points = pts
        best_lines = lines
        best_score = score
        
    return best_points, best_lines

import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidence(points, lines):
    """Count incidences efficiently using vectorized operations."""
    # points: (n, 2), lines: (m, 3)
    # Compute A*x + B*y + C for all pairs
    Ax = lines[:, 0:1] * points[:, 0:1]  # (m, n)
    By = lines[:, 1:2] * points[:, 1:2]  # (m, n)
    C = lines[:, 2:3]
    vals = Ax + By + C
    return np.sum(vals == 0)

def _find_best_lines_for_points(points, n_lines):
    """Find the best n_lines lines that maximize incidences with given points."""
    n_points = len(points)
    
    # Generate candidate lines from all pairs of points
    candidate_lines = set()
    
    # Add horizontal, vertical, and diagonal lines
    for i in range(n_points):
        for j in range(i+1, n_points):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(int(A), int(B), int(C))
            candidate_lines.add(norm)
    
    # Also add lines through single points with various slopes
    for i in range(n_points):
        x, y = points[i]
        # Horizontal
        candidate_lines.add(_normalize_line(0, 1, -y))
        # Vertical
        candidate_lines.add(_normalize_line(1, 0, -x))
        # Diagonal
        candidate_lines.add(_normalize_line(1, -1, y - x))
        candidate_lines.add(_normalize_line(1, 1, -(x + y)))
    
    candidate_lines = list(candidate_lines)
    
    # Score each candidate line by number of incidences
    line_scores = []
    for A, B, C in candidate_lines:
        vals = A * points[:, 0] + B * points[:, 1] + C
        score = np.sum(vals == 0)
        if score >= 2:  # Only consider lines with at least 2 incidences
            line_scores.append((score, A, B, C))
    
    # Sort by score descending
    line_scores.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    
    # Greedily select lines
    selected = []
    selected_norms = set()
    
    for score, A, B, C in line_scores:
        norm = (A, B, C)
        if norm in selected_norms:
            continue
        selected.append((A, B, C))
        selected_norms.add(norm)
        if len(selected) >= n_lines:
            break
    
    # If we don't have enough, add lines with score 1
    if len(selected) < n_lines:
        for A, B, C in candidate_lines:
            norm = (A, B, C)
            if norm in selected_norms:
                continue
            selected.append((A, B, C))
            selected_norms.add(norm)
            if len(selected) >= n_lines:
                break
    
    # If still not enough, generate arbitrary lines
    while len(selected) < n_lines:
        A = random.randint(-5, 5)
        B = random.randint(-5, 5)
        C = random.randint(-20, 20)
        if A == 0 and B == 0:
            continue
        norm = _normalize_line(A, B, C)
        if norm not in selected_norms:
            selected.append(norm)
            selected_norms.add(norm)
    
    return np.array(selected[:n_lines], dtype=np.int64)

def _local_search_points(points, lines, M, I, n_points, n_lines, T, deadline, max_iters=5000):
    """Local search on point positions."""
    point_set = set(map(tuple, points.tolist()))
    best_I = I
    best_points = points.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        i = random.randint(0, n_points - 1)
        old_px, old_py = int(points[i, 0]), int(points[i, 1])
        
        # Try several candidate positions
        candidates = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                new_px, new_py = old_px + dx, old_py + dy
                if (new_px, new_py) not in point_set:
                    candidates.append((new_px, new_py))
        
        if not candidates:
            continue
        
        # Evaluate a subset of candidates
        random.shuffle(candidates)
        best_delta = -1000
        best_pos = None
        
        for new_px, new_py in candidates[:15]:
            new_M_i = (lines[:, 0] * new_px + lines[:, 1] * new_py + lines[:, 2] == 0)
            delta = np.sum(new_M_i) - np.sum(M[i])
            if delta > best_delta:
                best_delta = delta
                best_pos = (new_px, new_py)
        
        if best_pos is None:
            continue
        
        new_px, new_py = best_pos
        new_M_i = (lines[:, 0] * new_px + lines[:, 1] * new_py + lines[:, 2] == 0)
        delta_I = np.sum(new_M_i) - np.sum(M[i])
        
        if delta_I > 0 or random.random() < np.exp(delta_I / max(T, 1e-10)):
            M[i] = new_M_i
            point_set.discard((old_px, old_py))
            points[i, 0] = new_px
            points[i, 1] = new_py
            point_set.add((new_px, new_py))
            I += delta_I
            
            if I > best_I:
                best_I = I
                best_points = points.copy()
        
        T *= 0.99995
    
    return best_points, best_I

def _local_search_lines(points, lines, M, I, n_points, n_lines, T, deadline, max_iters=5000):
    """Local search on line coefficients."""
    norm_lines = [_normalize_line(int(lines[j, 0]), int(lines[j, 1]), int(lines[j, 2])) for j in range(n_lines)]
    best_I = I
    best_lines = lines.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        j = random.randint(0, n_lines - 1)
        
        # Try retargeting to random pairs of points
        best_delta = -1000
        best_line = None
        best_norm = None
        
        for _ in range(30):
            a = random.randint(0, n_points - 1)
            b = random.randint(0, n_points - 1)
            while b == a:
                b = random.randint(0, n_points - 1)
            
            x_a, y_a = int(points[a, 0]), int(points[a, 1])
            x_b, y_b = int(points[b, 0]), int(points[b, 1])
            A = y_b - y_a
            B = x_a - x_b
            C = -(A * x_a + B * y_a)
            
            if A == 0 and B == 0:
                continue
            
            norm = _normalize_line(A, B, C)
            
            # Check if this line is already used by another line index
            is_dup = False
            for l_idx in range(n_lines):
                if l_idx != j and norm_lines[l_idx] == norm:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            new_M_j = (A * points[:, 0] + B * points[:, 1] + C == 0)
            delta = np.sum(new_M_j) - np.sum(M[:, j])
            if delta > best_delta:
                best_delta = delta
                best_line = [A, B, C]
                best_norm = norm
        
        if best_line is None:
            continue
        
        A_new, B_new, C_new = best_line
        new_M_j = (A_new * points[:, 0] + B_new * points[:, 1] + C_new == 0)
        delta_I = np.sum(new_M_j) - np.sum(M[:, j])
        
        if delta_I > 0 or random.random() < np.exp(delta_I / max(T, 1e-10)):
            M[:, j] = new_M_j
            lines[j, 0] = A_new
            lines[j, 1] = B_new
            lines[j, 2] = C_new
            norm_lines[j] = best_norm
            I += delta_I
            
            if I > best_I:
                best_I = I
                best_lines = lines.copy()
        
        T *= 0.99995
    
    return best_lines, best_I

def _polish(points, lines, M, I, n_points, n_lines, deadline):
    """Final polish with aggressive local search."""
    best_points = points.copy()
    best_lines = lines.copy()
    best_I = I
    
    # Multiple passes of alternating point and line optimization
    for pass_num in range(3):
        if time.time() > deadline - 30:
            break
        
        # Point optimization
        T = 0.5
        points, I = _local_search_points(points, lines, M, I, n_points, n_lines, T, deadline - 30, max_iters=3000)
        
        if I > best_I:
            best_I = I
            best_points = points.copy()
        
        # Recompute M
        M = (lines[:, 0][np.newaxis, :] * points[:, 0][:, np.newaxis] + 
             lines[:, 1][np.newaxis, :] * points[:, 1][:, np.newaxis] + 
             lines[:, 2][np.newaxis, :] == 0)
        
        # Line optimization
        T = 0.5
        lines, I = _local_search_lines(points, lines, M, I, n_points, n_lines, T, deadline - 30, max_iters=3000)
        
        if I > best_I:
            best_I = I
            best_lines = lines.copy()
        
        # Recompute M
        M = (lines[:, 0][np.newaxis, :] * points[:, 0][:, np.newaxis] + 
             lines[:, 1][np.newaxis, :] * points[:, 1][:, np.newaxis] + 
             lines[:, 2][np.newaxis, :] == 0)
    
    return best_points, best_lines, best_I

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580.0
    
    n_points = 54
    n_lines = 27
    
    best_overall_I = 0
    best_overall_points = None
    best_overall_lines = None
    
    # Try multiple initial configurations
    configs = [
        # 6x9 grid
        (6, 9),
        # 9x6 grid
        (9, 6),
        # 7x8 grid (56 points, use 54)
        (7, 8),
        # 8x7 grid (56 points, use 54)
        (8, 7),
        # 5x11 grid (55 points, use 54)
        (5, 11),
        # 11x5 grid (55 points, use 54)
        (11, 5),
    ]
    
    for k, m in configs:
        if time.time() > deadline - 60:
            break
        
        # Generate grid points
        xs = np.arange(1, k+1)
        ys = np.arange(1, m+1)
        px, py = np.meshgrid(xs, ys, indexing='ij')
        all_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # If we have more than n_points, randomly select n_points
        if len(all_points) > n_points:
            indices = random.sample(range(len(all_points)), n_points)
            points = all_points[indices]
        elif len(all_points) < n_points:
            # Pad with nearby points
            points = all_points.copy()
            while len(points) < n_points:
                base = random.choice(all_points)
                new_pt = base + [random.randint(-1, 1), random.randint(-1, 1)]
                if tuple(new_pt) not in set(map(tuple, points.tolist())):
                    points = np.vstack([points, new_pt.reshape(1, 2)])
        else:
            points = all_points.copy()
        
        # Find best lines for these points
        lines = _find_best_lines_for_points(points, n_lines)
        
        # Compute initial incidence
        M = (lines[:, 0][np.newaxis, :] * points[:, 0][:, np.newaxis] + 
             lines[:, 1][np.newaxis, :] * points[:, 1][:, np.newaxis] + 
             lines[:, 2][np.newaxis, :] == 0)
        I = np.sum(M)
        
        # Local search
        T = 1.5
        points, I = _local_search_points(points, lines, M, I, n_points, n_lines, T, deadline - 90, max_iters=2000)
        
        # Recompute M
        M = (lines[:, 0][np.newaxis, :] * points[:, 0][:, np.newaxis] + 
             lines[:, 1][np.newaxis, :] * points[:, 1][:, np.newaxis] + 
             lines[:, 2][np.newaxis, :] == 0)
        
        T = 1.5
        lines, I = _local_search_lines(points, lines, M, I, n_points, n_lines, T, deadline - 90, max_iters=2000)
        
        # Recompute M
        M = (lines[:, 0][np.newaxis, :] * points[:, 0][:, np.newaxis] + 
             lines[:, 1][np.newaxis, :] * points[:, 1][:, np.newaxis] + 
             lines[:, 2][np.newaxis, :] == 0)
        
        # Polish
        if time.time() < deadline - 30:
            points, lines, I = _polish(points, lines, M, I, n_points, n_lines, deadline - 30)
        
        if I > best_overall_I:
            best_overall_I = I
            best_overall_points = points.copy()
            best_overall_lines = lines.copy()
    
    # Final polish on best configuration
    if best_overall_points is not None and time.time() < deadline - 10:
        M = (best_overall_lines[:, 0][np.newaxis, :] * best_overall_points[:, 0][:, np.newaxis] + 
             best_overall_lines[:, 1][np.newaxis, :] * best_overall_points[:, 1][:, np.newaxis] + 
             best_overall_lines[:, 2][np.newaxis, :] == 0)
        I = np.sum(M)
        best_overall_points, best_overall_lines, best_overall_I = _polish(
            best_overall_points, best_overall_lines, M, I, n_points, n_lines, deadline - 5)
    
    # Validate and return
    points = best_overall_points.astype(np.int64)
    lines = best_overall_lines.astype(np.int64)
    
    # Ensure no duplicate points
    point_tuples = list(map(tuple, points.tolist()))
    if len(set(point_tuples)) < n_points:
        # Remove duplicates
        unique_points = []
        seen = set()
        for pt in point_tuples:
            if pt not in seen:
                unique_points.append(pt)
                seen.add(pt)
        while len(unique_points) < n_points:
            # Add a new point
            base = random.choice(unique_points)
            new_pt = (base[0] + random.randint(-2, 2), base[1] + random.randint(-2, 2))
            if new_pt not in seen:
                unique_points.append(new_pt)
                seen.add(new_pt)
        points = np.array(unique_points, dtype=np.int64)
    
    # Ensure no duplicate lines
    norm_lines = [_normalize_line(int(lines[j, 0]), int(lines[j, 1]), int(lines[j, 2])) for j in range(n_lines)]
    unique_lines = []
    seen_norms = set()
    for j in range(n_lines):
        if norm_lines[j] not in seen_norms:
            unique_lines.append(norm_lines[j])
            seen_norms.add(norm_lines[j])
    while len(unique_lines) < n_lines:
        # Add a new line
        A = random.randint(-5, 5)
        B = random.randint(-5, 5)
        C = random.randint(-20, 20)
        if A == 0 and B == 0:
            continue
        norm = _normalize_line(A, B, C)
        if norm not in seen_norms:
            unique_lines.append(norm)
            seen_norms.add(norm)
    lines = np.array(unique_lines[:n_lines], dtype=np.int64)
    
    return points, lines

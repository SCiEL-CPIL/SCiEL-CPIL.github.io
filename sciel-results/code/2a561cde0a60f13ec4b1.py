import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line (A, B, C) so gcd(|A|,|B|,|C|) = 1 and first nonzero is positive."""
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _incidence_count(points, lines):
    """Count (p,l) pairs where A*x + B*y + C == 0."""
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = points[:, 0:1].T
    y = points[:, 1:2].T
    val = A * x + B * y + C
    return int(np.count_nonzero(val == 0))

def _compute_incidence_matrix(points, lines):
    """Compute boolean incidence matrix (27, 54)."""
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = points[:, 0:1].T
    y = points[:, 1:2].T
    val = A * x + B * y + C
    return (val == 0)

def _build_grid_points(cols, rows):
    """Build a grid of points."""
    xs = np.arange(cols)
    ys = np.arange(rows)
    PX, PY = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)

def _best_lines_for_points(points, k=27):
    """Find the k lines with highest incidence counts through the given points."""
    n_pts = len(points)
    line_counts = {}
    
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            if A == 0 and B == 0:
                continue
                
            norm = _normalize_line(A, B, C)
            if norm not in line_counts:
                line_counts[norm] = 0
    
    if not line_counts:
        return np.zeros((k, 3), dtype=np.int64)
    
    lines_list = list(line_counts.keys())
    counts = []
    for A, B, C in lines_list:
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.count_nonzero(val == 0))
        counts.append(cnt)
        
    counts = np.array(counts)
    top_idx = np.argsort(-counts)[:k]
    selected_lines = [lines_list[i] for i in top_idx]
    
    # Pad if needed
    while len(selected_lines) < k:
        selected_lines.append((1, 0, 0))
        
    return np.array(selected_lines, dtype=np.int64)

def _is_valid(points, lines):
    """Quick validity check."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    if len(set(map(tuple, points.tolist()))) != 54:
        return False
        
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        A, B, C = int(A), int(B), int(C)
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm in line_set:
            return False
        line_set.add(norm)
    return True

def _simulated_annealing(points, lines, deadline, seed=42):
    """
    Simulated annealing to escape local optima.
    """
    points = points.copy()
    lines = lines.copy()
    
    M = _compute_incidence_matrix(points, lines)
    current_score = int(np.sum(M))
    best_score = current_score
    best_points = points.copy()
    best_lines = lines.copy()
    
    point_set = set(map(tuple, points.tolist()))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        norm = _normalize_line(int(A), int(B), int(C))
        line_set.add(norm)
    
    rng = np.random.default_rng(seed)
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        
        if iters % 5000 == 0:
            current_score = _incidence_count(points, lines)
            M = _compute_incidence_matrix(points, lines)
            point_set = set(map(tuple, points.tolist()))
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                best_lines = lines.copy()
            
            if T < T_min:
                T = 1.0  # Restart with higher temperature
                # Random perturbation
                for _ in range(10):
                    p_idx = rng.integers(0, 54)
                    dx = rng.integers(-10, 11)
                    dy = rng.integers(-10, 11)
                    nx, ny = points[p_idx][0] + dx, points[p_idx][1] + dy
                    if (nx, ny) not in point_set or (nx, ny) == tuple(points[p_idx]):
                        point_set.discard(tuple(points[p_idx]))
                        points[p_idx] = [nx, ny]
                        point_set.add((nx, ny))
                M = _compute_incidence_matrix(points, lines)
                current_score = int(np.sum(M))
                continue
        
        T *= cooling_rate
        
        move_type = rng.random()
        
        if move_type < 0.6:
            # Move a single point
            p_idx = rng.integers(0, 54)
            orig_x, orig_y = points[p_idx]
            dx = rng.integers(-8, 9)
            dy = rng.integers(-8, 9)
            if dx == 0 and dy == 0:
                continue
            new_x, new_y = orig_x + dx, orig_y + dy
            
            if (new_x, new_y) in point_set and (new_x, new_y) != (orig_x, orig_y):
                continue
            
            old_inc = int(np.sum(M[:, p_idx]))
            new_inc = 0
            for l_idx in range(27):
                A, B, C = lines[l_idx]
                if A * new_x + B * new_y + C == 0:
                    new_inc += 1
            
            delta = new_inc - old_inc
            if delta >= 0 or rng.random() < np.exp(delta / T):
                point_set.discard((orig_x, orig_y))
                points[p_idx] = [new_x, new_y]
                point_set.add((new_x, new_y))
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    M[l_idx, p_idx] = (A * new_x + B * new_y + C == 0)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    
        elif move_type < 0.85:
            # Replace a line
            l_idx = rng.integers(0, 27)
            old_inc = int(np.sum(M[l_idx, :]))
            
            # Pick a random pair of points to define a new line
            p1_idx = rng.integers(0, 54)
            p2_idx = rng.integers(0, 54)
            if p1_idx == p2_idx:
                continue
                
            x1, y1 = points[p1_idx]
            x2, y2 = points[p2_idx]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            if A == 0 and B == 0:
                continue
                
            norm = _normalize_line(A, B, C)
            if norm in line_set:
                continue
            
            new_inc = 0
            for p_idx in range(54):
                x, y = points[p_idx]
                if norm[0] * x + norm[1] * y + norm[2] == 0:
                    new_inc += 1
            
            delta = new_inc - old_inc
            if delta >= 0 or rng.random() < np.exp(delta / T):
                line_set.discard(_normalize_line(int(lines[l_idx][0]), int(lines[l_idx][1]), int(lines[l_idx][2])))
                lines[l_idx] = norm
                line_set.add(norm)
                for p_idx in range(54):
                    x, y = points[p_idx]
                    M[l_idx, p_idx] = (norm[0] * x + norm[1] * y + norm[2] == 0)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    
        else:
            # Move two points
            p1_idx = rng.integers(0, 54)
            p2_idx = rng.integers(0, 54)
            if p1_idx == p2_idx:
                continue
            
            orig_x1, orig_y1 = points[p1_idx]
            orig_x2, orig_y2 = points[p2_idx]
            
            dx1 = rng.integers(-6, 7)
            dy1 = rng.integers(-6, 7)
            dx2 = rng.integers(-6, 7)
            dy2 = rng.integers(-6, 7)
            if dx1 == 0 and dy1 == 0 and dx2 == 0 and dy2 == 0:
                continue
            
            nx1, ny1 = orig_x1 + dx1, orig_y1 + dy1
            nx2, ny2 = orig_x2 + dx2, orig_y2 + dy2
            
            if (nx1, ny1) in point_set and (nx1, ny1) != (orig_x1, orig_y1):
                continue
            if (nx2, ny2) in point_set and (nx2, ny2) != (orig_x2, orig_y2):
                continue
            if (nx1, ny1) == (nx2, ny2):
                continue
            
            old_inc1 = int(np.sum(M[:, p1_idx]))
            old_inc2 = int(np.sum(M[:, p2_idx]))
            
            new_inc1 = 0
            new_inc2 = 0
            for l_idx in range(27):
                A, B, C = lines[l_idx]
                if A * nx1 + B * ny1 + C == 0:
                    new_inc1 += 1
                if A * nx2 + B * ny2 + C == 0:
                    new_inc2 += 1
            
            delta = (new_inc1 + new_inc2) - (old_inc1 + old_inc2)
            if delta >= 0 or rng.random() < np.exp(delta / T):
                point_set.discard((orig_x1, orig_y1))
                point_set.discard((orig_x2, orig_y2))
                points[p1_idx] = [nx1, ny1]
                points[p2_idx] = [nx2, ny2]
                point_set.add((nx1, ny1))
                point_set.add((nx2, ny2))
                
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    M[l_idx, p1_idx] = (A * nx1 + B * ny1 + C == 0)
                    M[l_idx, p2_idx] = (A * nx2 + B * ny2 + C == 0)
                    
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()

    return best_points, best_lines, best_score

def _local_search(points, lines, deadline, seed=42):
    """
    Greedy local search with multiple move types.
    """
    points = points.copy()
    lines = lines.copy()
    
    M = _compute_incidence_matrix(points, lines)
    current_score = int(np.sum(M))
    best_score = current_score
    best_points = points.copy()
    best_lines = lines.copy()
    
    point_set = set(map(tuple, points.tolist()))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        norm = _normalize_line(int(A), int(B), int(C))
        line_set.add(norm)
    
    rng = np.random.default_rng(seed)
    
    iters = 0
    no_improve = 0
    while time.time() < deadline:
        iters += 1
        no_improve += 1
        
        if iters % 3000 == 0:
            current_score = _incidence_count(points, lines)
            M = _compute_incidence_matrix(points, lines)
            point_set = set(map(tuple, points.tolist()))
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                best_lines = lines.copy()
                no_improve = 0
        
        if no_improve > 5000:
            # Perturb to escape local optimum
            for _ in range(5):
                p_idx = rng.integers(0, 54)
                dx = rng.integers(-3, 4)
                dy = rng.integers(-3, 4)
                nx, ny = points[p_idx][0] + dx, points[p_idx][1] + dy
                if (nx, ny) not in point_set or (nx, ny) == tuple(points[p_idx]):
                    point_set.discard(tuple(points[p_idx]))
                    points[p_idx] = [nx, ny]
                    point_set.add((nx, ny))
            M = _compute_incidence_matrix(points, lines)
            current_score = int(np.sum(M))
            no_improve = 0
            continue
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Move a single point to maximize incidences
            p_idx = rng.integers(0, 54)
            orig_x, orig_y = points[p_idx]
            old_inc = int(np.sum(M[:, p_idx]))
            
            best_delta = 0
            best_pos = None
            
            # Try candidates: nearby points and points on existing lines
            candidates = set()
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    if dx == 0 and dy == 0:
                        continue
                    candidates.add((orig_x + dx, orig_y + dy))
            
            # Also try points on each line
            for l_idx in range(27):
                A, B, C = lines[l_idx]
                if B != 0:
                    for dx in range(-6, 7):
                        nx = orig_x + dx
                        num = -(A * nx + C)
                        if num % B == 0:
                            ny = num // B
                            if -200 <= nx <= 200 and -200 <= ny <= 200:
                                candidates.add((nx, ny))
                elif A != 0:
                    for dy in range(-6, 7):
                        ny = orig_y + dy
                        num = -(B * ny + C)
                        if num % A == 0:
                            nx = num // A
                            if -200 <= nx <= 200 and -200 <= ny <= 200:
                                candidates.add((nx, ny))
            
            for new_x, new_y in candidates:
                if (new_x, new_y) in point_set and (new_x, new_y) != (orig_x, orig_y):
                    continue
                
                new_inc = 0
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    if A * new_x + B * new_y + C == 0:
                        new_inc += 1
                        
                delta = new_inc - old_inc
                if delta > best_delta:
                    best_delta = delta
                    best_pos = (new_x, new_y)
            
            if best_pos is not None and best_delta > 0:
                point_set.discard((orig_x, orig_y))
                points[p_idx] = best_pos
                point_set.add(best_pos)
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    M[l_idx, p_idx] = (A * best_pos[0] + B * best_pos[1] + C == 0)
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    no_improve = 0
                    
        elif move_type < 0.8:
            # Replace a line with the best available line
            l_idx = rng.integers(0, 27)
            old_inc = int(np.sum(M[l_idx, :]))
            
            best_new_inc = old_inc
            best_norm = None
            
            # Sample pairs
            for _ in range(300):
                p1_idx = rng.integers(0, 54)
                p2_idx = rng.integers(0, 54)
                if p1_idx == p2_idx:
                    continue
                    
                x1, y1 = points[p1_idx]
                x2, y2 = points[p2_idx]
                
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                
                if A == 0 and B == 0:
                    continue
                    
                norm = _normalize_line(A, B, C)
                if norm in line_set:
                    continue
                
                new_inc = 0
                for p_idx in range(54):
                    x, y = points[p_idx]
                    if norm[0] * x + norm[1] * y + norm[2] == 0:
                        new_inc += 1
                        
                if new_inc > best_new_inc:
                    best_new_inc = new_inc
                    best_norm = norm
            
            if best_norm is not None and best_new_inc > old_inc:
                line_set.discard(_normalize_line(int(lines[l_idx][0]), int(lines[l_idx][1]), int(lines[l_idx][2])))
                lines[l_idx] = best_norm
                line_set.add(best_norm)
                for p_idx in range(54):
                    x, y = points[p_idx]
                    M[l_idx, p_idx] = (best_norm[0] * x + best_norm[1] + best_norm[2] == 0)
                current_score += best_new_inc - old_inc
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    no_improve = 0
                    
        else:
            # Move two points
            p1_idx = rng.integers(0, 54)
            p2_idx = rng.integers(0, 54)
            if p1_idx == p2_idx:
                continue
            
            orig_x1, orig_y1 = points[p1_idx]
            orig_x2, orig_y2 = points[p2_idx]
            
            best_delta = 0
            best_move = None
            
            for _ in range(20):
                dx1 = rng.integers(-5, 6)
                dy1 = rng.integers(-5, 6)
                dx2 = rng.integers(-5, 6)
                dy2 = rng.integers(-5, 6)
                if dx1 == 0 and dy1 == 0 and dx2 == 0 and dy2 == 0:
                    continue
                
                nx1, ny1 = orig_x1 + dx1, orig_y1 + dy1
                nx2, ny2 = orig_x2 + dx2, orig_y2 + dy2
                
                if (nx1, ny1) in point_set and (nx1, ny1) != (orig_x1, orig_y1):
                    continue
                if (nx2, ny2) in point_set and (nx2, ny2) != (orig_x2, orig_y2):
                    continue
                if (nx1, ny1) == (nx2, ny2):
                    continue
                
                old_inc1 = int(np.sum(M[:, p1_idx]))
                old_inc2 = int(np.sum(M[:, p2_idx]))
                
                new_inc1 = 0
                new_inc2 = 0
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    if A * nx1 + B * ny1 + C == 0:
                        new_inc1 += 1
                    if A * nx2 + B * ny2 + C == 0:
                        new_inc2 += 1
                        
                delta = (new_inc1 + new_inc2) - (old_inc1 + old_inc2)
                if delta > best_delta:
                    best_delta = delta
                    best_move = (nx1, ny1, nx2, ny2)
            
            if best_move is not None and best_delta > 0:
                nx1, ny1, nx2, ny2 = best_move
                point_set.discard((orig_x1, orig_y1))
                point_set.discard((orig_x2, orig_y2))
                points[p1_idx] = [nx1, ny1]
                points[p2_idx] = [nx2, ny2]
                point_set.add((nx1, ny1))
                point_set.add((nx2, ny2))
                
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    M[l_idx, p1_idx] = (A * nx1 + B * ny1 + C == 0)
                    M[l_idx, p2_idx] = (A * nx2 + B * ny2 + C == 0)
                    
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    no_improve = 0

    return best_points, best_lines, best_score

def _build_hexagonal_points():
    """Build points in a hexagonal-like pattern."""
    points_list = []
    seen = set()
    for row in range(12):
        offset = row % 2
        for col in range(6):
            x = col * 2 + offset
            y = row
            if (x, y) not in seen:
                seen.add((x, y))
                points_list.append((x, y))
                if len(points_list) == 54:
                    break
        if len(points_list) == 54:
            break
    
    if len(points_list) < 54:
        # Fill remaining with a grid
        for row in range(12, 20):
            for col in range(6):
                x = col
                y = row
                if (x, y) not in seen:
                    seen.add((x, y))
                    points_list.append((x, y))
                    if len(points_list) == 54:
                        break
            if len(points_list) == 54:
                break
    
    return np.array(points_list[:54], dtype=np.int64)

def _build_clustered_points():
    """Build points in clusters to encourage multiple collinearities."""
    points_list = []
    seen = set()
    
    # Cluster 1: 3x3 grid
    for i in range(3):
        for j in range(3):
            points_list.append((i, j))
            seen.add((i, j))
    
    # Cluster 2: another 3x3 grid offset
    for i in range(3):
        for j in range(3):
            p = (i + 5, j + 4)
            if p not in seen:
                points_list.append(p)
                seen.add(p)
    
    # Cluster 3: another 3x3 grid
    for i in range(3):
        for j in range(3):
            p = (i + 10, j)
            if p not in seen:
                points_list.append(p)
                seen.add(p)
    
    # Cluster 4: another 3x3 grid
    for i in range(3):
        for j in range(3):
            p = (i + 10, j + 5)
            if p not in seen:
                points_list.append(p)
                seen.add(p)
    
    # Cluster 5: another 3x3 grid
    for i in range(3):
        for j in range(3):
            p = (i, j + 9)
            if p not in seen:
                points_list.append(p)
                seen.add(p)
    
    # Cluster 6: another 3x3 grid
    for i in range(3):
        for j in range(3):
            p = (i + 5, j + 9)
            if p not in seen:
                points_list.append(p)
                seen.add(p)
    
    # Fill remaining
    for row in range(15):
        for col in range(15):
            p = (col, row)
            if p not in seen:
                seen.add(p)
                points_list.append(p)
                if len(points_list) == 54:
                    break
        if len(points_list) == 54:
            break
    
    return np.array(points_list[:54], dtype=np.int64)

def szemeredi_trotter_incidences():
    deadline = time.time() + 600  # Safety margin
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Grid 9x6
    p1 = _build_grid_points(9, 6)
    l1 = _best_lines_for_points(p1, 27)
    if _is_valid(p1, l1):
        initial_configs.append((p1, l1, _incidence_count(p1, l1)))
    
    # Grid 6x9
    p2 = _build_grid_points(6, 9)
    l2 = _best_lines_for_points(p2, 27)
    if _is_valid(p2, l2):
        initial_configs.append((p2, l2, _incidence_count(p2, l2)))
    
    # Grid 18x3
    p3 = _build_grid_points(18, 3)
    l3 = _best_lines_for_points(p3, 27)
    if _is_valid(p3, l3):
        initial_configs.append((p3, l3, _incidence_count(p3, l3)))
    
    # Hexagonal
    p4 = _build_hexagonal_points()
    l4 = _best_lines_for_points(p4, 27)
    if _is_valid(p4, l4):
        initial_configs.append((p4, l4, _incidence_count(p4, l4)))
    
    # Clustered
    p5 = _build_clustered_points()
    l5 = _best_lines_for_points(p5, 27)
    if _is_valid(p5, l5):
        initial_configs.append((p5, l5, _incidence_count(p5, l5)))
    
    # Grid 7x8 (56 points, take 54)
    p6 = _build_grid_points(7, 8)[:54]
    l6 = _best_lines_for_points(p6, 27)
    if _is_valid(p6, l6):
        initial_configs.append((p6, l6, _incidence_count(p6, l6)))
    
    if not initial_configs:
        p1 = _build_grid_points(9, 6)
        l1 = _best_lines_for_points(p1, 27)
        return p1, l1
    
    # Sort by initial score
    initial_configs.sort(key=lambda x: -x[2])
    
    best_points, best_lines, best_score = initial_configs[0][0].copy(), initial_configs[0][1].copy(), initial_configs[0][2]
    
    # Time allocation
    n_configs = len(initial_configs)
    # Give more time to better initial configs
    total_time = deadline - time.time() - 10
    weights = [1.0 / (i + 1) for i in range(n_configs)]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    
    for i, (pts, lns, score) in enumerate(initial_configs):
        if time.time() >= deadline - 10:
            break
        
        alloc_time = total_time * weights[i]
        config_deadline = min(deadline - 10, time.time() + alloc_time)
        
        # Run local search
        pts2, lns2, score2 = _local_search(pts, lns, config_deadline, seed=42 + i)
        if score2 > best_score:
            best_score = score2
            best_points = pts2
            best_lines = lns2
        
        # If time remains, run simulated annealing
        if time.time() < config_deadline - 5:
            sa_deadline = min(config_deadline, time.time() + (config_deadline - time.time()) * 0.5)
            pts3, lns3, score3 = _simulated_annealing(pts, lns, sa_deadline, seed=100 + i)
            if score3 > best_score:
                best_score = score3
                best_points = pts3
                best_lines = lns3
        
        # If still time, run another local search from the SA result
        if time.time() < config_deadline - 5:
            pts4, lns4, score4 = _local_search(pts3, lns3, config_deadline, seed=200 + i)
            if score4 > best_score:
                best_score = score4
                best_points = pts4
                best_lines = lns4
    
    # Final validation
    if not _is_valid(best_points, best_lines):
        for pts, lns, score in initial_configs:
            if _is_valid(pts, lns):
                return pts, lns
        return initial_configs[0][0], initial_configs[0][1]
        
    return best_points, best_lines

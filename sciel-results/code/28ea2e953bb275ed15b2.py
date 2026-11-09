import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    # Canonical form: first non-zero is positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _compute_top27(points):
    """
    Compute the top 27 lines by incidence count for a set of 54 points.
    Returns (score, lines_array).
    """
    n = len(points)
    if n < 2:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    line_counts = {}
    # Generate all lines from pairs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            key = _normalize_line(A, B, C)
            if key is not None:
                if key in line_counts:
                    line_counts[key] += 1
                else:
                    line_counts[key] = 1
    
    # Convert pair counts to point incidences?
    # No, pair count is not point incidence.
    # A line with k points has C(k,2) pairs.
    # We need to find k for each line.
    
    # We have unique lines. For each, count points.
    # This is O(L * N). L is approx N^2/2 = 1400. N=54.
    # 1400 * 54 = 75600 ops.
    
    incidences = []
    for line_key, _ in line_counts.items():
        A, B, C = line_key
        count = 0
        for pt in points:
            if A * pt[0] + B * pt[1] + C == 0:
                count += 1
        incidences.append((count, line_key))
        
    # Sort by count descending
    incidences.sort(key=lambda x: (-x[0], x[1]))
    
    # Take top 27
    if len(incidences) >= 27:
        top27 = incidences[:27]
    else:
        top27 = incidences
        # Pad with dummy lines if necessary (should not happen for 54 points)
        while len(top27) < 27:
            top27.append((0, (1, 0, 1000000 + len(top27))))
            
    score = sum(c for c, _ in top27)
    lines_arr = np.array([l for _, l in top27], dtype=np.int64)
    
    return score, lines_arr

def _generate_grid(a, b):
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _check_valid_points(points):
    if points.shape != (54, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    if not np.allclose(points, np.round(points)):
        return False
    pt_set = set(map(tuple, points))
    if len(pt_set) != 54:
        return False
    return True

def _local_search(points, deadline, best_score, best_points, best_lines):
    n = len(points)
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1),
                 (-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,2),(-2,2),(2,-2)]
    
    current_points = points.copy()
    current_score = best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n):
            for dx, dy in neighbors:
                if time.time() > deadline:
                    break
                new_points = current_points.copy()
                new_points[idx] = current_points[idx] + np.array([dx, dy])
                
                if not _check_valid_points(new_points):
                    continue
                
                score, lines = _compute_top27(new_points)
                if score > current_score:
                    current_score = score
                    current_points = new_points
                    best_score = score
                    best_points = new_points.copy()
                    best_lines = lines
                    improved = True
                    break
            if time.time() > deadline:
                break
                
    return best_score, best_points, best_lines

def _simulated_annealing(points, deadline, best_score, best_points, best_lines):
    n = len(points)
    current = points.copy()
    current_score, _ = _compute_top27(current)
    
    T = 5.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        idx = np.random.randint(n)
        dx = np.random.randint(-3, 4)
        dy = np.random.randint(-3, 4)
        if dx == 0 and dy == 0:
            continue
        
        new_points = current.copy()
        new_points[idx] = current[idx] + np.array([dx, dy])
        
        if not _check_valid_points(new_points):
            T *= cooling
            continue
        
        new_score, new_lines = _compute_top27(new_points)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                best_lines = new_lines
        
        T *= cooling
    
    return best_score, best_points, best_lines

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 610  # Safety margin
    
    np.random.seed(42)
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, score, lines):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines

    # Phase 1: Try different grid configurations
    grid_configs = [
        (9, 6),
        (6, 9),
        (7, 8),
        (8, 7),
        (5, 11),
        (11, 5),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
    ]
    
    for a, b in grid_configs:
        if time.time() > deadline - 60:
            break
        pts = _generate_grid(a, b)
        if len(pts) != 54:
            continue
        if not _check_valid_points(pts):
            continue
        score, lines = _compute_top27(pts)
        update_best(pts, score, lines)
        
        # Quick local search from this grid
        if time.time() < deadline - 30:
            best_score, best_points, best_lines = _local_search(
                best_points, deadline - 10, best_score, best_points, best_lines
            )

    # Phase 2: Simulated Annealing from the best found so far
    if best_points is not None and time.time() < deadline - 10:
        best_score, best_points, best_lines = _simulated_annealing(
            best_points, deadline, best_score, best_points, best_lines
        )

    return best_points, best_lines

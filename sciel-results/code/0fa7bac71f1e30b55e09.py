# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    pt_set = set()
    for i in range(54):
        x, y = points[i]
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return False
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) in pt_set:
            return False
        pt_set.add((xi, yi))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(C):
            return False
        if abs(A - round(A)) > 1e-6 or abs(B - round(B)) > 1e-6 or abs(C - round(C)) > 1e-6:
            return False
        Ai, Bi, Ci = int(round(A)), int(round(B)), int(round(C))
        if Ai == 0 and Bi == 0:
            return False
        key = _normalize_line(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True

def _count_incidences(points, lines):
    x = points[:, 0:1].astype(np.int64)
    y = points[:, 1:2].astype(np.int64)
    A = lines[:, 0:1].astype(np.int64)
    B = lines[:, 1:2].astype(np.int64)
    C = lines[:, 2:3].astype(np.int64)
    vals = x * A.T + y * B.T + C.T
    return int(np.sum(vals == 0))

def _get_best_lines(points):
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(int(A), int(B), int(C))
            if key[0] == 0 and key[1] == 0:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    sorted_lines = sorted(line_counts.items(), key=lambda kv: kv[1], reverse=True)
    selected = [key for key, cnt in sorted_lines[:27]]
    
    if len(selected) < 27:
        existing_set = set(selected)
        for i in range(n):
            if len(selected) >= 27:
                break
            x, y = points[i]
            for A, B, C in [(1, 0, -x), (0, 1, -y), (1, 1, -(x+y)), (1, -1, y-x)]:
                key = _normalize_line(A, B, C)
                if not (key[0] == 0 and key[1] == 0) and key not in existing_set:
                    selected.append(key)
                    existing_set.add(key)
                    break
    return np.array(selected[:27], dtype=np.int64)

def _generate_candidate_points():
    """Generate a diverse set of candidate point configurations."""
    candidates = []
    
    # 1. 6x9 Grid
    xs = list(range(1, 7))
    ys = list(range(1, 10))
    pts = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
    candidates.append(pts)
    
    # 2. 9x6 Grid
    xs = list(range(1, 10))
    ys = list(range(1, 7))
    pts = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
    candidates.append(pts)
    
    # 3. 7x8 Grid with one row/col removed to make 54? 7*8=56. Remove 2.
    # Let's try a slightly perturbed grid
    base = np.array([(x, y) for x in range(1, 8) for y in range(1, 8)], dtype=np.int64)
    # Remove last 2 points
    pts = base[:54].copy()
    candidates.append(pts)
    
    # 4. "Star" pattern: Points on lines through origin
    # This is hard to generate directly as a set of 54 unique integer points with high collinearity
    # Let's stick to grid variations which are known to be strong.
    
    # 5. Grid with shifted rows (shear)
    pts = np.zeros((54, 2), dtype=np.int64)
    idx = 0
    for y in range(1, 10):
        for x in range(1, 7):
            pts[idx, 0] = x + y // 2  # Shear
            pts[idx, 1] = y
            idx += 1
    # Ensure uniqueness
    unique_pts = np.unique(pts, axis=0)
    if len(unique_pts) == 54:
        candidates.append(unique_pts)
    
    # 6. Random initialization (seeded for reproducibility)
    rng = np.random.default_rng(123)
    for _ in range(5):
        pts = rng.integers(0, 20, size=(54, 2))
        # Ensure unique
        unique_pts = np.unique(pts, axis=0)
        if len(unique_pts) < 54:
            # Fill remaining with random
            while len(unique_pts) < 54:
                new_pt = rng.integers(0, 20, size=(1, 2))
                unique_pts = np.vstack([unique_pts, new_pt])
                unique_pts = np.unique(unique_pts, axis=0)
        if len(unique_pts) == 54:
            candidates.append(unique_pts.astype(np.int64))
            
    return candidates

def _steepest_ascent_local_search(points, deadline):
    """Deterministic local search: always accept the best single-point move."""
    best_points = points.copy()
    best_lines = _get_best_lines(best_points)
    best_count = _count_incidences(best_points, best_lines)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        best_move = None
        best_move_count = best_count - 1 # Must be strictly better
        
        # Try moving each point by a small offset
        # To speed up, we can limit the search space of offsets
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for i in range(54):
            orig_x, orig_y = best_points[i]
            for dx, dy in offsets:
                new_x, new_y = orig_x + dx, orig_y + dy
                # Check if new point is already occupied
                if any(best_points[j, 0] == new_x and best_points[j, 1] == new_y for j in range(54) if j != i):
                    continue
                
                new_points = best_points.copy()
                new_points[i, 0] = new_x
                new_points[i, 1] = new_y
                
                new_lines = _get_best_lines(new_points)
                new_count = _count_incidences(new_points, new_lines)
                
                if new_count > best_move_count:
                    best_move_count = new_count
                    best_move = (i, new_x, new_y)
        
        if best_move is not None:
            i, new_x, new_y = best_move
            best_points[i, 0] = new_x
            best_points[i, 1] = new_y
            best_lines = _get_best_lines(best_points)
            best_count = best_move_count
            improved = True
            
    return best_points, best_lines, best_count

def szemeredi_trotter_incidences():
    deadline = time.time() + 600
    safety_margin = 30
    
    # Generate candidates
    candidates = _generate_candidate_points()
    
    best_overall_points = None
    best_overall_lines = None
    best_overall_count = 0
    
    # Evaluate each candidate and refine
    for cand in candidates:
        if time.time() > deadline - safety_margin:
            break
        
        # Initial refinement
        pts, lines, count = _steepest_ascent_local_search(cand, deadline)
        
        if count > best_overall_count:
            best_overall_count = count
            best_overall_points = pts
            best_overall_lines = lines
            
    # If we still have time, do a few more random restarts with local search
    rng = np.random.default_rng(42)
    while time.time() < deadline - safety_margin:
        # Generate a random point set
        pts = rng.integers(0, 30, size=(54, 2))
        unique_pts = np.unique(pts, axis=0)
        if len(unique_pts) < 54:
            while len(unique_pts) < 54:
                new_pt = rng.integers(0, 30, size=(1, 2))
                unique_pts = np.vstack([unique_pts, new_pt])
                unique_pts = np.unique(unique_pts, axis=0)
        if len(unique_pts) != 54:
            continue
        unique_pts = unique_pts.astype(np.int64)
        
        # Refine
        refined_pts, refined_lines, refined_count = _steepest_ascent_local_search(unique_pts, deadline)
        
        if refined_count > best_overall_count:
            best_overall_count = refined_count
            best_overall_points = refined_pts
            best_overall_lines = refined_lines
    
    # Final validation
    if not _is_valid(best_overall_points, best_overall_lines):
        # Fallback to a known valid configuration
        xs = list(range(1, 7))
        ys = list(range(1, 10))
        best_overall_points = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
        best_overall_lines = _get_best_lines(best_overall_points)
        
    return best_overall_points, best_overall_lines
# EVOLVE-BLOCK-END

import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _all_lines_and_counts(points):
    """Compute all lines through pairs of points and their incidence counts."""
    n = points.shape[0]
    lines_dict = {}
    point_lines = [set() for _ in range(n)]
    
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            key = _normalize_line(A, B, C)
            if key is not None:
                if key not in lines_dict:
                    lines_dict[key] = 0
                point_lines[i].add(key)
                point_lines[j].add(key)
    
    # Count incidences for each line
    for key in lines_dict:
        A, B, C = key
        vals = A * points[:, 0] + B * points[:, 1] + C
        lines_dict[key] = int(np.sum(vals == 0))
        
    return lines_dict, point_lines

def _top_k_sum(counts, k=27):
    if len(counts) < k:
        return sum(counts.values())
    sorted_vals = sorted(counts.values(), reverse=True)
    return sum(sorted_vals[:k])

def _evaluate_configuration(points):
    """Evaluate the best 27 lines for a given set of points."""
    lines_dict, _ = _all_lines_and_counts(points)
    return _top_k_sum(lines_dict, 27)

def _get_top_lines(points, k=27):
    lines_dict, _ = _all_lines_and_counts(points)
    sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
    selected = [l for l, c in sorted_lines[:k]]
    
    if len(selected) < k:
        all_keys = list(lines_dict.keys())
        sel_keys = set(selected)
        pad_keys = [key for key in all_keys if key not in sel_keys]
        while len(pad_keys) < k - len(selected):
            pad_keys.append((1, 0, 1000000 + len(pad_keys)))
        selected.extend(pad_keys[:k - len(selected)])
        
    return np.array(selected, dtype=np.int64)

def _generate_grid(a, b):
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _evaluate_removal(base_points, num_remove, time_budget=30):
    """Try removing num_remove points from base_points to maximize incidence."""
    n = base_points.shape[0]
    if n < num_remove:
        return -1, None
    
    lines_dict, point_lines = _all_lines_and_counts(base_points)
    start = time.time()
    
    best_score = -1
    best_mask = None
    
    if num_remove == 0:
        score = _top_k_sum(lines_dict)
        return score, base_points.copy()
        
    if num_remove == 1:
        for i in range(n):
            if time.time() - start > time_budget:
                break
            temp_counts = dict(lines_dict)
            for key in point_lines[i]:
                temp_counts[key] -= 1
            score = _top_k_sum(temp_counts)
            if score > best_score:
                best_score = score
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                best_mask = mask
    elif num_remove == 2:
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() - start > time_budget:
                    break
                temp_counts = dict(lines_dict)
                for key in point_lines[i]:
                    temp_counts[key] -= 1
                for key in point_lines[j]:
                    temp_counts[key] -= 1
                score = _top_k_sum(temp_counts)
                if score > best_score:
                    best_score = score
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    best_mask = mask
    
    if best_mask is None:
        return -1, None
    return best_score, base_points[best_mask]

def _local_swap_search(points, max_rounds=5, time_budget=120):
    """Local search: try moving each point to nearby positions."""
    current_points = points.copy()
    current_score = _evaluate_configuration(current_points)
    deadline = time.time() + time_budget
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        n = len(current_points)
        for i in range(n):
            if time.time() > deadline:
                break
            orig_pt = current_points[i]
            # Try all 8 neighbors plus staying
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx == 0 and dy == 0:
                        continue
                    cand = (int(orig_pt[0] + dx), int(orig_pt[1] + dy))
                    # Check no duplicate
                    if np.any((current_points[:, 0] == cand[0]) & (current_points[:, 1] == cand[1])):
                        continue
                    trial = current_points.copy()
                    trial[i] = cand
                    trial_score = _evaluate_configuration(trial)
                    if trial_score > current_score:
                        current_points = trial
                        current_score = trial_score
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    return current_score, current_points

def _try_addition_and_removal(points, time_budget=60):
    """Try adding a point near existing points then removing one for better configuration."""
    current_points = points.copy()
    current_score = _evaluate_configuration(current_points)
    deadline = time.time() + time_budget
    
    n = len(current_points)
    # Get all lines for quick evaluation
    lines_dict, point_lines = _all_lines_and_counts(current_points)
    
    # Try adding each candidate point and removing each existing point
    # Candidate points: near existing points
    candidates = set()
    for i in range(n):
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                cand = (int(current_points[i, 0] + dx), int(current_points[i, 1] + dy))
                if cand not in set(map(tuple, current_points)):
                    candidates.add(cand)
    
    best_score = current_score
    best_points = current_points.copy()
    
    for cand in candidates:
        if time.time() > deadline:
            break
        # Add candidate
        new_points = np.vstack([current_points, np.array(cand, dtype=np.int64)])
        # Now remove one point to get back to 54
        new_lines_dict, new_point_lines = _all_lines_and_counts(new_points)
        n_new = len(new_points)
        
        for i in range(n_new):
            if time.time() > deadline:
                break
            temp_counts = dict(new_lines_dict)
            for key in new_point_lines[i]:
                temp_counts[key] -= 1
            score = _top_k_sum(temp_counts)
            if score > best_score:
                best_score = score
                mask = np.ones(n_new, dtype=bool)
                mask[i] = False
                best_points = new_points[mask]
    
    return best_score, best_points

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, score):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts
            best_lines = _get_top_lines(pts, 27)
    
    # 1. Try various grid configurations
    grid_configs = [
        (8, 7, 2),   # 8x7=56, remove 2
        (7, 8, 2),   # 7x8=56, remove 2
        (6, 9, 0),   # 6x9=54, remove 0
        (9, 6, 0),   # 9x6=54, remove 0
        (5, 11, 1),  # 5x11=55, remove 1
        (11, 5, 1),  # 11x5=55, remove 1
        (4, 14, 2),  # 4x14=56, remove 2
        (14, 4, 2),  # 14x4=56, remove 2
        (3, 18, 0),  # 3x18=54, remove 0
        (18, 3, 0),  # 18x3=54, remove 0
    ]
    
    for a, b, remove in grid_configs:
        if time.time() > deadline - 120:
            break
        pts = _generate_grid(a, b)
        if len(pts) != a * b:
            continue
        
        if remove == 0:
            score = _evaluate_configuration(pts)
            update_best(pts, score)
        else:
            remaining = deadline - 120 - time.time()
            if remaining > 10:
                score, pts_best = _evaluate_removal(pts, remove, time_budget=min(remaining, 40))
                if pts_best is not None:
                    real_score = _evaluate_configuration(pts_best)
                    update_best(pts_best, real_score)
    
    # 2. Local search on best configuration
    if best_points is not None and time.time() < deadline - 60:
        remaining = deadline - 60 - time.time()
        if remaining > 30:
            new_score, new_points = _local_swap_search(best_points, max_rounds=5, 
                                                        time_budget=min(remaining, 150))
            if new_score > best_score:
                update_best(new_points, new_score)
    
    # 3. Try addition-removal search
    if best_points is not None and time.time() < deadline - 30:
        remaining = deadline - 30 - time.time()
        if remaining > 20:
            new_score, new_points = _try_addition_and_removal(best_points, 
                                                               time_budget=min(remaining, 60))
            if new_score > best_score:
                update_best(new_points, new_score)
    
    # 4. Final local search if time permits
    if best_points is not None and time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        if remaining > 15:
            new_score, new_points = _local_swap_search(best_points, max_rounds=3,
                                                        time_budget=min(remaining, 40))
            if new_score > best_score:
                update_best(new_points, new_score)
    
    # Fallback
    if best_points is None:
        pts = _generate_grid(6, 9)
        score = _evaluate_configuration(pts)
        update_best(pts, score)
    
    return best_points, best_lines

import numpy as np
import time
from math import gcd
from itertools import combinations, product

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

def _count_incidence_for_line(A, B, C, points):
    vals = A * points[:, 0] + B * points[:, 1] + C
    return int(np.sum(vals == 0))

def _get_all_lines_and_counts(points):
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
                
    for key in lines_dict:
        A, B, C = key
        lines_dict[key] = _count_incidence_for_line(A, B, C, points)
        
    return lines_dict, point_lines

def _top_k_sum(counts, k=27):
    if len(counts) < k:
        return sum(counts.values())
    sorted_vals = sorted(counts.values(), reverse=True)
    return sum(sorted_vals[:k])

def _evaluate_removal(base_points, num_remove):
    n = base_points.shape[0]
    if n < num_remove:
        return -1, None
    
    lines_dict, point_lines = _get_all_lines_and_counts(base_points)
    
    best_score = -1
    best_points = None
    
    if num_remove == 0:
        score = _top_k_sum(lines_dict)
        return score, base_points
        
    if num_remove == 1:
        for i in range(n):
            temp_counts = dict(lines_dict)
            for key in point_lines[i]:
                temp_counts[key] -= 1
            score = _top_k_sum(temp_counts)
            if score > best_score:
                best_score = score
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                best_points = base_points[mask]
                
    elif num_remove == 2:
        for i in range(n):
            for j in range(i + 1, n):
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
                    best_points = base_points[mask]
                    
    else:
        # For larger removals, we might need a heuristic or smaller search space
        pass
        
    return best_score, best_points

def _generate_grid(a, b):
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _get_top_lines(points, k=27):
    lines_dict, _ = _get_all_lines_and_counts(points)
    sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
    selected = [l for l, c in sorted_lines[:k]]
    
    # Pad if necessary
    if len(selected) < k:
        all_keys = list(lines_dict.keys())
        sel_keys = set(selected)
        pad_keys = [key for key in all_keys if key not in sel_keys]
        while len(pad_keys) < k - len(selected):
            pad_keys.append((1, 0, 1000000 + len(pad_keys)))
        selected.extend(pad_keys[:k - len(selected)])
        
    return np.array(selected, dtype=np.int64)

def _evaluate_configuration(points):
    """
    Given exactly 54 points, evaluate the best possible set of 27 lines.
    This is equivalent to _top_k_sum of all lines defined by pairs of these points.
    """
    lines_dict, _ = _get_all_lines_and_counts(points)
    return _top_k_sum(lines_dict, 27)

def _local_swap_search(points, max_rounds=3, time_budget=None):
    """
    Local search: try moving each point to a nearby position (±1 in x and y).
    Keep the move if it improves the score.
    """
    current_points = points.copy()
    current_score = _evaluate_configuration(current_points)
    
    if time_budget is not None:
        deadline = time.time() + time_budget
    else:
        deadline = float('inf')
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        n = len(current_points)
        for i in range(n):
            if time.time() > deadline:
                break
            orig_pt = current_points[i]
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    cand = (int(orig_pt[0] + dx), int(orig_pt[1] + dy))
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
            best_points = pts
            best_lines = _get_top_lines(pts, 27)
            
    # 1. Try exact grids
    grid_candidates = [
        (6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2), (1, 54), (54, 1)
    ]
    for a, b in grid_candidates:
        if time.time() > deadline - 30:
            break
        pts = _generate_grid(a, b)
        if len(pts) != 54:
            continue
        score = _evaluate_configuration(pts)
        update_best(pts, score)
        
    # 2. Try grids with removals
    if time.time() < deadline - 30:
        pts_base = _generate_grid(7, 8)
        score, pts_best = _evaluate_removal(pts_base, 2)
        if pts_best is not None:
            real_score = _evaluate_configuration(pts_best)
            update_best(pts_best, real_score)
            
    if time.time() < deadline - 30:
        pts_base = _generate_grid(5, 11)
        score, pts_best = _evaluate_removal(pts_base, 1)
        if pts_best is not None:
            real_score = _evaluate_configuration(pts_best)
            update_best(pts_best, real_score)
            
    if time.time() < deadline - 30:
        pts_base = _generate_grid(11, 5)
        score, pts_best = _evaluate_removal(pts_base, 1)
        if pts_best is not None:
            real_score = _evaluate_configuration(pts_best)
            update_best(pts_best, real_score)
            
    if time.time() < deadline - 30:
        pts_base = _generate_grid(4, 14)
        score, pts_best = _evaluate_removal(pts_base, 2)
        if pts_best is not None:
            real_score = _evaluate_configuration(pts_best)
            update_best(pts_best, real_score)
            
    if time.time() < deadline - 30:
        pts_base = _generate_grid(14, 4)
        score, pts_best = _evaluate_removal(pts_base, 2)
        if pts_best is not None:
            real_score = _evaluate_configuration(pts_best)
            update_best(pts_best, real_score)
            
    # 3. Local search on the best configuration found so far
    if best_points is not None and time.time() < deadline - 20:
        remaining_time = deadline - 20 - time.time()
        new_score, new_points = _local_swap_search(best_points, max_rounds=3, 
                                                    time_budget=remaining_time)
        if new_score > best_score:
            update_best(new_points, new_score)
        
    # Fallback
    if best_points is None:
        pts = _generate_grid(6, 9)
        score = _evaluate_configuration(pts)
        update_best(pts, score)
        
    return best_points, best_lines

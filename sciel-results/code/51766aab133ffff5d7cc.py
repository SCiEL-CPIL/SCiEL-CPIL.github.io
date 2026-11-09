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

def _generate_grid(a, b):
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _get_top_lines(points, k=27):
    lines_dict, _ = _get_all_lines_and_counts(points)
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

def _evaluate_configuration(points):
    lines_dict, _ = _get_all_lines_and_counts(points)
    return _top_k_sum(lines_dict, 27)

def _find_line_through_two_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    A = dy
    B = -dx
    C = dx * p1[1] - dy * p1[0]
    return _normalize_line(A, B, C)

def _point_on_line(pt, line):
    A, B, C = line
    return A * pt[0] + B * pt[1] + C == 0

def _smart_local_search(points, time_budget, max_rounds=10):
    """
    Smart local search: for each point, try moving it to positions that
    lie on existing high-value lines or create new multi-point lines.
    """
    current_points = points.copy()
    current_score = _evaluate_configuration(current_points)
    deadline = time.time() + time_budget
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        n = len(current_points)
        
        # Recompute lines and counts
        lines_dict, point_lines = _get_all_lines_and_counts(current_points)
        
        # Get top lines sorted by count
        top_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
        
        for i in range(n):
            if time.time() > deadline:
                break
            orig_pt = current_points[i]
            orig_lines = point_lines[i]
            
            # Candidate positions: points on top lines (excluding already occupied)
            occupied = set(map(tuple, current_points))
            candidates = set()
            
            # For each top line, find integer points on it near the grid
            for line, count in top_lines[:15]:
                if count < 2:
                    break
                A, B, C = line
                # Find points on this line within a reasonable range
                # Line: A*x + B*y + C = 0
                # Try x in a range around the original point
                for x in range(max(1, int(orig_pt[0]) - 5), min(20, int(orig_pt[0]) + 6)):
                    if A * x + C == 0 and B != 0:
                        continue
                    if B != 0:
                        num = -(A * x + C)
                        if num % B == 0:
                            y = num // B
                            if 1 <= y <= 20:
                                cand = (x, y)
                                if cand not in occupied and cand != tuple(orig_pt):
                                    candidates.add(cand)
                # Also try varying y
                for y in range(max(1, int(orig_pt[1]) - 5), min(20, int(orig_pt[1]) + 6)):
                    if B * y + C == 0 and A != 0:
                        continue
                    if A != 0:
                        num = -(B * y + C)
                        if num % A == 0:
                            x = num // A
                            if 1 <= x <= 20:
                                cand = (x, y)
                                if cand not in occupied and cand != tuple(orig_pt):
                                    candidates.add(cand)
            
            # Also try nearby positions (±1, ±2)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    cand = (int(orig_pt[0]) + dx, int(orig_pt[1]) + dy)
                    if cand[0] >= 1 and cand[1] >= 1 and cand not in occupied:
                        candidates.add(cand)
            
            # Try each candidate
            best_trial_score = current_score
            best_trial_pt = None
            for cand in candidates:
                if time.time() > deadline:
                    break
                trial = current_points.copy()
                trial[i] = cand
                trial_score = _evaluate_configuration(trial)
                if trial_score > best_trial_score:
                    best_trial_score = trial_score
                    best_trial_pt = cand
            
            if best_trial_pt is not None:
                current_points[i] = best_trial_pt
                current_score = best_trial_score
                improved = True
                # Early exit if we found a good improvement
                break
        
        if not improved:
            break
    
    return current_score, current_points

def _pair_swap_search(points, time_budget, max_rounds=5):
    """
    Try removing two points and adding two new points that create a line
    with 3+ points total.
    """
    current_points = points.copy()
    current_score = _evaluate_configuration(current_points)
    deadline = time.time() + time_budget
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        n = len(current_points)
        
        lines_dict, point_lines = _get_all_lines_and_counts(current_points)
        top_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
        
        # Try pairs of points to remove
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                # Remove points i and j
                remaining = [current_points[k] for k in range(n) if k != i and k != j]
                remaining_set = set(map(tuple, remaining))
                
                # Find lines in the remaining set that have 2+ points
                # and try to add points on those lines
                rem_lines = {}
                for a in range(len(remaining)):
                    for b in range(a + 1, len(remaining)):
                        key = _find_line_through_two_points(remaining[a], remaining[b])
                        if key is not None:
                            if key not in rem_lines:
                                rem_lines[key] = set()
                            rem_lines[key].add(a)
                            rem_lines[key].add(b)
                
                # Find lines with exactly 2 points in remaining (adding a 3rd would be +1)
                # or lines with 1 point (adding 2 would be +2)
                candidate_new_pts = set()
                for line, pt_indices in rem_lines.items():
                    if len(pt_indices) == 2:
                        A, B, C = line
                        # Find a point on this line not in remaining
                        for x in range(1, 15):
                            if B != 0:
                                num = -(A * x + C)
                                if num % B == 0:
                                    y = num // B
                                    if 1 <= y <= 15 and (x, y) not in remaining_set:
                                        candidate_new_pts.add((x, y))
                        for y in range(1, 15):
                            if A != 0:
                                num = -(B * y + C)
                                if num % A == 0:
                                    x = num // A
                                    if 1 <= x <= 15 and (x, y) not in remaining_set:
                                        candidate_new_pts.add((x, y))
                
                # Try adding pairs of candidate points
                cands = list(candidate_new_pts)[:20]
                for a_idx in range(len(cands)):
                    if time.time() > deadline:
                        break
                    for b_idx in range(a_idx + 1, len(cands)):
                        trial = np.array(remaining + [cands[a_idx], cands[b_idx]], dtype=np.int64)
                        if len(trial) != 54:
                            continue
                        # Check no duplicates
                        if len(set(map(tuple, trial))) != 54:
                            continue
                        trial_score = _evaluate_configuration(trial)
                        if trial_score > current_score:
                            current_points = trial
                            current_score = trial_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return current_score, current_points

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
                    
    return best_score, best_points

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
        if time.time() > deadline - 60:
            break
        pts = _generate_grid(a, b)
        if len(pts) != 54:
            continue
        score = _evaluate_configuration(pts)
        update_best(pts, score)
        
    # 2. Try grids with removals
    if time.time() < deadline - 60:
        pts_base = _generate_grid(7, 8)
        score, pts_best = _evaluate_removal(pts_base, 2)
        if pts_best is not None:
            real_score = _evaluate_configuration(pts_best)
            update_best(pts_best, real_score)
            
    if time.time() < deadline - 60:
        pts_base = _generate_grid(5, 11)
        score, pts_best = _evaluate_removal(pts_base, 1)
        if pts_best is not None:
            real_score = _evaluate_configuration(pts_best)
            update_best(pts_best, real_score)
            
    if time.time() < deadline - 60:
        pts_base = _generate_grid(11, 5)
        score, pts_best = _evaluate_removal(pts_base, 1)
        if pts_best is not None:
            real_score = _evaluate_configuration(pts_best)
            update_best(pts_best, real_score)
            
    if time.time() < deadline - 60:
        pts_base = _generate_grid(4, 14)
        score, pts_best = _evaluate_removal(pts_base, 2)
        if pts_best is not None:
            real_score = _evaluate_configuration(pts_best)
            update_best(pts_best, real_score)
            
    if time.time() < deadline - 60:
        pts_base = _generate_grid(14, 4)
        score, pts_best = _evaluate_removal(pts_base, 2)
        if pts_best is not None:
            real_score = _evaluate_configuration(pts_best)
            update_best(pts_best, real_score)
            
    # 3. Smart local search on the best configuration
    if best_points is not None and time.time() < deadline - 30:
        remaining_time = deadline - 30 - time.time()
        # Split time between smart search and pair swap
        t1 = remaining_time * 0.6
        t2 = remaining_time * 0.4
        
        new_score, new_points = _smart_local_search(best_points, time_budget=t1, max_rounds=5)
        if new_score > best_score:
            update_best(new_points, new_score)
        
        if time.time() < deadline - 10:
            remaining_time2 = deadline - 10 - time.time()
            new_score2, new_points2 = _pair_swap_search(best_points, time_budget=remaining_time2, max_rounds=3)
            if new_score2 > best_score:
                update_best(new_points2, new_score2)
    
    # 4. Additional local search rounds if time permits
    if best_points is not None and time.time() < deadline - 10:
        remaining_time = deadline - 10 - time.time()
        new_score, new_points = _smart_local_search(best_points, time_budget=remaining_time, max_rounds=3)
        if new_score > best_score:
            update_best(new_points, new_score)
        
    # Fallback
    if best_points is None:
        pts = _generate_grid(6, 9)
        score = _evaluate_configuration(pts)
        update_best(pts, score)
        
    return best_points, best_lines

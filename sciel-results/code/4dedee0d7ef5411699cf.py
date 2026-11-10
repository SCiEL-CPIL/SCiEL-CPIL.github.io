import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140

    # Use a larger grid for more diversity
    N = 9
    points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(points)

    # Precompute all lines (as frozensets of point indices)
    # For each pair, find all collinear points
    all_lines = set()
    for i in range(n_points):
        for j in range(i + 1, n_points):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            pts = {i, j}
            for k in range(n_points):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    pts.add(k)
            all_lines.add(frozenset(pts))

    # Classify lines
    forbidden_lines = [l for l in all_lines if len(l) >= 5]
    target_lines = [l for l in all_lines if len(l) == 4]
    longer_lines = [l for l in all_lines if len(l) > 4]

    # Build point-to-line index for fast delta computation
    # point_lines[i] = list of (line, size) where line is frozenset, size is len(line)
    point_lines = defaultdict(list)
    for line in all_lines:
        for p in line:
            point_lines[p].append((line, len(line)))

    def is_valid(selected):
        sel_frozen = frozenset(selected)
        for fl in forbidden_lines:
            if fl.issubset(sel_frozen):
                return False
        return True

    def get_score(selected):
        sel_frozen = frozenset(selected)
        score = 0
        for tl in target_lines:
            if tl.issubset(sel_frozen):
                score += 1
        for ll in longer_lines:
            if len(ll & sel_frozen) == 4:
                score += 1
        return score

    def delta_score_1swap(selected, p_out, p_in):
        """Compute score change when removing p_out and adding p_in."""
        sel = set(selected)
        new_sel = set(sel)
        new_sel.remove(p_out)
        new_sel.add(p_in)
        sel_frozen = frozenset(new_sel)
        
        # Check validity
        for fl in forbidden_lines:
            if fl.issubset(sel_frozen):
                return None  # Invalid
        
        # Compute delta
        delta = 0
        # Lines that had p_out but not p_in: lose p_out
        for line, size in point_lines[p_out]:
            if p_in not in line:
                # Before: line had some count from selected
                # After: line has one less
                before_count = len(line & sel_frozen) + 1  # +1 because p_out was in selected
                after_count = before_count - 1
                if before_count == 4 and size == 4:
                    delta -= 1
                elif before_count == 4 and size > 4:
                    delta -= 1
                elif after_count == 4 and size == 4:
                    delta += 1
                elif after_count == 4 and size > 4:
                    delta += 1
            # If p_in is also in line, net change is 0 for this line
        
        # Lines that have p_in but not p_out: gain p_in
        for line, size in point_lines[p_in]:
            if p_out not in line:
                before_count = len(line & sel_frozen) - 1  # -1 because p_in wasn't in selected
                after_count = before_count + 1
                if before_count == 4 and size == 4:
                    delta -= 1
                elif before_count == 4 and size > 4:
                    delta -= 1
                elif after_count == 4 and size == 4:
                    delta += 1
                elif after_count == 4 and size > 4:
                    delta += 1
        
        return delta

    best_score = -1
    best_indices = None

    def hill_climb(initial_set, max_iters=10000):
        nonlocal best_score, best_indices
        
        current = set(initial_set)
        current_score = get_score(current)
        
        if not is_valid(current):
            return -1, set()
        
        if current_score > best_score:
            best_score = current_score
            best_indices = set(current)
        
        no_improve = 0
        stagnation_limit = 500
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if no_improve > stagnation_limit:
                break
            
            # Try random 1-for-1 swap
            to_remove = random.choice(list(current))
            candidates_in = [p for p in range(n_points) if p not in current]
            if not candidates_in:
                break
            to_add = random.choice(candidates_in)
            
            delta = delta_score_1swap(current, to_remove, to_add)
            if delta is None:
                no_improve += 1
                continue
            
            if delta > 0:
                current.remove(to_remove)
                current.add(to_add)
                current_score += delta
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_indices = set(current)
            else:
                no_improve += 1
        
        return current_score, current

    # Multiple restarts with diverse initial configurations
    n_restarts = 80
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate diverse initial sets
        initial = None
        for _ in range(50):
            cand = set(random.sample(range(n_points), 16))
            if is_valid(cand):
                initial = cand
                break
        
        if initial is None:
            continue
        
        _, _ = hill_climb(initial)
    
    # Final polish: exhaustive 1-swap on best solution
    if best_indices is not None:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_score = get_score(best_indices)
            for p_out in list(best_indices):
                if time.time() > deadline:
                    break
                for p_in in range(n_points):
                    if p_in in best_indices:
                        continue
                    delta = delta_score_1swap(best_indices, p_out, p_in)
                    if delta is not None and delta > 0:
                        best_indices.remove(p_out)
                        best_indices.add(p_in)
                        improved = True
                        break
                if improved:
                    break
    
    # Fallback
    if best_indices is None:
        for _ in range(200):
            candidate = set(random.sample(range(n_points), 16))
            if is_valid(candidate):
                best_indices = candidate
                break
        if best_indices is None:
            best_indices = set(random.sample(range(n_points), 16))

    return [[points[i][0], points[i][1]] for i in sorted(best_indices)]

import random
import time
from math import gcd
import numpy as np
from collections import Counter, defaultdict

def _get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = -(A * x1 + B * y1)
    
    g = gcd(abs(A), abs(B))
    if g == 0:
        g = abs(C)
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _evaluate_vectorized(points):
    """Fast vectorized evaluation using NumPy."""
    n = len(points)
    if n != 16:
        return False, 0
    
    xs = np.array([p[0] for p in points], dtype=np.int64)
    ys = np.array([p[1] for p in points], dtype=np.int64)
    
    if np.any(np.abs(xs) > 1000) or np.any(np.abs(ys) > 1000):
        return False, 0
    
    pts = np.column_stack((xs, ys))
    if len(set(map(tuple, pts.tolist()))) != 16:
        return False, 0
    
    iu, ju = np.triu_indices(16, k=1)
    dx = xs[ju] - xs[iu]
    dy = ys[ju] - ys[iu]
    
    ax = np.abs(dx)
    ay = np.abs(dy)
    g = np.gcd(ax, ay)
    g_safe = np.where(g == 0, 1, g)
    dxn = dx // g_safe
    dyn = dy // g_safe
    
    neg = (dxn < 0) | ((dxn == 0) & (dyn < 0))
    dxn = np.where(neg, -dxn, dxn)
    dyn = np.where(neg, -dyn, dyn)
    
    c = dyn * xs[iu] - dxn * ys[iu]
    keys = np.column_stack((dxn, dyn, c))
    keys_t = list(map(tuple, keys.tolist()))
    cnt = Counter(keys_t)
    
    max_count = 0
    score = 0
    for k, v in cnt.items():
        if v > max_count:
            max_count = v
        if v == 6:  # exactly 4 points -> C(4,2)=6 pairs
            score += 1
    
    if max_count >= 10:  # 5 or more points -> C(5,2)=10 pairs
        return False, 0
    
    return True, score

def _build_line_index(points):
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    return line_points

def _evaluate_from_index(points, line_points):
    valid = True
    score = 0
    for key, idx_set in line_points.items():
        cnt = len(idx_set)
        if cnt >= 5:
            valid = False
            break
        elif cnt == 4:
            score += 1
    return valid, score

def _incremental_update(points, line_points, old_idx, old_pt, new_pt):
    n = len(points)
    lines_to_remove = []
    for key, idx_set in line_points.items():
        if old_idx in idx_set:
            lines_to_remove.append(key)
    
    for key in lines_to_remove:
        idx_set = line_points[key]
        idx_set.discard(old_idx)
        if len(idx_set) <= 1:
            del line_points[key]
    
    for j in range(n):
        if j == old_idx:
            continue
        key = _get_line_key(new_pt, points[j])
        if key not in line_points:
            line_points[key] = set()
        line_points[key].add(old_idx)
        line_points[key].add(j)
    
    return line_points

def _generate_informed_moves(points, line_points, idx, rng):
    old_x, old_y = points[idx]
    moves = []
    
    # 1) Small perturbations
    for _ in range(8):
        dx = rng.randint(-5, 5)
        dy = rng.randint(-5, 5)
        moves.append([old_x + dx, old_y + dy])
        
    # 2) Align with 3-point lines (extend them to make 4-point lines)
    for key, idx_set in list(line_points.items()):
        if len(idx_set) == 3 and idx not in idx_set:
            A, B, C = key
            # Find integer points on this line
            if B != 0:
                for t in range(-10, 11):
                    num = -C - A * t
                    if num % B == 0:
                        y_val = num // B
                        if abs(t) <= 1000 and abs(y_val) <= 1000:
                            cand = [t, y_val]
                            is_distinct = True
                            for i, p in enumerate(points):
                                if i != idx and p[0] == t and p[1] == y_val:
                                    is_distinct = False
                                    break
                            if is_distinct:
                                moves.append(cand)
            elif A != 0:
                for t in range(-10, 11):
                    num = -C - B * t
                    if num % A == 0:
                        x_val = num // A
                        if abs(x_val) <= 1000 and abs(t) <= 1000:
                            cand = [x_val, t]
                            is_distinct = True
                            for i, p in enumerate(points):
                                if i != idx and p[0] == x_val and p[1] == t:
                                    is_distinct = False
                                    break
                            if is_distinct:
                                moves.append(cand)
            if len(moves) > 30:
                break
        if len(moves) > 30:
            break
            
    # 3) Medium jumps
    for _ in range(5):
        dx = rng.randint(-15, 15)
        dy = rng.randint(-15, 15)
        moves.append([old_x + dx, old_y + dy])
        
    # 4) Random jumps
    for _ in range(3):
        moves.append([rng.randint(-50, 50), rng.randint(-50, 50)])
        
    return moves

def _anneal_from_start(points, line_points, start_score, deadline, rng, initial_temp=2.0):
    current_score = start_score
    best_score = start_score
    best_points = [p[:] for p in points]
    
    temp = initial_temp
    cool_rate = 0.995
    no_improve_count = 0
    
    while time.time() < deadline and temp > 0.01:
        idx = rng.randint(0, 15)
        old_pt = [points[idx][0], points[idx][1]]
        
        candidates = _generate_informed_moves(points, line_points, idx, rng)
        
        if not candidates:
            temp *= cool_rate
            continue
        
        new_pt = candidates[rng.randint(0, len(candidates) - 1)]
        
        if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
            temp *= cool_rate
            continue
        
        is_distinct = True
        for i, p in enumerate(points):
            if i != idx and p[0] == new_pt[0] and p[1] == new_pt[1]:
                is_distinct = False
                break
        if not is_distinct:
            temp *= cool_rate
            continue
        
        points[idx] = new_pt
        new_line_points = _incremental_update(points, line_points, idx, old_pt, new_pt)
        is_valid, new_score = _evaluate_from_index(points, new_line_points)
        
        if is_valid and new_score >= current_score:
            accept = True
        elif is_valid and new_score < current_score:
            delta = current_score - new_score
            prob = 2.718281828 ** (-delta / max(temp, 0.001))
            accept = rng.random() < prob
        else:
            accept = temp > 5.0 and rng.random() < 0.02
            
        if accept:
            current_score = new_score
            line_points = new_line_points
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            points[idx] = old_pt
            line_points = _build_line_index(points)
            is_valid, current_score = _evaluate_from_index(points, line_points)
            no_improve_count += 1
        
        # Restart if stuck for too long
        if no_improve_count > 200:
            temp = max(temp, 1.5)
            no_improve_count = 0
            
        temp *= cool_rate
        
    return best_points, best_score

def _affine_seed(rng):
    """Generate a 4x4 grid affinely transformed."""
    base = [[x, y] for y in range(4) for x in range(4)]
    
    for _ in range(100):
        a = rng.randint(-3, 3)
        b = rng.randint(-3, 3)
        c = rng.randint(-3, 3)
        d = rng.randint(-3, 3)
        e = rng.randint(-20, 20)
        f = rng.randint(-20, 20)
        
        det = a * d - b * c
        if det == 0:
            continue
            
        pts = []
        valid = True
        for x, y in base:
            nx = a * x + b * y + e
            ny = c * x + d * y + f
            if abs(nx) > 900 or abs(ny) > 900:
                valid = False
                break
            pts.append([nx, ny])
            
        if not valid:
            continue
            
        if len(set(tuple(p) for p in pts)) != 16:
            continue
            
        is_valid, score = _evaluate_vectorized(pts)
        if is_valid and score > 0:
            return pts, score
            
    return None, 0

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    rng = random.Random(42)
    
    best_points = None
    best_score = 0
    
    # Try 4x4 grid first
    points = [[x, y] for x in range(4) for y in range(4)]
    line_points = _build_line_index(points)
    is_valid, current_score = _evaluate_from_index(points, line_points)
    if is_valid:
        best_points = [p[:] for p in points]
        best_score = current_score
        
    # Try affine seeds
    for restart in range(15):
        if time.time() >= deadline:
            break
            
        pts, score = _affine_seed(rng)
        if pts is not None and score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            
    # Try random starts and anneal
    for restart in range(25):
        if time.time() >= deadline:
            break
            
        # Random points in a box
        pts = []
        seen = set()
        while len(pts) < 16:
            p = [rng.randint(-40, 40), rng.randint(-40, 40)]
            if tuple(p) not in seen:
                seen.add(tuple(p))
                pts.append(p)
                
        pline_points = _build_line_index(pts)
        is_valid, p_score = _evaluate_from_index(pts, pline_points)
        
        if not is_valid:
            continue
            
        phase_deadline = min(deadline, time.time() + 4)
        new_best_points, new_best_score = _anneal_from_start(
            pts, pline_points, p_score, phase_deadline, rng, initial_temp=2.0
        )
        
        if new_best_score > best_score:
            best_score = new_best_score
            best_points = [p[:] for p in new_best_points]
            
    # Final annealing from best found
    if best_points is not None and time.time() < deadline - 5:
        final_line_points = _build_line_index(best_points)
        is_valid, final_score = _evaluate_from_index(best_points, final_line_points)
        if is_valid:
            final_best_points, final_best_score = _anneal_from_start(
                best_points, final_line_points, final_score, deadline, rng, initial_temp=1.5
            )
            if final_best_score > best_score:
                best_score = final_best_score
                best_points = [p[:] for p in final_best_points]
                
    # If we haven't found anything, return the 4x4 grid
    if best_points is None:
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points

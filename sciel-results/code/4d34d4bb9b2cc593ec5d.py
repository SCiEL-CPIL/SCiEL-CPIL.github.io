"""Bootstrap spawn: model-and-optimize for 16 integer points, maximize exact-4 lines."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from fractions import Fraction

def _line_key(p1, p2):
    """Compute normalized line key (a, b, c) for line through p1, p2: a*x + b*y = c."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n, dy_n = dx // g, dy // g
    # Normalize direction to be unique
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * p1[0] + b * p1[1]
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _score_lines(points):
    """
    Returns (num_exact_4_lines, has_5_collinear).
    points: list of 16 [x, y]
    """
    n = 16
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    exact_4 = 0
    has_5 = False
    for pts in line_pts.values():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def _get_candidates_on_line(line_key, existing_indices, points):
    """
    Given a line key (a, b, c) and the indices of points already on it,
    find other integer points on this line within [-1000, 1000] that are not in the set.
    """
    a, b, c = line_key
    candidates = []
    
    # Direction vector for integer steps on the line
    # Line: a*x + b*y = c
    # Step: (b, -a) is a direction. Since gcd(a,b)=1, this generates all integer points on the line.
    if not existing_indices:
        return []
    
    ref_idx = list(existing_indices)[0]
    x_ref, y_ref = points[ref_idx]
    
    dx_step = b
    dy_step = -a
    
    k_min = 10**9
    k_max = -10**9
    
    if dx_step > 0:
        k_min = max(k_min, math.ceil((-1000 - x_ref) / dx_step))
        k_max = min(k_max, math.floor((1000 - x_ref) / dx_step))
    elif dx_step < 0:
        k_min = max(k_min, math.ceil((1000 - x_ref) / dx_step))
        k_max = min(k_max, math.floor((-1000 - x_ref) / dx_step))
    else:
        if not (-1000 <= x_ref <= 1000):
            return []
    
    if dy_step > 0:
        k_min = max(k_min, math.ceil((-1000 - y_ref) / dy_step))
        k_max = min(k_max, math.floor((1000 - y_ref) / dy_step))
    elif dy_step < 0:
        k_min = max(k_min, math.ceil((1000 - y_ref) / dy_step))
        k_max = min(k_max, math.floor((-1000 - y_ref) / dy_step))
    else:
        if not (-1000 <= y_ref <= 1000):
            return []
    
    if k_min > k_max:
        return []
    
    existing_set = set(tuple(points[i]) for i in range(16))
    
    for k in range(k_min, k_max + 1):
        x = x_ref + k * dx_step
        y = y_ref + k * dy_step
        pt = (x, y)
        if pt not in existing_set:
            candidates.append(pt)
            
    return candidates

def _local_search(points, deadline, rng):
    """
    Simulated Annealing with targeted line-completion moves and swaps.
    """
    current = [p[:] for p in points]
    current_score, h5 = _score_lines(current)
    if h5:
        return current, 0
        
    best = [p[:] for p in current]
    best_score = current_score
    
    T = 2.0
    T_min = 0.1
    cooling = 0.99
    
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Generate move
        move_type = rng.random()
        
        if move_type < 0.6:
            # Targeted: Complete a 3-point line
            line_pts = {}
            for i in range(16):
                for j in range(i + 1, 16):
                    key = _line_key(current[i], current[j])
                    if key is None: continue
                    if key not in line_pts:
                        line_pts[key] = set()
                    line_pts[key].add(i)
                    line_pts[key].add(j)
            
            three_lines = [k for k, v in line_pts.items() if len(v) == 3]
            
            if three_lines:
                target_line = rng.choice(three_lines)
                pts_on_line = line_pts[target_line]
                candidates = _get_candidates_on_line(target_line, pts_on_line, current)
                
                if candidates:
                    new_pt = rng.choice(candidates)
                    replace_candidates = [i for i in range(16) if i not in pts_on_line]
                    if replace_candidates:
                        idx = rng.choice(replace_candidates)
                        old_pt = current[idx][:]
                        current[idx] = list(new_pt)
                        
                        new_score, new_h5 = _score_lines(current)
                        
                        if not new_h5:
                            delta = new_score - current_score
                            if delta > 0 or rng.random() < math.exp(delta / T):
                                current_score = new_score
                                if current_score > best_score:
                                    best_score = current_score
                                    best = [p[:] for p in current]
                                    no_improve = 0
                                else:
                                    no_improve += 1
                            else:
                                current[idx] = old_pt
                                no_improve += 1
                        else:
                            current[idx] = old_pt
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        elif move_type < 0.8:
            # Swap move
            i, j = rng.sample(range(16), 2)
            old_i, old_j = current[i][:], current[j][:]
            current[i], current[j] = old_j, old_i
            
            new_score, new_h5 = _score_lines(current)
            
            if not new_h5:
                delta = new_score - current_score
                if delta > 0 or rng.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = [p[:] for p in current]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    current[i], current[j] = old_i, old_j
                    no_improve += 1
            else:
                current[i], current[j] = old_i, old_j
                no_improve += 1
                
        else:
            # Random move: Perturb a random point
            idx = rng.randint(0, 15)
            old_pt = current[idx][:]
            step = rng.randint(1, 20)
            new_x = old_pt[0] + rng.randint(-step, step)
            new_y = old_pt[1] + rng.randint(-step, step)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            if any([new_x, new_y] == p for p in current):
                continue
                
            current[idx] = [new_x, new_y]
            new_score, new_h5 = _score_lines(current)
            
            if not new_h5:
                delta = new_score - current_score
                if delta > 0 or rng.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = [p[:] for p in current]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    current[idx] = old_pt
                    no_improve += 1
            else:
                current[idx] = old_pt
                no_improve += 1
        
        T = max(T_min, T * cooling)
        
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    rng = random.Random(42)
    
    # Base: 4x4 grid
    base = [[x, y] for x in range(4) for y in range(4)]
    base_score, _ = _score_lines(base)
    best_points = [p[:] for p in base]
    best_score = base_score
    
    # Multiple restarts
    restarts = 10
    for r in range(restarts):
        if time.time() > deadline - 20:
            break
            
        if r == 0:
            current = [p[:] for p in base]
        else:
            # Perturb base significantly to escape local optima
            current = [p[:] for p in base]
            for _ in range(rng.randint(10, 20)):
                idx = rng.randint(0, 15)
                new_x = current[idx][0] + rng.randint(-30, 30)
                new_y = current[idx][1] + rng.randint(-30, 30)
                new_pt = [new_x, new_y]
                if any(new_pt == p for p in current):
                    continue
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                current[idx] = new_pt
        
        s, h5 = _score_lines(current)
        if h5:
            continue
            
        opt_pts, opt_score = _local_search(current, deadline, rng)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    # Final verification
    final_score, final_h5 = _score_lines(best_points)
    if final_h5:
        # Fallback to base
        best_points = [p[:] for p in base]
        
    return best_points
# EVOLVE-BLOCK-END

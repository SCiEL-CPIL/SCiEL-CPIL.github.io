import time
import random
import math
from itertools import combinations

def _make_line_key(x1, y1, x2, y2):
    """Compute normalized line key from two points."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    # Line equation: a*x + b*y + c = 0
    a, b = dy_n, -dx_n
    c = -(a * x1 + b * y1)
    g2 = math.gcd(abs(a), abs(b), abs(c))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _score_points(points):
    """
    Returns (num_exact_4_lines, has_5_collinear) for 16 points.
    Uses a dictionary of line keys to point count.
    """
    n = 16
    line_counts = {}
    # Precompute all pairs
    pairs = list(combinations(range(n), 2))
    for i, j in pairs:
        p1 = points[i]
        p2 = points[j]
        key = _make_line_key(p1[0], p1[1], p2[0], p2[1])
        if key is None:
            continue
        if key not in line_counts:
            line_counts[key] = 0
        line_counts[key] += 1
    
    # Wait, counting pairs is not enough to know if 4 distinct points are on the line.
    # If 4 points are collinear, there are C(4,2)=6 pairs.
    # If 5 points are collinear, there are C(5,2)=10 pairs.
    # We need to map line keys to the set of points on that line to be sure.
    
    # Let's use sets for accuracy, it's fast enough for 16 points.
    line_pts = {}
    for i, j in pairs:
        p1 = points[i]
        p2 = points[j]
        key = _make_line_key(p1[0], p1[1], p2[0], p2[1])
        if key is None:
            continue
        if key not in line_pts:
            line_pts[key] = set()
        line_pts[key].add(i)
        line_pts[key].add(j)
        
    exact_4 = 0
    has_5 = False
    for key, pts in line_pts.items():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def _get_line_info(points):
    """
    Returns a dict: line_key -> set of point indices.
    Also returns list of keys with exactly 3 points and list with >=5.
    """
    n = 16
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            key = _make_line_key(p1[0], p1[1], p2[0], p2[1])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
            
    lines_3 = []
    lines_5_plus = []
    for key, pts in line_pts.items():
        cnt = len(pts)
        if cnt == 3:
            lines_3.append((key, pts))
        elif cnt >= 5:
            lines_5_plus.append((key, pts))
            
    return line_pts, lines_3, lines_5_plus

def _find_completion_point(pts_on_line, key, points, rng):
    """
    Given 3 points on a line and the line key (a,b,c), find a 4th integer point
    on the line that is not already in the set.
    Returns [x, y] or None.
    """
    a, b, c = key
    # Equation: a*x + b*y + c = 0
    # We need integer solutions.
    # If b != 0, y = (-a*x - c) / b.
    # If b == 0, a*x + c = 0 => x = -c/a (if a != 0).
    
    existing_pts = set(tuple(points[i]) for i in pts_on_line)
    
    # Get the x-coordinates of existing points to guide search
    xs = [points[i][0] for i in pts_on_line]
    ys = [points[i][1] for i in pts_on_line]
    
    candidates = []
    
    # Strategy: Look near the existing points
    # Range of x to check
    min_x = min(xs) - 10
    max_x = max(xs) + 10
    
    if b != 0:
        for x in range(min_x, max_x + 1):
            if abs(x) > 1000:
                continue
            num = -a * x - c
            if num % b == 0:
                y = num // b
                if abs(y) <= 1000:
                    pt = (x, y)
                    if pt not in existing_pts:
                        candidates.append(pt)
    else:
        # b == 0, so a*x + c = 0
        if a != 0:
            if c % a == 0:
                x = -c // a
                if abs(x) <= 1000:
                    # y can be anything? No, the line is vertical x = const.
                    # Wait, if b=0, a*x+c=0. This is a vertical line x = -c/a.
                    # Any y is on the line.
                    # We need to pick a y that is not in existing_pts.
                    # Existing points have specific y's.
                    # Let's pick a y near the existing y's.
                    for y in range(min(ys) - 10, max(ys) + 10):
                        if abs(y) > 1000:
                            continue
                        pt = (x, y)
                        if pt not in existing_pts:
                            candidates.append(pt)
    
    if candidates:
        return list(rng.choice(candidates))
    return None

def construct():
    deadline = time.time() + 145
    rng = random.Random(42)
    
    # Initial configurations
    # 1. 4x4 Grid
    base_grid = [[x, y] for x in range(4) for y in range(4)]
    
    # 2. Affine transform of grid (slanted)
    base_slant = [[x + y, x] for x in range(4) for y in range(4)]
    
    # 3. Random cluster
    base_random = [[rng.randint(-10, 10), rng.randint(-10, 10)] for _ in range(16)]
    # Ensure distinct
    unique_base = []
    for p in base_random:
        if p not in unique_base:
            unique_base.append(p)
    while len(unique_base) < 16:
        p = [rng.randint(-10, 10), rng.randint(-10, 10)]
        if p not in unique_base:
            unique_base.append(p)
    base_random = unique_base

    starts = [base_grid, base_slant, base_random]
    
    best_score = -1
    best_points = None
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        if restart < len(starts):
            current = [p[:] for p in starts[restart]]
        else:
            # Perturb best found so far
            if best_points:
                current = [p[:] for p in best_points]
                # Randomly perturb a few points
                for _ in range(rng.randint(1, 4)):
                    idx = rng.randint(0, 15)
                    dx = rng.randint(-5, 5)
                    dy = rng.randint(-5, 5)
                    current[idx] = [current[idx][0] + dx, current[idx][1] + dy]
                    # Check bounds and distinctness roughly
                    if abs(current[idx][0]) > 1000 or abs(current[idx][1]) > 1000:
                        current[idx] = [p for p in best_points if p != current[idx]][0] # Fallback
            else:
                current = [p[:] for p in base_grid]
        
        # Ensure distinct
        for i in range(16):
            for j in range(i+1, 16):
                if current[i] == current[j]:
                    current[j] = [current[j][0] + 1, current[j][1]]
        
        current_score, h5 = _score_points(current)
        if h5:
             # If we start with 5-collinear, try to fix immediately
             # For now, just accept and let SA fix it, or skip
             pass
             
        T = 1.0
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            move_type = rng.random()
            
            if move_type < 0.3:
                # Strategy 1: Complete a 3-line
                _, lines_3, lines_5 = _get_line_info(current)
                if lines_3:
                    key, pts = rng.choice(lines_3)
                    new_pt = _find_completion_point(pts, key, current, rng)
                    if new_pt:
                        # Move a random point NOT in pts to new_pt
                        candidates_idx = [i for i in range(16) if i not in pts]
                        if candidates_idx:
                            idx = rng.choice(candidates_idx)
                            old = current[idx][:]
                            current[idx] = new_pt
                            s_new, h5_new = _score_points(current)
                            
                            if h5_new:
                                # Reject if creates 5-collinear? 
                                # Or accept with penalty?
                                # Let's reject 5-collinear strictly to keep search simple
                                current[idx] = old
                                no_improve += 1
                            else:
                                if s_new >= current_score or (T > 0.1 and rng.random() < math.exp((s_new - current_score) / T)):
                                    if s_new > current_score:
                                        no_improve = 0
                                        if s_new > best_score:
                                            best_score = s_new
                                            best_points = [p[:] for p in current]
                                    else:
                                        no_improve += 1
                                    current_score = s_new
                                else:
                                    current[idx] = old
                                    no_improve += 1
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            elif move_type < 0.5:
                # Strategy 2: Break 5-collinear
                _, _, lines_5 = _get_line_info(current)
                if lines_5:
                    key, pts = rng.choice(lines_5)
                    idx = rng.choice(list(pts))
                    old = current[idx][:]
                    # Move away randomly
                    for _ in range(10):
                        new_x = old[0] + rng.randint(-5, 5)
                        new_y = old[1] + rng.randint(-5, 5)
                        if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                            if not any([new_x, new_y] == p for p in current):
                                current[idx] = [new_x, new_y]
                                s_new, h5_new = _score_points(current)
                                if not h5_new:
                                    if s_new >= current_score or (T > 0.1 and rng.random() < math.exp((s_new - current_score) / T)):
                                        if s_new > current_score:
                                            no_improve = 0
                                            if s_new > best_score:
                                                best_score = s_new
                                                best_points = [p[:] for p in current]
                                        else:
                                            no_improve += 1
                                        current_score = s_new
                                    else:
                                        current[idx] = old
                                        no_improve += 1
                                else:
                                    current[idx] = old
                                    no_improve += 1
                                break
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            elif move_type < 0.8:
                # Strategy 3: Swap two points
                i, j = rng.sample(range(16), 2)
                old_i, old_j = current[i][:], current[j][:]
                current[i], current[j] = old_j, old_i
                s_new, h5_new = _score_points(current)
                
                if h5_new:
                    current[i], current[j] = old_i, old_j
                    no_improve += 1
                else:
                    if s_new >= current_score or (T > 0.1 and rng.random() < math.exp((s_new - current_score) / T)):
                        if s_new > current_score:
                            no_improve = 0
                            if s_new > best_score:
                                best_score = s_new
                                best_points = [p[:] for p in current]
                        else:
                            no_improve += 1
                        current_score = s_new
                    else:
                        current[i], current[j] = old_i, old_j
                        no_improve += 1
                        
            else:
                # Strategy 4: Random Perturb
                idx = rng.randint(0, 15)
                old = current[idx][:]
                new_x = old[0] + rng.randint(-5, 5)
                new_y = old[1] + rng.randint(-5, 5)
                if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                    if not any([new_x, new_y] == p for p in current):
                        current[idx] = [new_x, new_y]
                        s_new, h5_new = _score_points(current)
                        
                        if h5_new:
                            current[idx] = old
                            no_improve += 1
                        else:
                            if s_new >= current_score or (T > 0.1 and rng.random() < math.exp((s_new - current_score) / T)):
                                if s_new > current_score:
                                    no_improve = 0
                                    if s_new > best_score:
                                        best_score = s_new
                                        best_points = [p[:] for p in current]
                                else:
                                    no_improve += 1
                                current_score = s_new
                            else:
                                current[idx] = old
                                no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            T = max(0.05, T * 0.98)
            
    # Final verification
    if best_points is None:
        best_points = base_grid
        best_score, _ = _score_points(best_points)
        
    return best_points

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            g = _gcd3(A, B, C)
            if g > 1:
                A //= g
                B //= g
                C //= g
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _gcd3(a, b, c):
    from math import gcd
    g = gcd(abs(a), abs(b))
    g = gcd(g, abs(c))
    return g

def _all_collinear(points):
    if len(points) < 3:
        return False
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if cross != 0:
        return False
    for i in range(3, len(points)):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) - (y1 - y0) * (xi - x0) != 0:
            return False
    return True

def _get_ordinary_lines(points):
    """Return list of (i, j) indices for ordinary lines."""
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            g = _gcd3(A, B, C)
            if g > 1:
                A //= g
                B //= g
                C //= g
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = []
    for key, pts in line_points.items():
        if len(pts) == 2:
            i, j = tuple(pts)
            ordinary.append((i, j))
    return ordinary

def _find_collinear_candidates(p1, p2, points, max_dist=100):
    """Find integer points C such that p1, p2, C are collinear and C is not in points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return []
    dx //= g
    dy //= g
    
    candidates = []
    # Check in both directions
    for sign in [1, -1]:
        for k in range(1, max_dist + 1):
            cx = x2 + sign * k * dx
            cy = y2 + sign * k * dy
            if abs(cx) > 10**9 or abs(cy) > 10**9:
                break
            # Check if already in points
            if [cx, cy] in points:
                continue
            candidates.append([cx, cy])
    return candidates

def _targeted_sa(points, time_limit):
    current = [p[:] for p in points]
    current_score = _count_ordinary_lines(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    n = len(current)
    T = 5.0
    T_end = 0.01
    start_time = time.time()
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit:
        if T < T_end:
            break
            
        ordinary_lines = _get_ordinary_lines(current)
        
        if not ordinary_lines:
            break
            
        # Pick a random ordinary line
        i, j = random.choice(ordinary_lines)
        
        # Strategy 1: Move i or j to complete the line
        # Strategy 2: Move a random point to lie on an existing line of 2 points (making it 3)
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Move point i to a candidate on line (i, j)
            p_i = current[i]
            p_j = current[j]
            candidates = _find_collinear_candidates(p_i, p_j, current, max_dist=50)
            if candidates:
                new_pos = random.choice(candidates)
                orig = current[i][:]
                current[i] = new_pos
                new_score = _count_ordinary_lines(current)
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best = [p[:] for p in current]
                else:
                    current[i] = orig
            else:
                # No candidates, try random move
                orig = current[i][:]
                new_x = orig[0] + random.randint(-20, 20)
                new_y = orig[1] + random.randint(-20, 20)
                if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    pos = (new_x, new_y)
                    dup = False
                    for k in range(n):
                        if k != i and current[k][0] == new_x and current[k][1] == new_y:
                            dup = True
                            break
                    if not dup:
                        current[i] = [new_x, new_y]
                        new_score = _count_ordinary_lines(current)
                        delta = new_score - current_score
                        if delta < 0 or random.random() < math.exp(-delta / T):
                            current_score = new_score
                            if new_score < best_score:
                                best_score = new_score
                                best = [p[:] for p in current]
                        else:
                            current[i] = orig
        else:
            # Move a random point to lie on a line formed by two other points
            k = random.randint(0, n - 1)
            orig = current[k][:]
            
            # Find a pair (l, m) such that moving k to line (l, m) creates a 3-point line
            # We just pick a random pair and try to project k onto it? No, we need integer points.
            # Instead, pick a random pair (l, m) and try to move k to a point on line (l, m)
            l = random.randint(0, n - 1)
            m = random.randint(0, n - 1)
            if l == k or m == k or l == m:
                continue
                
            p_l = current[l]
            p_m = current[m]
            candidates = _find_collinear_candidates(p_l, p_m, current, max_dist=50)
            if candidates:
                new_pos = random.choice(candidates)
                # Ensure new_pos is not same as orig
                if new_pos != orig:
                    current[k] = new_pos
                    new_score = _count_ordinary_lines(current)
                    delta = new_score - current_score
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best = [p[:] for p in current]
                    else:
                        current[k] = orig
            else:
                # Random move
                new_x = orig[0] + random.randint(-20, 20)
                new_y = orig[1] + random.randint(-20, 20)
                if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    pos = (new_x, new_y)
                    dup = False
                    for t in range(n):
                        if t != k and current[t][0] == new_x and current[t][1] == new_y:
                            dup = True
                            break
                    if not dup:
                        current[k] = [new_x, new_y]
                        new_score = _count_ordinary_lines(current)
                        delta = new_score - current_score
                        if delta < 0 or random.random() < math.exp(-delta / T):
                            current_score = new_score
                            if new_score < best_score:
                                best_score = new_score
                                best = [p[:] for p in current]
                        else:
                            current[k] = orig
        
        T *= cooling_rate
        
    return best, best_score

def _generate_star():
    pts = []
    for i in range(1, 4):
        pts.append([i, 0])
        pts.append([-i, 0])
        pts.append([0, i])
        pts.append([0, -i])
    return pts

def construct():
    deadline = time.time() + 145
    best_points = None
    best_score = float('inf')
    
    configs = []
    configs.append(_generate_star())
    
    # Add grid
    grid = []
    for i in range(3):
        for j in range(4):
            grid.append([j, i])
    configs.append(grid)
    
    random.seed(42)
    for config in configs:
        if time.time() > deadline:
            break
        seen = set()
        valid = True
        for p in config:
            tp = tuple(p)
            if tp in seen or abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
                valid = False
                break
            seen.add(tp)
        if not valid or len(seen) != 12:
            continue
        if _all_collinear(config):
            continue
            
        time_left = deadline - time.time()
        if time_left < 5:
            continue
            
        pts, score = _targeted_sa(config, time_limit=time_left * 0.9)
        if score < best_score:
            best_score = score
            best_points = pts
            
        if best_score <= 6:
            break
            
    if best_points is None:
        return _generate_star()
        
    return best_points
# EVOLVE-BLOCK-END

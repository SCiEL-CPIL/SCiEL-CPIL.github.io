"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from math import gcd

def _gcd3(a, b, c):
    g = gcd(abs(a), abs(b))
    g = gcd(g, abs(c))
    return g

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
            # Normalize sign
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

def _is_valid(points):
    """Check that points are distinct, within bounds, and not all collinear."""
    n = len(points)
    if n != 12:
        return False
    seen = set()
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        tp = tuple(p)
        if tp in seen:
            return False
        seen.add(tp)
    if _all_collinear(points):
        return False
    return True

def _generate_candidates():
    """Generate a diverse set of initial configurations."""
    candidates = []
    
    # 1. Grids
    for rows, cols in [(3, 4), (4, 3), (2, 6), (6, 2)]:
        pts = [[x, y] for x in range(cols) for y in range(rows)]
        candidates.append(pts)
        
    # 2. Points on a circle (approximate integer coordinates)
    # Use small integer coordinates on a circle of radius 5
    circle_pts = [[3, 4], [-3, 4], [3, -4], [-3, -4], [5, 0], [-5, 0], [0, 5], [0, -5], [1, 1], [-1, 1], [1, -1], [-1, -1]]
    # Scale to avoid coincidences if needed, but these are distinct
    candidates.append(circle_pts)
    
    # 3. Points on a parabola y = x^2
    pts = [[x, x*x] for x in range(12)]
    candidates.append(pts)
    
    # 4. Perturbed grids
    random.seed(42)
    for _ in range(10):
        rows, cols = random.choice([(3, 4), (4, 3)])
        base = [[x, y] for x in range(cols) for y in range(rows)]
        pts = [p[:] for p in base]
        for p in pts:
            p[0] += random.randint(-5, 5)
            p[1] += random.randint(-5, 5)
        if _is_valid(pts):
            candidates.append(pts)
            
    # 5. Random points in a box
    for _ in range(5):
        pts = []
        for _ in range(12):
            pts.append([random.randint(-20, 20), random.randint(-20, 20)])
        if _is_valid(pts):
            candidates.append(pts)
            
    return candidates

def _simulated_annealing(points, deadline):
    """Simulated annealing to minimize ordinary lines."""
    current = [p[:] for p in points]
    current_score = _count_ordinary_lines(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    n = len(current)
    T = 50.0  # Start with higher temperature
    T_min = 0.1
    cooling = 0.998
    
    while T > T_min and time.time() < deadline:
        i = random.randint(0, n - 1)
        orig = current[i][:]
        
        # Step size: large enough to jump, small enough to refine
        # Scale step with temperature
        step = int(T * 0.5) + 1
        dx = random.randint(-step, step)
        dy = random.randint(-step, step)
        
        new_pos = [orig[0] + dx, orig[1] + dy]
        
        if abs(new_pos[0]) > 10**9 or abs(new_pos[1]) > 10**9:
            continue
            
        # Check distinctness
        dup = False
        for j in range(n):
            if j != i and current[j][0] == new_pos[0] and current[j][1] == new_pos[1]:
                dup = True
                break
        if dup:
            continue
            
        current[i] = new_pos
        new_score = _count_ordinary_lines(current)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = [p[:] for p in current]
                if best_score <= 6:
                    return best, best_score
        else:
            current[i] = orig
            
        T *= cooling
        
    return best, best_score

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    best_points = None
    best_score = float('inf')
    
    random.seed(42)
    
    candidates = _generate_candidates()
    
    # Sort candidates by initial score to prioritize good starts?
    # Or just iterate. Sorting might waste time if many are bad.
    # Let's just iterate, but maybe shuffle to ensure variety in time budget distribution
    random.shuffle(candidates)
    
    for config in candidates:
        if time.time() > deadline:
            break
        if not _is_valid(config):
            continue
        
        # Allocate time to each candidate
        # If we have 20 candidates, give each ~5s?
        # Better: run SA for a fixed short time per candidate, or until global deadline
        # Let's run SA with a sub-deadline
        sub_deadline = min(deadline, time.time() + 5.0)
        
        pts, score = _simulated_annealing(config, sub_deadline)
        
        if score < best_score:
            best_score = score
            best_points = pts
            if best_score <= 6:
                break
                
    if best_points is None:
        # Fallback: return a valid configuration
        return [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
# EVOLVE-BLOCK-END

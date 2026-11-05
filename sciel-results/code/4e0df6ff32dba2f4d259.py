import time
import random
from math import gcd
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)

def _collinear_check(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True

def _valid(points):
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)

def _nearest_integer_on_line(p, q, r):
    """
    Given points p, q, r, if p, q, r are not collinear, find the nearest integer point 
    to p that lies on the line defined by q and r.
    If q and r are the same, return None.
    """
    x1, y1 = q
    x2, y2 = r
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return None
        
    # Line equation: A*x + B*y + C = 0
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    
    # We want to find integer (x, y) close to p=(px, py) such that A*x + B*y + C = 0.
    # This is a linear Diophantine equation.
    # However, we just need *a* point. We can parameterize the line.
    # Point on line: (x1 + t*dx, y1 + t*dy)
    # We want to minimize distance to p.
    # (x1 + t*dx - px)^2 + (y1 + t*dy - py)^2
    # Let v = (dx, dy), w = (x1-px, y1-py)
    # Minimize |v*t + w|^2
    # t = - (v.w) / |v|^2
    vx, vy = dx, dy
    wx, wy = x1 - p[0], y1 - p[1]
    dot = vx * wx + vy * wy
    norm_sq = vx * vx + vy * vy
    
    if norm_sq == 0:
        return None
        
    t_real = -dot / norm_sq
    
    # Try nearest integer values for t? No, t is a parameter for the line, not necessarily integer steps.
    # The points on the line with integer coordinates are of the form (x1 + k*gx, y1 + k*gy) where g = gcd(dx, dy).
    # Let's find the integer k closest to the real t * g?
    # Actually, the set of integer points on the line is an arithmetic progression.
    # Let g = gcd(abs(dx), abs(dy))
    # Direction vector for integer steps: (dx//g, dy//g)
    # The integer points are (x1 + m*(dx//g), y1 + m*(dy//g)) for integer m.
    
    g = gcd(abs(dx), abs(dy))
    if g == 0: return None
    
    sx = dx // g
    sy = dy // g
    
    # We need m such that (x1 + m*sx, y1 + m*sy) is closest to p.
    # Minimize (x1 + m*sx - px)^2 + (y1 + m*sy - py)^2
    # Derivative wrt m: 2*sx*(x1 + m*sx - px) + 2*sy*(y1 + m*sy - py) = 0
    # m*(sx^2 + sy^2) = px*sx - x1*sx + py*sy - y1*sy
    # m = ( (px-x1)*sx + (py-y1)*sy ) / (sx^2 + sy^2)
    
    num = (p[0] - x1) * sx + (p[1] - y1) * sy
    den = sx * sx + sy * sy
    
    if den == 0:
        return None
        
    m_real = num / den
    
    # Check m_real rounded to nearest integers
    candidates = []
    m_floor = int(m_real)
    for m in [m_floor, m_floor + 1, m_floor - 1]:
        nx = x1 + m * sx
        ny = y1 + m * sy
        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
            candidates.append((nx, ny))
            
    if not candidates:
        return None
        
    # Return the one closest to p
    best_pt = min(candidates, key=lambda pt: (pt[0]-p[0])**2 + (pt[1]-p[1])**2)
    return best_pt

def _generate_initial_configs():
    configs = []
    
    # 4x3 Grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3x4 Grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2x6 Grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 6x2 Grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # Triangle configuration: 3 lines, 4 points each.
    # Lines: y=0, x=0, x+y=4 (adjusted to fit integers)
    # Intersections: (0,0), (4,0), (0,4)
    # Points on y=0: (0,0), (1,0), (2,0), (4,0) -> 4 pts
    # Points on x=0: (0,0), (0,1), (0,2), (0,4) -> 4 pts
    # Points on x+y=4: (4,0), (0,4), (1,3), (2,2) -> 4 pts
    # Union: (0,0), (1,0), (2,0), (4,0), (0,1), (0,2), (0,4), (1,3), (2,2)
    # That's 9 points. We need 12. Add 3 more.
    # Add (3,1), (1,2), (3,0)? 
    # Let's just use a 3x4 grid as a proxy for now, it's robust.
    
    # Random points
    rng = random.Random(42)
    for _ in range(5):
        pts = []
        while len(pts) < 12:
            x = rng.randint(-10, 10)
            y = rng.randint(-10, 10)
            if (x, y) not in pts:
                pts.append([x, y])
        configs.append(pts)
        
    # "Hesse" like configuration (simplified)
    # Points on 3 lines forming a triangle, with extra points.
    # Let's try to construct a specific one known to have low ordinary lines.
    # One such configuration for 12 points:
    # 4 points on L1, 4 on L2, 4 on L3.
    # If L1, L2, L3 are concurrent, we have 1 intersection.
    # Points: 1 shared + 3 on L1 + 3 on L2 + 3 on L3 = 10 points.
    # Add 2 more.
    
    return configs

def _local_search(points, max_time=10.0):
    """
    Local search with specific moves:
    1. Hill climb with small perturbations.
    2. Line alignment moves.
    """
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    best = [p[:] for p in cur]
    best_score = cur_score
    
    start_time = time.time()
    rng = random.Random(123)
    
    while time.time() - start_time < max_time:
        improved = False
        
        # 1. Try small perturbations (Hill Climb)
        for idx in range(12):
            if time.time() - start_time > max_time:
                break
            old_x, old_y = cur[idx]
            # Try neighbors
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = old_x + dx, old_y + dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    new_pt = (nx, ny)
                    if new_pt in [tuple(p) for p in cur]:
                        continue
                    new_pts = [p[:] for p in cur]
                    new_pts[idx] = [nx, ny]
                    if not _valid(new_pts):
                        continue
                    new_score = _count_ordinary(new_pts)
                    if new_score < cur_score:
                        cur = new_pts
                        cur_score = new_score
                        improved = True
                        if cur_score < best_score:
                            best = [p[:] for p in cur]
                            best_score = cur_score
                        break
                if improved:
                    break
            if improved:
                break
                
        if improved:
            continue
            
        # 2. Try Line Alignment Moves
        # For each point i, and each pair j, k (j!=i, k!=i, j!=k)
        # Check if moving i to the line defined by j, k improves the score.
        # We only do this for a subset of pairs to save time.
        if time.time() - start_time > max_time:
            break
            
        # Pick a random point to move
        idx = rng.randint(0, 11)
        # Pick 2 random other points to define a line
        others = [i for i in range(12) if i != idx]
        j, k = rng.sample(others, 2)
        
        target_line_pt = _nearest_integer_on_line(cur[idx], cur[j], cur[k])
        
        if target_line_pt:
            if target_line_pt not in [tuple(p) for p in cur]:
                new_pts = [p[:] for p in cur]
                new_pts[idx] = list(target_line_pt)
                if _valid(new_pts):
                    new_score = _count_ordinary(new_pts)
                    if new_score < cur_score:
                        cur = new_pts
                        cur_score = new_score
                        improved = True
                        if cur_score < best_score:
                            best = [p[:] for p in cur]
                            best_score = cur_score
                    elif new_score == cur_score and rng.random() < 0.1:
                        # Accept neutral moves occasionally to escape plateaus
                        cur = new_pts
                        # cur_score unchanged
                        # Don't update best
                        improved = True # Treat as progress to restart perturbation search? No, just continue.
        
        if not improved:
            # 3. Random SA step if stuck
            idx = rng.randint(0, 11)
            old_x, old_y = cur[idx]
            nx = old_x + rng.randint(-10, 10)
            ny = old_y + rng.randint(-10, 10)
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                new_pt = (nx, ny)
                if new_pt not in [tuple(p) for p in cur]:
                    new_pts = [p[:] for p in cur]
                    new_pts[idx] = [nx, ny]
                    if _valid(new_pts):
                        new_score = _count_ordinary(new_pts)
                        # Simple acceptance: accept if better or random
                        if new_score < cur_score or rng.random() < 0.05:
                            cur = new_pts
                            cur_score = new_score
                            if cur_score < best_score:
                                best = [p[:] for p in cur]
                                best_score = cur_score

    return best, best_score

def construct():
    best_pts = None
    best_score = 999
    
    # Try known configurations
    configs = _generate_initial_configs()
    
    for cfg in configs:
        if not _valid(cfg):
            continue
        score = _count_ordinary(cfg)
        if score < best_score:
            best_score = score
            best_pts = [p[:] for p in cfg]
            
    if best_pts is None:
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary(best_pts)
        
    # Run local search on the best initial config
    if time.time() < _DEADLINE - 10:
        # Allocate remaining time to local search
        remaining = _DEADLINE - time.time() - 5
        ls_pts, ls_score = _local_search(best_pts, max_time=remaining)
        if ls_score < best_score:
            best_score = ls_score
            best_pts = ls_pts
            
    # If we have time, try local search on other promising configs
    for cfg in configs:
        if time.time() > _DEADLINE - 10:
            break
        if not _valid(cfg):
            continue
        score = _count_ordinary(cfg)
        if score < best_score + 2: # Only try if competitive
            remaining = _DEADLINE - time.time() - 5
            if remaining < 2:
                break
            ls_pts, ls_score = _local_search(cfg, max_time=remaining)
            if ls_score < best_score:
                best_score = ls_score
                best_pts = ls_pts
                
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))

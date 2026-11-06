"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def construct():
    deadline = time.time() + 140
    
    # The Böröczky-type construction: 
    # Take points on a conic and points on a line.
    # For 12 points: 6 on a conic, 6 on a line.
    # The conic is a circle x^2 + y^2 = R^2.
    # The line is y = 0 (x-axis).
    # Points on the circle that are also on the line: (±R, 0).
    # We need 6 points on the circle and 6 on the line, with no overlap.
    # So we need 6 points on the circle excluding (±R, 0), and 6 points on the line excluding (±R, 0).
    
    # Let's use a circle of radius R where there are many integer points.
    # R = 65: points include (±65,0), (±0,65), (±16,63), (±63,16), (±25,60), (±60,25), (±33,56), (±56,33), (±39,52), (±52,39), (±16,63)
    # Actually, let's use a known configuration.
    
    # A better approach: use the known optimal or near-optimal configurations.
    # For 12 points, the minimum number of ordinary lines is 6 (over reals).
    # A configuration achieving 6 ordinary lines: 
    # Take 4 points on a line, and 8 points forming a complete quadrilateral or similar.
    
    # Let me try a specific construction:
    # 12 points: 3x4 grid perturbed, or a configuration from the literature.
    
    # Configuration 1: 3 points on x-axis, 3 on y-axis, 6 elsewhere
    # Configuration 2: Based on Böröczky: n/2 on a conic, n/2 on a line.
    # For n=12: 6 on a circle, 6 on a line.
    
    # Circle x^2 + y^2 = 65^2 = 4225
    # Integer points: (±65,0), (±0,65), (±16,63), (±63,16), (±25,60), (±60,25), (±33,56), (±56,33), (±39,52), (±52,39)
    # That's 20 points. We need 6 of them, not including (±65,0).
    
    # Line: y = 100 (to avoid the circle points on x-axis)
    # 6 points on y = 100: (x, 100) for various x.
    
    # Let me try a more systematic approach with local search.
    
    random.seed(12345)
    
    # Initial configurations
    configs = []
    
    # Config 1: 4x3 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Config 2: 3x4 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Config 3: 6x2 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # Config 4: Böröczky-like: 6 on circle, 6 on line
    # Circle x^2 + y^2 = 65^2
    circle_pts = [
        [16, 63], [63, 16], [-16, 63], [-63, 16],
        [25, 60], [60, 25]
    ]
    line_pts = [
        [0, 100], [1, 100], [2, 100], [3, 100], [4, 100], [5, 100]
    ]
    configs.append(circle_pts + line_pts)
    
    # Config 5: Another Böröczky-like with different line
    line_pts2 = [
        [-3, 100], [-2, 100], [-1, 100], [0, 100], [1, 100], [2, 100]
    ]
    configs.append(circle_pts + line_pts2)
    
    # Config 6: Points from a conic and a line, different radius
    # Circle x^2 + y^2 = 85^2 = 7225
    # Points: (±85,0), (±0,85), (±36,77), (±77,36), (±40,75), (±75,40), (±51,68), (±68,51)
    circle_pts2 = [
        [36, 77], [77, 36], [-36, 77], [-77, 36],
        [40, 75], [75, 40]
    ]
    line_pts3 = [
        [0, 200], [1, 200], [2, 200], [3, 200], [4, 200], [5, 200]
    ]
    configs.append(circle_pts2 + line_pts3)
    
    # Find best initial config
    best_pts = None
    best_score = float('inf')
    for config in configs:
        if not _is_valid(config):
            continue
        score = _count_ordinary(config)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in config]
    
    if best_pts is None:
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary(best_pts)
    
    # Local search: Simulated Annealing
    def _sa_search(pts, time_budget):
        current_pts = [list(p) for p in pts]
        current_score = _count_ordinary(current_pts)
        best_pts_sa = [list(p) for p in current_pts]
        best_score_sa = current_score
        
        T = 10.0
        cooling = 0.99999
        min_T = 0.01
        
        deadline_sa = time.time() + time_budget
        
        while time.time() < deadline_sa:
            if T < min_T:
                T = min_T
            
            idx = random.randint(0, 11)
            orig_x, orig_y = current_pts[idx]
            
            # Generate a candidate move
            # Move to intersection of lines through other points, or random nearby point
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to place on a line through 2 other points
                others = [i for i in range(12) if i != idx]
                if len(others) >= 2:
                    i1, i2 = random.sample(others, 2)
                    x1, y1 = current_pts[i1]
                    x2, y2 = current_pts[i2]
                    # Line through (x1,y1) and (x2,y2)
                    # Try a random point on this line
                    t = random.randint(-50, 50)
                    dx = x2 - x1
                    dy = y2 - y1
                    nx = x1 + t * dx
                    ny = y1 + t * dy
                    if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                        # Check duplicate
                        dup = False
                        for j, p in enumerate(current_pts):
                            if j != idx and p[0] == nx and p[1] == ny:
                                dup = True
                                break
                        if not dup:
                            new_pts = [list(p) for p in current_pts]
                            new_pts[idx] = [nx, ny]
                            new_score = _count_ordinary(new_pts)
                            delta = new_score - current_score
                            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                                current_pts = new_pts
                                current_score = new_score
                                if current_score < best_score_sa:
                                    best_score_sa = current_score
                                    best_pts_sa = [list(p) for p in current_pts]
            else:
                # Random move
                step = random.randint(1, 20)
                nx = orig_x + random.randint(-step, step)
                ny = orig_y + random.randint(-step, step)
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    dup = False
                    for j, p in enumerate(current_pts):
                        if j != idx and p[0] == nx and p[1] == ny:
                            dup = True
                            break
                    if not dup:
                        new_pts = [list(p) for p in current_pts]
                        new_pts[idx] = [nx, ny]
                        new_score = _count_ordinary(new_pts)
                        delta = new_score - current_score
                        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                            current_pts = new_pts
                            current_score = new_score
                            if current_score < best_score_sa:
                                best_score_sa = current_score
                                best_pts_sa = [list(p) for p in current_pts]
            
            T *= cooling
        
        return best_pts_sa, best_score_sa
    
    # Run SA for remaining time
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        sa_pts, sa_score = _sa_search(best_pts, time_remaining - 5)
        if sa_score < best_score:
            best_pts = sa_pts
            best_score = sa_score
    
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END

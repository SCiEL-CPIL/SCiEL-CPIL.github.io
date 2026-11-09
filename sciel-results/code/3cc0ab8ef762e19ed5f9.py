"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
from itertools import combinations

def count_ordinary_lines(points):
    """Count the number of lines that contain exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_to_points = {}
    
    for i, j in combinations(range(n), 2):
        x1, y1 = points[i]
        x2, y2 = points[j]
        
        A = y2 - y1
        B = -(x2 - x1)
        C = x2 * y1 - y2 * x1
        
        def _gcd(a, b):
            a, b = abs(a), abs(b)
            while b:
                a, b = b, a % b
            return a
        
        g = _gcd(_gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0 or C != 0) else 1
        if g > 0:
            A //= g
            B //= g
            C //= g
        
        if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
            A = -A
            B = -B
            C = -C
        
        key = (A, B, C)
        if key not in line_to_points:
            line_to_points[key] = set()
        line_to_points[key].add(i)
        line_to_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_to_points.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def construct():
    """
    Construct 12 integer points minimizing the number of ordinary lines.
    Uses hexagonal lattice symmetry and structured local search.
    """
    
    # Start with a hexagon-based symmetric configuration
    # Use a distorted hexagon with integer coordinates
    # Center: (0,0)
    # 6 vertices of a hexagon-like shape
    # 5 additional points for symmetry
    
    # Initial configuration: 4x3 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    
    best_points = [p[:] for p in points]
    best_ordinary = count_ordinary_lines(points)
    
    random.seed(42)
    
    # Simulated annealing with both single-point moves and swap moves
    T = 15.0
    T_min = 0.1
    cooling = 0.99998
    max_iter = 300000
    
    for iteration in range(max_iter):
        if T <= T_min:
            break
            
        if random.random() < 0.65:
            # Single point move
            idx = random.randint(0, 11)
            old_x, old_y = points[idx]
            
            if random.random() < 0.7:
                step = random.randint(1, 5)
                new_x = old_x + random.choice([-step, step])
                new_y = old_y + random.choice([-step, step])
            else:
                new_x = random.randint(-40, 40)
                new_y = random.randint(-40, 40)
            
            if new_x == old_x and new_y == old_y:
                continue
            
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(points) if i != idx):
                continue
            
            points[idx] = [new_x, new_y]
            new_ordinary = count_ordinary_lines(points)
            
            delta = new_ordinary - best_ordinary
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                if new_ordinary < best_ordinary:
                    best_ordinary = new_ordinary
                    best_points = [p[:] for p in points]
            else:
                points[idx] = [old_x, old_y]
        else:
            # Swap two points
            i, j = random.sample(range(12), 2)
            old_i = points[i][:]
            old_j = points[j][:]
            
            points[i], points[j] = old_j, old_i
            new_ordinary = count_ordinary_lines(points)
            
            delta = new_ordinary - best_ordinary
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                if new_ordinary < best_ordinary:
                    best_ordinary = new_ordinary
                    best_points = [p[:] for p in points]
            else:
                points[i], points[j] = old_i, old_j
        
        T *= cooling
        
        if best_ordinary <= 6:
            break
    
    result = [p[:] for p in best_points]
    return result
# EVOLVE-BLOCK-END

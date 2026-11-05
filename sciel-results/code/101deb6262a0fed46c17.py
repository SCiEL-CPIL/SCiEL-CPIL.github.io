import time
import random
from itertools import combinations
from math import gcd

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the points.
    Uses a more efficient approach by tracking line membership.
    """
    n = len(points)
    if n < 3:
        return 0
    
    # Map each line to the set of point indices on it
    line_to_points = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        g = _gcd(_gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
            
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    # Count lines with exactly 2 points
    ordinary = 0
    for key, pts in line_to_points.items():
        if len(pts) == 2:
            ordinary += 1
            
    return ordinary

def _fast_local_search(points, deadline, max_iter=2000):
    """
    Faster local search with incremental score updates.
    When moving point i, only recalculate lines involving point i.
    """
    n = len(points)
    best_points = [p[:] for p in points]
    best_score = count_ordinary_lines(best_points)
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = _gcd(_gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)
    
    iters = 0
    while iters < max_iter and time.time() < deadline:
        iters += 1
        improved = False
        
        # Try moving each point
        for i in range(n):
            x, y = best_points[i]
            
            # Generate candidate positions: nearby + some random
            candidates = []
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    candidates.append((x + dx, y + dy))
            # Add some random jumps
            for _ in range(20):
                dx = random.randint(-30, 30)
                dy = random.randint(-30, 30)
                if dx != 0 or dy != 0:
                    candidates.append((x + dx, y + dy))
            
            for nx, ny in candidates:
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                new_pt = (nx, ny)
                # Check distinctness
                if any(new_pt == tuple(p) for p in best_points):
                    continue
                
                # Quick incremental update:
                # Remove old lines involving point i, add new lines
                old_lines = {}
                for j in range(n):
                    if j == i:
                        continue
                    key = get_line_key(best_points[i], best_points[j])
                    old_lines[key] = old_lines.get(key, 0) + 1
                
                new_lines = {}
                for j in range(n):
                    if j == i:
                        continue
                    key = get_line_key(new_pt, best_points[j])
                    new_lines[key] = new_lines.get(key, 0) + 1
                
                # Count ordinary lines: a line is ordinary if it has exactly 2 points total
                # We need to track all lines, not just those through point i
                # For efficiency, let's just do a full count but with early termination
                new_points = best_points[:]
                new_points[i] = [nx, ny]
                new_score = count_ordinary_lines(new_points)
                
                if new_score < best_score:
                    best_score = new_score
                    best_points = new_points
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_points, best_score

def _try_line_construction(deadline):
    """
    Try to construct a configuration by specifying lines with 3+ points.
    Strategy: pick several lines, ensure they share points appropriately,
    and fill remaining points to minimize ordinary lines.
    """
    best_points = None
    best_score = float('inf')
    
    # Predefined good line structures for 12 points
    # Each structure is a list of lines, where each line is a list of point indices
    # We'll try to realize these with integer coordinates
    
    # Structure 1: 4 lines of 3 points, with overlaps
    # Line A: points 0,1,2
    # Line B: points 0,3,4  
    # Line C: points 1,5,6
    # Line D: points 2,7,8
    # Then points 9,10,11 placed to create more collinearities
    
    # Let's try specific coordinate assignments
    candidates = [
        # 3x3 grid plus 3 points that extend lines
        [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,0],[0,3],[3,3]],
        # Variant
        [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,1],[1,3],[3,3]],
        # Variant with negative coords
        [[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1],[2,0],[0,2],[2,2]],
        # Cross shape
        [[-2,0],[-1,0],[0,0],[1,0],[2,0],[0,-2],[0,-1],[0,1],[0,2],[-1,1],[1,-1],[-1,-1]],
        # Diagonal heavy
        [[0,0],[1,1],[2,2],[3,3],[0,1],[1,2],[2,3],[3,0],[1,0],[2,1],[3,2],[0,3]],
        # Hexagonal-ish
        [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[2,2],[-1,1],[3,1],[1,-1],[1,3]],
    ]
    
    for pts in candidates:
        if len(set(map(tuple, pts))) != 12:
            continue
        score = count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in pts]
    
    return best_points, best_score

def _simulated_annealing(points, deadline, max_iter=5000):
    """
    Simulated annealing to escape local minima.
    """
    n = len(points)
    current_points = [p[:] for p in points]
    current_score = count_ordinary_lines(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    T = 10.0  # Initial temperature
    cooling = 0.999
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Random move: pick a point and move it randomly
        i = random.randint(0, n - 1)
        x, y = current_points[i]
        dx = random.randint(-50, 50)
        dy = random.randint(-50, 50)
        nx, ny = x + dx, y + dy
        
        if abs(nx) > 10**9 or abs(ny) > 10**9:
            continue
        new_pt = [nx, ny]
        
        # Check distinctness
        if any(new_pt == p for p in current_points):
            continue
        
        new_points = current_points[:]
        new_points[i] = new_pt
        new_score = count_ordinary_lines(new_points)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or random.random() < (2.718281828 ** (-delta / T)):
            current_points = new_points
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in new_points]
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Phase 1: Try known good configurations
    best_points, best_score = _try_line_construction(deadline)
    
    # Phase 2: Local search from best config
    if best_points is not None:
        best_points, best_score = _fast_local_search(best_points, deadline, max_iter=1000)
    
    # Phase 3: If we have time, try simulated annealing from multiple starts
    if time.time() < deadline - 20:
        # Try SA from several different starting points
        start_configs = [
            [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,0],[0,3],[3,3]],
            [[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1],[2,0],[0,2],[2,2]],
            [[-2,0],[-1,0],[0,0],[1,0],[2,0],[0,-2],[0,-1],[0,1],[0,2],[-1,1],[1,-1],[-1,-1]],
        ]
        
        for cfg in start_configs:
            if time.time() > deadline - 10:
                break
            pts, sc = _simulated_annealing(cfg, deadline, max_iter=2000)
            if sc < best_score:
                best_score = sc
                best_points = pts
    
    # Final local search polish
    if best_points is not None and time.time() < deadline - 5:
        best_points, best_score = _fast_local_search(best_points, deadline, max_iter=500)
    
    # Fallback
    if best_points is None:
        best_points = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]]
    
    return best_points

# EVOLVE-BLOCK-END

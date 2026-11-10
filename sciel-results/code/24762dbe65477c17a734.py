import time
import random
from itertools import combinations
from math import gcd

# EVOLVE-BLOCK-START

def _count_ordinary(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize
            g = gcd(abs(A), abs(B))
            if g == 0:
                # Points are identical? Should not happen.
                continue
            A //= g
            B //= g
            C //= g
            
            # Sign convention
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
            
    ordinary = 0
    for pts in line_points.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _all_collinear(points):
    if len(points) < 3:
        return False
    x0, y0 = points[0]
    x1, y1 = points[1]
    for i in range(2, len(points)):
        x2, y2 = points[i]
        if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
            return False
    return True

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_points = None
    best_score = float('inf')
    
    # Helper to update best
    def update_best(pts):
        nonlocal best_points, best_score
        if len(set(map(tuple, pts))) != 12:
            return
        if _all_collinear(pts):
            return
        s = _count_ordinary(pts)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in pts]
    
    # 1. Try some structured seeds
    # 3x4 grid
    update_best([[x, y] for x in range(4) for y in range(3)])
    # 4x3 grid
    update_best([[x, y] for x in range(3) for y in range(4)])
    # 6x2 grid
    update_best([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # 2. Simulated Annealing
    # We will run multiple restarts.
    # Since n=12, the search space is small enough for SA to find the global min (6) quickly.
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        # Initialize with random points in a small box
        # Using a small box helps find collinearities
        pts = []
        seen = set()
        while len(pts) < 12:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
        
        if _all_collinear(pts):
            continue
            
        current_score = _count_ordinary(pts)
        if current_score < best_score:
            best_score = current_score
            best_points = [list(p) for p in pts]
            
        # Annealing
        T = 10.0
        cooling = 0.95
        max_iters = 2000
        
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
                
            # Pick a random point to move
            idx = random.randint(0, 11)
            old_x, old_y = pts[idx]
            
            # Move to a nearby point
            # Radius of move can be adapted, but small moves are good for fine-tuning
            # However, to escape local minima, we need larger moves sometimes.
            # Let's try a mix.
            if random.random() < 0.1:
                # Large jump
                new_x = random.randint(-10, 10)
                new_y = random.randint(-10, 10)
            else:
                # Small step
                new_x = old_x + random.randint(-2, 2)
                new_y = old_y + random.randint(-2, 2)
                
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
                
            # Check distinctness
            if (new_x, new_y) in seen:
                # If it's the same point, skip
                if new_x == old_x and new_y == old_y:
                    continue
                # If it collides with another point, skip
                # Note: seen contains all points. If (new_x, new_y) is in seen,
                # it must be the same point (since we are moving it) or a collision.
                # Actually, we remove old point from seen? No, we just check.
                # Let's just check against all other points.
                collision = False
                for j, (px, py) in enumerate(pts):
                    if j != idx and px == new_x and py == new_y:
                        collision = True
                        break
                if collision:
                    continue
                    
            # Update point
            pts[idx] = [new_x, new_y]
            
            if _all_collinear(pts):
                # Revert
                pts[idx] = [old_x, old_y]
                continue
                
            new_score = _count_ordinary(pts)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [list(p) for p in pts]
            else:
                # Revert
                pts[idx] = [old_x, old_y]
                
            T *= cooling
            if T < 0.1:
                break

    # Fallback
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        
    return best_points

# EVOLVE-BLOCK-END

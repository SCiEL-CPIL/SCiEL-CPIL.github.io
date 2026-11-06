"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter

def _doubled_area(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

def _evaluate(points):
    """Return (max_multiplicity, best_area) for a set of points."""
    areas = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    areas.append(a)
    if not areas:
        return 0, 0
    c = Counter(areas)
    best_area, best_mult = c.most_common(1)[0]
    return best_mult, best_area

def construct():
    deadline = time.time() + 80.0
    
    # Seed: 6x5 grid
    best_points = [(x, y) for x in range(6) for y in range(5)]
    best_mult, best_area = _evaluate(best_points)
    
    # Try various rectangular grid configurations
    candidates = [
        [(x, y) for x in range(10) for y in range(3)],
        [(x, y) for x in range(15) for y in range(2)],
        [(x, y) for x in range(6) for y in range(5)],
        [(x, y) for x in range(5) for y in range(6)],
        [(x, y) for x in range(3) for y in range(10)],
        [(x, y) for x in range(2) for y in range(15)],
    ]
    
    for pts in candidates:
        if len(pts) != 30 or len(set(pts)) != 30:
            continue
        m, a = _evaluate(pts)
        if m > best_mult:
            best_mult = m
            best_area = a
            best_points = list(pts)
    
    # Try slanted/parallelogram grids
    for a1, b1, c1, d1 in [(1,0,0,1),(1,0,1,1),(1,0,0,2),(1,1,0,1),(2,0,0,1),(1,0,2,1),(1,1,1,0)]:
        for w in range(3, 11):
            for h in range(3, 11):
                if w * h != 30:
                    continue
                pts = [(i*a1 + j*c1, i*b1 + j*d1) for i in range(w) for j in range(h)]
                # Check bounds and uniqueness
                if any(x < -30 or x > 30 or y < -30 or y > 30 for x, y in pts):
                    continue
                if len(set(pts)) != 30:
                    continue
                m, a = _evaluate(pts)
                if m > best_mult:
                    best_mult = m
                    best_area = a
                    best_points = list(pts)
    
    # Local search
    import random
    random.seed(42)
    current = list(best_points)
    current_mult = best_mult
    current_area = best_area
    
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for idx in range(30):
            if time.time() > deadline - 5:
                break
            x0, y0 = current[idx]
            # Try nearby moves
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    if (nx, ny) == current[idx]:
                        continue
                    new_pts = list(current)
                    new_pts[idx] = (nx, ny)
                    if len(set(new_pts)) != 30:
                        continue
                    m, a = _evaluate(new_pts)
                    if m > current_mult:
                        current_mult = m
                        current = new_pts
                        current_area = a
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved and iteration % 100 == 0:
            # Random restart
            w, h = random.choice([(6,5),(5,6),(10,3),(3,10),(15,2),(2,15)])
            current = [(x, y) for x in range(w) for y in range(h)]
            current_mult, current_area = _evaluate(current)
        
        if current_mult > best_mult:
            best_mult = current_mult
            best_points = list(current)
            best_area = current_area
    
    return best_points
# EVOLVE-BLOCK-END

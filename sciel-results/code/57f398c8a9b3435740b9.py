import random
import time
import math

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, -x1)
    if dy == 0:
        return (0, 1, -y1)
    g = _gcd(dx, dy)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    g2 = _gcd(dy, dx)
    if g2 > 1:
        dy //= g2
        dx //= g2
        c //= g2
    return (dy, -dx, c)

def _count_ordinary(points):
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return sum(1 for pts in lines.values() if len(pts) == 2)

def _generate_3x4():
    return [[x, y] for x in range(4) for y in range(3)]

def _generate_structured():
    # 4 points on each of 3 lines forming a triangle
    # Line 1: y = 0, x = 0,1,2,3
    # Line 2: y = x, x = 0,1,2,3 -> (0,0),(1,1),(2,2),(3,3) but (0,0) shared
    # Better: 3 separate lines
    # L1: y=0, points (0,0),(1,0),(2,0),(3,0)
    # L2: x=0, points (0,1),(0,2),(0,3),(0,4)
    # L3: x+y=5, points (1,4),(2,3),(3,2),(4,1)
    pts = []
    for x in range(4):
        pts.append([x, 0])
    for y in range(1, 5):
        pts.append([0, y])
    for x in range(1, 5):
        pts.append([x, 5-x])
    # Check distinctness
    seen = set()
    unique = []
    for p in pts:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique.append(p)
    return unique

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    candidates = [_generate_3x4(), _generate_structured()]
    best_points = None
    best_score = float('inf')
    for cand in candidates:
        if len(cand) != 12:
            continue
        s = _count_ordinary(cand)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in cand]
    
    if best_points is None:
        best_points = _generate_3x4()
        best_score = _count_ordinary(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Local search with targeted moves
    max_iter = 20000
    stuck = 0
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random point
        idx = random.randint(0, 11)
        old_x, old_y = current_points[idx]
        
        # Generate candidate new positions
        candidates = []
        
        # 1. Try to place on a line through 2 other points (create 3-collinearity)
        for i in range(12):
            if i == idx:
                continue
            for j in range(i + 1, 12):
                if j == idx:
                    continue
                p1 = current_points[i]
                p2 = current_points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Find integer points on this line near the old position
                # Line: (x1 + t*dx, y1 + t*dy) for integer t
                if dx == 0:
                    # Vertical line x = p1[0]
                    for t in range(-2, 3):
                        nx = p1[0]
                        ny = p1[1] + t * abs(dy)
                        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                            candidates.append([nx, ny])
                elif dy == 0:
                    for t in range(-2, 3):
                        nx = p1[0] + t * abs(dx)
                        ny = p1[1]
                        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                            candidates.append([nx, ny])
                else:
                    g = _gcd(dx, dy)
                    dxr = dx // g
                    dyr = dy // g
                    for t in range(-2, 3):
                        nx = p1[0] + t * dxr
                        ny = p1[1] + t * dyr
                        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                            candidates.append([nx, ny])
        
        # 2. Random small moves
        for _ in range(5):
            nx = old_x + random.randint(-3, 3)
            ny = old_y + random.randint(-3, 3)
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                candidates.append([nx, ny])
        
        # 3. Random larger moves
        for _ in range(3):
            nx = old_x + random.randint(-20, 20)
            ny = old_y + random.randint(-20, 20)
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                candidates.append([nx, ny])
        
        found_improvement = False
        for cand in candidates:
            # Check distinctness
            is_distinct = True
            for j in range(12):
                if j != idx and current_points[j] == cand:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
            
            current_points[idx] = cand
            new_score = _count_ordinary(current_points)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                found_improvement = True
                break
            else:
                current_points[idx] = [old_x, old_y]
        
        if not found_improvement:
            stuck += 1
            if stuck > 200:
                stuck = 0
                # Restart from best
                current_points = [p[:] for p in best_points]
                current_score = best_score
        else:
            stuck = 0
    
    return best_points

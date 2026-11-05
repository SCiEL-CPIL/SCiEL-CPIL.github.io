import time
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(abs(dx), abs(dy))
    if g == 0:
        return (0, 0, 0)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _evaluate(points):
    n = len(points)
    point_set = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False, 0
        pt = (x, y)
        if pt in point_set:
            return False, 0
        point_set.add(pt)
        
    lines = {}
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            key = _line_key(p1, p2)
            lines[key] = lines.get(key, 0) + 1
            
    score = 0
    for count in lines.values():
        if count >= 10:
            return False, 0
        if count == 6:
            score += 1
            
    return True, score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    current_points = [[x, y] for x in range(4) for y in range(4)]
    is_valid, current_score = _evaluate(current_points)
    if not is_valid:
        current_points = [[i, 0] for i in range(16)]
        is_valid, current_score = _evaluate(current_points)
        
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    while time.time() < deadline:
        idx = random.randint(0, 15)
        old_point = current_points[idx]
        
        r = random.random()
        if r < 0.3:
            nx = old_point[0] + random.randint(-3, 3)
            ny = old_point[1] + random.randint(-3, 3)
        elif r < 0.6:
            nx = old_point[0] + random.randint(-10, 10)
            ny = old_point[1] + random.randint(-10, 10)
        else:
            nx = random.randint(-1000, 1000)
            ny = random.randint(-1000, 1000)
        
        new_point = [nx, ny]
        
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
            
        collision = False
        for k, p in enumerate(current_points):
            if k != idx and p == new_point:
                collision = True
                break
        if collision:
            continue
        
        current_points[idx] = new_point
        is_valid, new_score = _evaluate(current_points)
        
        if is_valid:
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
            if new_score >= current_score:
                current_score = new_score
            else:
                current_points[idx] = old_point
        else:
            current_points[idx] = old_point
            
    return best_points

import random
from math import gcd

def count_ordinary(points):
    n = len(points)
    ordinary = 0
    lines_seen = set()
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            a = dy
            b = -dx
            c = a*x1 + b*y1
            g2 = gcd(abs(a), gcd(abs(b), abs(c)))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key in lines_seen:
                continue
            count = 0
            for k in range(n):
                xk, yk = points[k]
                if a*xk + b*yk == c:
                    count += 1
            lines_seen.add(key)
            if count == 2:
                ordinary += 1
    return ordinary

def construct():
    best_points = None
    best_score = float('inf')
    
    random.seed(42)
    
    for trial in range(500):
        points = [[random.randint(-15, 15), random.randint(-15, 15)] for _ in range(12)]
        # Ensure distinct
        while len(set(tuple(p) for p in points)) < 12:
            points = [[random.randint(-15, 15), random.randint(-15, 15)] for _ in range(12)]
        
        score = count_ordinary(points)
        
        # Local search
        improved = True
        while improved:
            improved = False
            for _ in range(50):  # random restarts within local search
                i = random.randint(0, 11)
                old_x, old_y = points[i]
                new_x = old_x + random.randint(-5, 5)
                new_y = old_y + random.randint(-5, 5)
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                if any(new_x == p[0] and new_y == p[1] for p in points if p != points[i]):
                    continue
                points[i] = [new_x, new_y]
                new_score = count_ordinary(points)
                if new_score < score:
                    score = new_score
                    improved = True
                else:
                    points[i] = [old_x, old_y]
        
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
            
        if best_score <= 6:
            break
    
    return best_points

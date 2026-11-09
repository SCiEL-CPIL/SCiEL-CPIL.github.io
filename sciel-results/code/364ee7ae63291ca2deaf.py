import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for 150s limit
    
    N = 20
    bounds = 100
    
    def is_valid_config(points):
        for i in range(N):
            xi, yi = points[i]
            slopes = {}
            for j in range(N):
                if i == j: continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    return False
                if dx == 0:
                    key = (1, 0)
                elif dy == 0:
                    key = (0, 1)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx //= g
                    dy //= g
                    if dx < 0:
                        dx = -dx
                        dy = -dy
                    key = (dx, dy)
                if key in slopes:
                    slopes[key] += 1
                    if slopes[key] >= 3:
                        return False
                else:
                    slopes[key] = 1
        return True

    def count_triples(points):
        count = 0
        for i in range(N):
            xi, yi = points[i]
            for j in range(i+1, N):
                xj, yj = points[j]
                for k in range(j+1, N):
                    xk, yk = points[k]
                    if (xj - xi) * (yk - yi) == (yj - yi) * (xk - xi):
                        count += 1
        return count

    def get_valid_random_point(points, exclude_idx):
        while True:
            x = random.randint(-bounds, bounds)
            y = random.randint(-bounds, bounds)
            p = [x, y]
            if p in points:
                continue
            
            slopes = {}
            for idx, q in enumerate(points):
                if idx == exclude_idx: continue
                xq, yq = q
                dx = xq - x
                dy = yq - y
                if dx == 0 and dy == 0:
                    return None
                if dx == 0:
                    key = (1, 0)
                elif dy == 0:
                    key = (0, 1)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx //= g
                    dy //= g
                    if dx < 0:
                        dx = -dx
                        dy = -dy
                    key = (dx, dy)
                
                if key in slopes:
                    slopes[key] += 1
                    if slopes[key] >= 3:
                        return None
                else:
                    slopes[key] = 1
            return p

    points = []
    for i in range(N):
        p = None
        while p is None:
            p = get_valid_random_point(points, -1)
            if p is None:
                continue
        points.append(p)
    
    if not is_valid_config(points):
        points = [[x, x*x] for x in range(-10, 10)]
        
    best_points = points.copy()
    best_triples = count_triples(points)
    current_triples = best_triples
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline:
        i = random.randint(0, N-1)
        old_p = points[i]
        
        new_p = None
        for _ in range(50):
            candidate = get_valid_random_point(points, i)
            if candidate is not None:
                new_p = candidate
                break
        
        if new_p is None:
            continue
            
        points[i] = new_p
        new_triples = count_triples(points)
        
        delta = current_triples - new_triples
        
        if new_triples > current_triples:
            current_triples = new_triples
            if new_triples > best_triples:
                best_triples = new_triples
                best_points = points.copy()
        else:
            if random.random() < math.exp(delta / T):
                current_triples = new_triples
            else:
                points[i] = old_p
        
        T = max(T_min, T * cooling)
        
    return best_points

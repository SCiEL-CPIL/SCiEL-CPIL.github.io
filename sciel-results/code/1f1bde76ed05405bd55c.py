import time
import random
import math

_START = time.time()
_DEADLINE = _START + 140.0

def _count_ordinary(points):
    n = len(points)
    pts = [(p[0], p[1]) for p in points]
    seen = set()
    ordinary = 0
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dy), abs(dx))
            if g == 0:
                continue
            a = dy // g
            b = -dx // g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
            c = a * x1 + b * y1
            key = (a, b, c)
            if key in seen:
                continue
            seen.add(key)
            cnt = 0
            for (px, py) in pts:
                if a * px + b * py == c:
                    cnt += 1
            if cnt == 2:
                ordinary += 1
    return ordinary

def construct():
    # Start with 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    best_count = _count_ordinary(points)
    best_points = [p[:] for p in points]
    
    random.seed(42)
    
    while time.time() < _DEADLINE:
        if best_count == 6:
            break
        
        improved = False
        # Shuffle order of points to try
        order = list(range(12))
        random.shuffle(order)
        
        for i in order:
            if time.time() > _DEADLINE:
                break
            # Try to move point i to a line defined by two other points
            for j in range(12):
                if i == j:
                    continue
                for k in range(12):
                    if i == k or j == k:
                        continue
                    p1 = points[j]
                    p2 = points[k]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    g = math.gcd(abs(dx), abs(dy))
                    if g == 0:
                        continue
                    dx //= g
                    dy //= g
                    
                    # Try a few points on this line
                    # We want to replace points[i] with a new point on this line.
                    # Try t = -5 to 5 relative to p1
                    found_improvement = False
                    for t in range(-10, 11):
                        if t == 0:
                            continue
                        new_x = p1[0] + t * dx
                        new_y = p1[1] + t * dy
                        if new_x == points[i][0] and new_y == points[i][1]:
                            continue
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        # Check if new point is already occupied
                        occupied = False
                        for m in range(12):
                            if m == i:
                                continue
                            if points[m][0] == new_x and points[m][1] == new_y:
                                occupied = True
                                break
                        if occupied:
                            continue
                        
                        # Try the move
                        old_x, old_y = points[i]
                        points[i] = [new_x, new_y]
                        count = _count_ordinary(points)
                        if count < best_count:
                            best_count = count
                            best_points = [p[:] for p in points]
                            improved = True
                            found_improvement = True
                            break
                        else:
                            points[i] = [old_x, old_y]
                    
                    if found_improvement:
                        break
                if found_improvement:
                    break
            if found_improvement:
                break
        
        if not improved:
            # Random perturbation
            i = random.randint(0, 11)
            old_x, old_y = points[i]
            improved_perturb = False
            for _ in range(50):
                new_x = random.randint(-10**6, 10**6)
                new_y = random.randint(-10**6, 10**6)
                occupied = False
                for m in range(12):
                    if m == i:
                        continue
                    if points[m][0] == new_x and points[m][1] == new_y:
                        occupied = True
                        break
                if occupied:
                    continue
                points[i] = [new_x, new_y]
                count = _count_ordinary(points)
                if count < best_count:
                    best_count = count
                    best_points = [p[:] for p in points]
                    improved_perturb = True
                    break
                else:
                    points[i] = [old_x, old_y]
            
            if not improved_perturb:
                # If stuck, restart with a new random configuration?
                # Or just continue with random moves.
                # Let's try a few random moves.
                for _ in range(10):
                    i = random.randint(0, 11)
                    old_x, old_y = points[i]
                    new_x = random.randint(-10**6, 10**6)
                    new_y = random.randint(-10**6, 10**6)
                    occupied = False
                    for m in range(12):
                        if m == i:
                            continue
                        if points[m][0] == new_x and points[m][1] == new_y:
                            occupied = True
                            break
                    if occupied:
                        continue
                    points[i] = [new_x, new_y]
                    count = _count_ordinary(points)
                    if count < best_count:
                        best_count = count
                        best_points = [p[:] for p in points]
                        break
                    else:
                        points[i] = [old_x, old_y]

    return best_points

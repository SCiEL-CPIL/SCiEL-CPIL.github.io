import time
import random
from collections import Counter

def _doubled_area(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    areas = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                d = abs(xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj))
                if d > 0:
                    areas.append(d)
    if not areas:
        return 0
    return Counter(areas).most_common(1)[0][1]

def _valid(points):
    if len(points) != 30:
        return False
    seen = set()
    for x, y in points:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return True

def _grid_start():
    points = []
    for i in range(6):
        for j in range(5):
            x = -15 + i * 5
            y = -10 + j * 5
            points.append((x, y))
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    rng = random.Random(42)
    
    # Start with grid
    best_points = _grid_start()
    best_score = _score(best_points)
    
    n = 30
    
    # Local search
    while time.time() < deadline:
        improved = False
        # Try random swaps of two points
        for _ in range(300):
            i, j = rng.sample(range(n), 2)
            candidate = best_points[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            # Quick validity check: just check no duplicates (positions are always in range after swap)
            if len(set(candidate)) != n:
                continue
            s = _score(candidate)
            if s > best_score:
                best_points = candidate
                best_score = s
                improved = True
                break
        if not improved:
            # Try moving one point to a random valid location
            found = False
            for _ in range(100):
                i = rng.randrange(n)
                nx = rng.randint(-30, 30)
                ny = rng.randint(-30, 30)
                candidate = best_points[:]
                candidate[i] = (nx, ny)
                if len(set(candidate)) != n:
                    continue
                s = _score(candidate)
                if s > best_score:
                    best_points = candidate
                    best_score = s
                    found = True
                    break
            if not found:
                # Try a fresh restart with perturbed grid
                if time.time() > start_time + 10:
                    start = _grid_start()
                    for _ in range(rng.randint(1, 3)):
                        idx = rng.randrange(30)
                        x, y = start[idx]
                        start[idx] = (x + rng.randint(-3, 3), y + rng.randint(-3, 3))
                    if len(set(start)) == 30 and all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in start):
                        best_points = start
                        best_score = _score(best_points)
    
    return best_points

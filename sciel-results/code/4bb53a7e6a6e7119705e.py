import time
import random
from math import gcd
from itertools import combinations

def _line_key(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * p1[0] - dx * p1[1]
    return (dx, dy, c)

def _evaluate(points):
    line_map = {}
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    count_4 = 0
    has_5 = False
    for key, idx_set in line_map.items():
        cnt = len(idx_set)
        if cnt == 4:
            count_4 += 1
        elif cnt >= 5:
            has_5 = True
    if has_5:
        return 0
    return count_4

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 5s safety margin
    
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _evaluate(points)
    best_points = [p[:] for p in points]
    
    # Phase 1: Hill climbing with single-point moves
    # Use a smaller neighborhood for speed
    improved = True
    iterations = 0
    while improved and time.time() < deadline:
        improved = False
        iterations += 1
        # Shuffle point order for diversity
        order = list(range(16))
        random.seed(iterations)
        random.shuffle(order)
        for idx in order:
            old_x, old_y = points[idx]
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 200 or abs(new_y) > 200:
                        continue
                    new_points = [p[:] for p in points]
                    new_points[idx] = [new_x, new_y]
                    # Quick distinct check
                    valid = True
                    for i in range(16):
                        for j in range(i+1, 16):
                            if new_points[i] == new_points[j]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    sc = _evaluate(new_points)
                    if sc > best_score:
                        points = new_points
                        best_score = sc
                        best_points = [p[:] for p in points]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    # Phase 2: Try structured perturbations
    if time.time() < deadline:
        # Try shifting entire rows or columns
        for row in range(4):
            for shift in range(-3, 4):
                if shift == 0:
                    continue
                new_points = [p[:] for p in points]
                for i in range(16):
                    if points[i][1] == row:
                        new_points[i][0] += shift
                if all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in new_points):
                    # Check distinct
                    valid = True
                    for i in range(16):
                        for j in range(i+1, 16):
                            if new_points[i] == new_points[j]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        sc = _evaluate(new_points)
                        if sc > best_score:
                            best_score = sc
                            best_points = [p[:] for p in new_points]
    
    # Phase 3: Random restarts with local search
    if time.time() < deadline:
        for restart in range(50):
            if time.time() > deadline:
                break
            # Start from best, perturb randomly
            pts = [p[:] for p in best_points]
            for _ in range(3):
                idx = random.randint(0, 15)
                pts[idx] = [random.randint(-20, 20), random.randint(-20, 20)]
            # Quick local search
            for _ in range(100):
                if time.time() > deadline:
                    break
                idx = random.randint(0, 15)
                ox, oy = pts[idx]
                nx = ox + random.randint(-3, 3)
                ny = oy + random.randint(-3, 3)
                if abs(nx) > 200 or abs(ny) > 200:
                    continue
                new_pts = [p[:] for p in pts]
                new_pts[idx] = [nx, ny]
                valid = True
                for i in range(16):
                    for j in range(i+1, 16):
                        if new_pts[i] == new_pts[j]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                sc = _evaluate(new_pts)
                if sc > _evaluate(pts):
                    pts = new_pts
            sc = _evaluate(pts)
            if sc > best_score:
                best_score = sc
                best_points = [p[:] for p in pts]
    
    return best_points

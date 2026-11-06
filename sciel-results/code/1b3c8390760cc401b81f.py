import time
import numpy as np
from collections import Counter
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    def fast_score(points):
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        areas = Counter()
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    d = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                    d = abs(d)
                    if d > 0:
                        areas[d] += 1
        if not areas:
            return 0
        return max(areas.values())

    best_points = None
    best_score = 0

    dims = [(1,30),(2,15),(3,10),(5,6),(6,5),(10,3),(15,2),(30,1)]
    
    for w, h in dims:
        for sx in range(1, 11):
            for sy in range(1, 11):
                ox = -(w - 1) * sx // 2
                oy = -(h - 1) * sy // 2
                pts = [(ox + x * sx, oy + y * sy) for x in range(w) for y in range(h)]
                if any(p[0] < -30 or p[0] > 30 or p[1] < -30 or p[1] > 30 for p in pts):
                    continue
                s = fast_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = list(pts)
                if time.time() - start_time > DEADLINE:
                    break
            if time.time() - start_time > DEADLINE:
                break
        if time.time() - start_time > DEADLINE:
            break

    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = fast_score(best_points)

    random.seed(42)
    current_points = list(best_points)
    current_score = best_score
    no_improve = 0
    max_no_improve = 30

    while time.time() - start_time < DEADLINE and no_improve < max_no_improve:
        improved = False
        for i in range(30):
            if time.time() - start_time > DEADLINE:
                break
            ox, oy = current_points[i]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    if (nx, ny) in current_points:
                        continue
                    new_points = current_points.copy()
                    new_points[i] = (nx, ny)
                    s = fast_score(new_points)
                    if s > current_score:
                        current_points = new_points
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_points = list(new_points)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            for _ in range(100):
                if time.time() - start_time > DEADLINE:
                    break
                i = random.randint(0, 29)
                ox, oy = current_points[i]
                nx = random.randint(max(-30, ox - 10), min(30, ox + 10))
                ny = random.randint(max(-30, oy - 10), min(30, oy + 10))
                if (nx, ny) in current_points:
                    continue
                new_points = current_points.copy()
                new_points[i] = (nx, ny)
                s = fast_score(new_points)
                if s > current_score:
                    current_points = new_points
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_points = list(new_points)
                    improved = True
                    break
        if not improved:
            no_improve += 1

    return best_points

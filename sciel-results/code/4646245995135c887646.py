import time
import random
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 80.0

    def seed_grid():
        return [(x, y) for x in range(6) for y in range(5)]

    def compute_multiplicity(points):
        n = len(points)
        cnt = Counter()
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                dx1 = xj - xi
                dy1 = yj - yi
                for k in range(j + 1, n):
                    xk, yk = points[k]
                    det = dx1 * (yk - yi) - dy1 * (xk - xi)
                    ad = det if det > 0 else -det
                    if ad > 0:
                        cnt[ad] += 1
        if not cnt:
            return 0
        return max(cnt.values())

    best_pts = seed_grid()
    best_val = compute_multiplicity(best_pts)

    grid = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    rng = random.Random(999)

    for it in range(15000):
        if time.time() > deadline - 3:
            break
        pts = list(best_pts)
        idx = rng.randrange(30)
        old = pts[idx]
        new = rng.choice(grid)
        if new == old or new in pts:
            continue
        pts[idx] = new
        val = compute_multiplicity(pts)
        if val >= best_val:
            best_val = val
            best_pts = pts

    return best_pts

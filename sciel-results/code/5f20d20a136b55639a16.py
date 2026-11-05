"""
Maximize the largest multiplicity of a single positive doubled area among
non-degenerate triangles formed by 30 distinct integer points in [-30,30]^2.
"""

import time
import random
from collections import Counter

def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    area_count = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            dx1, dy1 = xj - xi, yj - yi
            for k in range(j+1, n):
                xk, yk = points[k]
                dx2, dy2 = xk - xi, yk - yi
                d = abs(dx1 * dy2 - dy1 * dx2)
                if d > 0:
                    area_count[d] = area_count.get(d, 0) + 1
    if not area_count:
        return 0
    return max(area_count.values())

def _two_line_construction(n_per_line=15, spacing=1, h=2, x_offset=0):
    """15 points on y=0, 15 points on y=h."""
    pts = []
    for i in range(n_per_line):
        pts.append((x_offset + i * spacing, 0))
    for i in range(n_per_line):
        pts.append((x_offset + i * spacing, h))
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 80

    # Start with two-line construction
    best_pts = _two_line_construction(15, 1, 2, 0)
    best_score = _score(best_pts)

    # Try different parameters
    for n_per_line in [15]:
        for spacing in [1, 2]:
            for h in [1, 2, 3, 4]:
                for x_off in range(-10, 11, 2):
                    pts = _two_line_construction(n_per_line, spacing, h, x_off)
                    if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                        s = _score(pts)
                        if s > best_score:
                            best_score = s
                            best_pts = pts

    # Local search
    random.seed(123)
    while time.time() < deadline:
        # Try moving one point
        idx = random.randint(0, 29)
        orig = best_pts[idx]
        found = False
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = orig[0] + dx, orig[1] + dy
                if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                    continue
                if (nx, ny) in best_pts:
                    continue
                new_pts = list(best_pts)
                new_pts[idx] = (nx, ny)
                if len(set(new_pts)) != 30:
                    continue
                s = _score(new_pts)
                if s > best_score:
                    best_score = s
                    best_pts = new_pts
                    found = True
                    break
            if found:
                break

        # Occasionally try moving two points
        if not found and random.random() < 0.3:
            i1, i2 = random.sample(range(30), 2)
            for _ in range(30):
                n1 = (best_pts[i1][0] + random.randint(-6, 6), best_pts[i1][1] + random.randint(-6, 6))
                n2 = (best_pts[i2][0] + random.randint(-6, 6), best_pts[i2][1] + random.randint(-6, 6))
                if not (-30 <= n1[0] <= 30 and -30 <= n1[1] <= 30):
                    continue
                if not (-30 <= n2[0] <= 30 and -30 <= n2[1] <= 30):
                    continue
                new_pts = list(best_pts)
                new_pts[i1] = n1
                new_pts[i2] = n2
                if len(set(new_pts)) != 30:
                    continue
                s = _score(new_pts)
                if s > best_score:
                    best_score = s
                    best_pts = new_pts
                    found = True
                    break

    return best_pts

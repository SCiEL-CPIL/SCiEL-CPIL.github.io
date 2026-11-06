"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import Counter
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s

    def evaluate(points):
        """Compute the maximum multiplicity of any positive doubled area."""
        counts = Counter()
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx1 = x2 - x1
                dy1 = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    dx2 = x3 - x1
                    dy2 = y3 - y1
                    det = abs(dx1 * dy2 - dy1 * dx2)
                    if det > 0:
                        counts[det] += 1
        if not counts:
            return 0
        return max(counts.values())

    # Strategy: Start with the two-parallel-lines configuration (15 points on y=0,
    # 15 on y=1) which is known to give a score of 420. Then try to improve via
    # local search and grid configurations.

    # Initial configuration: two horizontal lines
    best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    best_score = evaluate(best_points)

    # Try grid configurations
    # A w x h grid gives points (x, y) for x in [0, w), y in [0, h)
    # We need exactly 30 distinct points.
    # Possible grids: 5x6, 6x5, 10x3, 3x10, 15x2, 2x15
    # Also partial grids if w*h > 30

    grid_configs = [
        (5, 6), (6, 5), (10, 3), (3, 10), (15, 2), (2, 15)
    ]

    for w, h in grid_configs:
        if w * h != 30:
            continue
        pts = [(x, y) for x in range(w) for y in range(h)]
        if len(set(pts)) != 30:
            continue
        if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in pts):
            continue
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            if time.time() > deadline:
                break

    # Try some shifted/scaled versions of the two-line configuration
    # Lines at y=0 and y=d for small d
    for d in range(1, 5):
        pts = [(x, 0) for x in range(15)] + [(x, d) for x in range(15)]
        if len(set(pts)) != 30:
            continue
        if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in pts):
            continue
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts

    # Local search: try moving individual points to improve score
    import random
    random.seed(42)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(30):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[idx]
            # Try small perturbations
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    new_points = list(best_points)
                    new_points[idx] = (new_x, new_y)
                    if len(set(new_points)) != 30:
                        continue
                    score = evaluate(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best_points
# EVOLVE-BLOCK-END

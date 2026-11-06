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

    # Initial configuration: two horizontal lines at y=0 and y=1, each with 15 points
    best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    best_score = evaluate(best_points)

    # Try grid configurations: w x h grids with exactly 30 points
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
            return best_points

    # Try different line distances for two parallel lines
    for d in range(1, 6):
        pts = [(x, 0) for x in range(15)] + [(x, d) for x in range(15)]
        if len(set(pts)) != 30:
            continue
        if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in pts):
            continue
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts
        if time.time() > deadline:
            return best_points

    # Try 3-line configurations with varying distributions
    # Lines at y=0, y=d, y=2d
    for d in range(1, 4):
        for a in range(8, 23):  # points on first line
            for b in range(8, 23 - a + 8):  # points on second line
                c = 30 - a - b
                if c < 1 or c > 30:
                    continue
                if b < 1:
                    continue
                # Check bounds
                if 2 * d > 30:
                    continue
                pts = [(x, 0) for x in range(a)] + [(x, d) for x in range(b)] + [(x, 2 * d) for x in range(c)]
                if len(set(pts)) != 30:
                    continue
                if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in pts):
                    continue
                score = evaluate(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
                if time.time() > deadline:
                    return best_points

    # Local search: try moving individual points to improve score
    import random
    random.seed(42)

    # First pass: small perturbations (dx, dy in [-2, 2])
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(30):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[idx]
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

    # Second pass: random perturbations with larger moves
    for iteration in range(500):
        if time.time() > deadline:
            break
        new_points = list(best_points)
        idx = random.randint(0, 29)
        old_x, old_y = new_points[idx]
        # Try a random move within a reasonable range
        new_x = old_x + random.randint(-5, 5)
        new_y = old_y + random.randint(-5, 5)
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            continue
        new_points[idx] = (new_x, new_y)
        if len(set(new_points)) != 30:
            continue
        score = evaluate(new_points)
        if score > best_score:
            best_score = score
            best_points = new_points

    # Final pass: try swapping pairs of points
    for iteration in range(200):
        if time.time() > deadline:
            break
        new_points = list(best_points)
        i = random.randint(0, 29)
        j = random.randint(0, 29)
        if i == j:
            continue
        # Swap the points
        new_points[i], new_points[j] = new_points[j], new_points[i]
        if len(set(new_points)) != 30:
            continue
        score = evaluate(new_points)
        if score > best_score:
            best_score = score
            best_points = new_points

    return best_points
# EVOLVE-BLOCK-END

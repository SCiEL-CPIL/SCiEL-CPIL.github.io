"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - t0

    def doubled_areas(points):
        """Return dict: doubled_area -> count (positive only)."""
        n = len(points)
        cnt = defaultdict(int)
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                for k in range(j + 1, n):
                    xk, yk = points[k]
                    d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    if d > 0:
                        cnt[d] += 1
        return cnt

    def score_of(points):
        if len(points) != 30 or len(set(points)) != 30:
            return -1
        for (x, y) in points:
            if not (-30 <= x <= 30 and -30 <= y <= 30):
                return -1
        cnt = doubled_areas(points)
        if not cnt:
            return 0
        return max(cnt.values())

    best_points = None
    best_score = -1

    def try_points(pts):
        nonlocal best_points, best_score
        if len(pts) != 30 or len(set(pts)) != 30:
            return
        for (x, y) in pts:
            if not (-30 <= x <= 30 and -30 <= y <= 30):
                return
        s = score_of(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]

    # Family A: two horizontal lines y=0 and y=h, m points on y=0, n on y=h
    for h in [1, 2, 3, 5, 7]:
        for m in range(2, 31):
            n = 30 - m
            if n < 1:
                continue
            if elapsed() > DEADLINE:
                break
            line = [(i, 0) for i in range(m)]
            off = [(i, h) for i in range(n)]
            pts = line + off
            try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family B: three parallel lines
    splits = [(10, 10, 10), (12, 9, 9), (11, 10, 9), (14, 8, 8), (16, 7, 7),
              (15, 8, 7), (9, 10, 11), (8, 11, 11), (13, 9, 8), (15, 10, 5)]
    y_configs = [(0, 1, 2), (0, 1, 3), (0, 2, 4)]
    for split in splits:
        if sum(split) != 30:
            continue
        if elapsed() > DEADLINE:
            break
        for ys in y_configs:
            pts = []
            for count, y in zip(split, ys):
                pts += [(i, y) for i in range(count)]
            if len(pts) != 30:
                continue
            try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family C: four parallel lines
    splits4 = [(8, 8, 7, 7), (10, 8, 7, 5), (9, 9, 7, 5), (10, 10, 5, 5),
               (12, 8, 6, 4), (10, 8, 8, 4)]
    y_configs4 = [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 1, 2, 5)]
    for split in splits4:
        if sum(split) != 30:
            continue
        if elapsed() > DEADLINE:
            break
        for ys in y_configs4:
            pts = []
            for count, y in zip(split, ys):
                pts += [(i, y) for i in range(count)]
            if len(pts) != 30:
                continue
            try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family D: two lines with staggered x-offsets
    for h in [1, 2]:
        for m in range(10, 21):
            n = 30 - m
            if n < 1:
                continue
            if elapsed() > DEADLINE:
                break
            for x_off in range(0, min(5, n)):
                line = [(i, 0) for i in range(m)]
                off = [(i + x_off, h) for i in range(n)]
                if any(x < -30 or x > 30 for x, y in off):
                    continue
                pts = line + off
                try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family E: Grid-like configurations
    for rows in range(2, 8):
        if elapsed() > DEADLINE:
            break
        for cols in range(2, 12):
            total = rows * cols
            if total > 30:
                continue
            base_pts = [(x, y) for y in range(rows) for x in range(cols)]
            remaining = 30 - total
            if remaining < 0:
                continue
            for x_start in range(0, 5):
                extra = [(x_start + i, rows) for i in range(remaining)]
                if any(x < -30 or x > 30 for x, y in extra):
                    continue
                pts = base_pts + extra
                try_points(pts)
            if elapsed() > DEADLINE:
                break

    # Family F: Multiple y-levels
    for y_split in [(15, 15, 0), (10, 10, 10), (12, 10, 8), (14, 9, 7)]:
        if sum(y_split) != 30:
            continue
        if elapsed() > DEADLINE:
            break
        for ys in [(0, 1, 2), (0, 1, 3)]:
            pts = []
            for count, y in zip(y_split, ys):
                pts += [(i, y) for i in range(count)]
            if len(pts) != 30:
                continue
            try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family G: Maximize area=1 with two closely spaced lines
    for m in range(13, 18):
        n = 30 - m
        if elapsed() > DEADLINE:
            break
        pts = [(i, 0) for i in range(m)] + [(i, 1) for i in range(n)]
        try_points(pts)
        for off in range(-2, 3):
            pts2 = [(i, 0) for i in range(m)] + [(i + off, 1) for i in range(n)]
            if all(-30 <= x <= 30 for x, y in pts2):
                try_points(pts2)
        if elapsed() > DEADLINE:
            break

    # Family H: 3 lines with optimized distribution
    for m0 in range(8, 15):
        for m1 in range(8, 15):
            m2 = 30 - m0 - m1
            if m2 < 1:
                continue
            if elapsed() > DEADLINE:
                break
            pts = [(i, 0) for i in range(m0)] + [(i, 1) for i in range(m1)] + [(i, 2) for i in range(m2)]
            try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Local search: perturb the best configuration found so far
    if best_points is not None and elapsed() < DEADLINE - 5.0:
        random.seed(42)
        current = best_points[:]
        current_score = best_score
        no_improve = 0
        max_no_improve = 50
        
        while elapsed() < DEADLINE - 2.0 and no_improve < max_no_improve:
            # Pick a random point to move
            idx = random.randint(0, 29)
            ox, oy = current[idx]
            
            # Try small perturbations
            improved = False
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    # Check if point already exists
                    if (nx, ny) in set(current):
                        continue
                    
                    candidate = current[:]
                    candidate[idx] = (nx, ny)
                    
                    if len(set(candidate)) != 30:
                        continue
                    
                    s = score_of(candidate)
                    if s > current_score:
                        current = candidate
                        current_score = s
                        no_improve = 0
                        improved = True
                        break
                    elif s == current_score:
                        # Accept equal score moves to explore
                        if random.random() < 0.3:
                            current = candidate
                            no_improve += 1
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
        
        # Update best if local search improved
        if current_score > best_score:
            best_score = current_score
            best_points = current[:]

    if best_points is None:
        return [(x, y) for x in range(6) for y in range(5)]

    return best_points
# EVOLVE-BLOCK-END

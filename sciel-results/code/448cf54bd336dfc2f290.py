import time
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

    def local_search(points, max_iters=500):
        """Try small perturbations to improve score."""
        best_pts = list(points)
        best_s = score_of(best_pts)
        for _ in range(max_iters):
            if elapsed() > DEADLINE - 5:
                break
            # Pick a random point and try moving it
            idx = int((time.time() * 1000003) % 30)
            ox, oy = best_pts[idx]
            improved = False
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    new_pts = best_pts[:]
                    new_pts[idx] = (nx, ny)
                    if len(set(new_pts)) != 30:
                        continue
                    s = score_of(new_pts)
                    if s > best_s:
                        best_pts = new_pts
                        best_s = s
                        improved = True
                        break
                if improved:
                    break
            if not improved and elapsed() > DEADLINE - 20:
                break
        return best_pts, best_s

    best_points = None
    best_score = -1

    # Family A: m consecutive points on y=0, n points on y=1 (same height), distinct x.
    for m in range(2, 31):
        n = 30 - m
        if n < 1:
            continue
        if elapsed() > DEADLINE:
            break
        line = [(i, 0) for i in range(m)]
        off = [(i, 1) for i in range(n)]
        pts = line + off
        s = score_of(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]

    # Family B: m on y=0, n on y=h with h>1
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
            s = score_of(pts)
            if s > best_score:
                best_score = s
                best_points = pts[:]
        if elapsed() > DEADLINE:
            break

    # Family C: 3 parallel lines
    for split in [(10, 10, 10), (12, 9, 9), (11, 10, 9), (14, 8, 8), (16, 7, 7), (15, 8, 7)]:
        if sum(split) != 30:
            continue
        if elapsed() > DEADLINE:
            break
        for ys in [(0, 1, 2), (0, 1, 3), (0, 2, 4)]:
            pts = []
            for count, y in zip(split, ys):
                pts += [(i, y) for i in range(count)]
            if len(pts) != 30 or len(set(pts)) != 30:
                continue
            s = score_of(pts)
            if s > best_score:
                best_score = s
                best_points = pts[:]
        if elapsed() > DEADLINE:
            break

    # Family E: Rectangular Grids K x L where K*L=30
    for K in range(2, 31):
        if 30 % K != 0:
            continue
        L = 30 // K
        if elapsed() > DEADLINE:
            break
        if L - 1 > 30 or K - 1 > 30:
            continue
        pts = [(x, y) for x in range(L) for y in range(K)]
        if len(pts) != 30 or len(set(pts)) != 30:
            continue
        s = score_of(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]

    # Family F: Try various 2-line configurations with different x-ranges
    for m in range(5, 26):
        n = 30 - m
        if n < 5:
            continue
        if elapsed() > DEADLINE:
            break
        # Try different x offsets for the two lines
        for xoff in range(-5, 6):
            line1 = [(i + xoff, 0) for i in range(m)]
            line2 = [(i, 1) for i in range(n)]
            pts = line1 + line2
            if len(set(pts)) != 30:
                continue
            for (x, y) in pts:
                if not (-30 <= x <= 30 and -30 <= y <= 30):
                    break
            else:
                s = score_of(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts[:]
        if elapsed() > DEADLINE:
            break

    # Family G: Try 4 parallel lines
    for split in [(8, 8, 7, 7), (9, 8, 7, 6), (10, 8, 6, 6)]:
        if sum(split) != 30:
            continue
        if elapsed() > DEADLINE:
            break
        for ys in [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4)]:
            pts = []
            for count, y in zip(split, ys):
                pts += [(i, y) for i in range(count)]
            if len(pts) != 30 or len(set(pts)) != 30:
                continue
            s = score_of(pts)
            if s > best_score:
                best_score = s
                best_points = pts[:]
        if elapsed() > DEADLINE:
            break

    # Local search on best candidate
    if best_points is not None and elapsed() < DEADLINE - 10:
        refined_pts, refined_s = local_search(best_points, max_iters=300)
        if refined_s > best_score:
            best_score = refined_s
            best_points = refined_pts

    if best_points is None:
        # fallback seed
        return [(x, y) for x in range(6) for y in range(5)]

    return best_points
# EVOLVE-BLOCK-END

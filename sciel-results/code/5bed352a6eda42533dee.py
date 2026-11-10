"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    best_points = None
    best_score = -10**9

    def score_points(pts):
        """Exact score: #lines with exactly 4 points; -inf if any 5 collinear."""
        n = len(pts)
        # distinctness
        if len(set(map(tuple, pts))) != n:
            return -10**9
        for x, y in pts:
            if abs(x) > 1000 or abs(y) > 1000:
                return -10**9
        counts = {}
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                a = dy
                b = -dx
                c = a * x1 + b * y1
                key = (a, b, c)
                counts[key] = counts.get(key, 0) + 1
        s = 0
        for c in counts.values():
            if c == 6:      # C(4,2) -> exactly 4 points
                s += 1
            elif c >= 10:   # C(5,2) -> 5 or more collinear
                return -10**9
        return s

    def _valid(pts):
        if len(set(map(tuple, pts))) != 16:
            return False
        for x, y in pts:
            if abs(x) > 1000 or abs(y) > 1000:
                return False
        return True

    def sa_search(pts, seed, time_limit):
        """Simulated annealing with diverse move types."""
        nonlocal best_points, best_score
        rng = random.Random(seed)
        t0 = time.time()
        cur = [list(p) for p in pts]
        cur_score = score_points(cur)
        if cur_score > best_score:
            best_score = cur_score
            best_points = [list(p) for p in cur]
        T = 4.0
        T_min = 0.02
        while T > T_min:
            if time.time() - t0 > time_limit:
                break
            mtype = rng.random()
            if mtype < 0.45:
                i = rng.randint(0, 15)
                dx = rng.randint(-3, 3)
                dy = rng.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                cand = [list(p) for p in cur]
                cand[i][0] += dx
                cand[i][1] += dy
                if not _valid(cand):
                    continue
            elif mtype < 0.75:
                i = rng.randint(0, 15)
                ax = rng.randint(0, 1)
                old = cand = [list(p) for p in cur]
                nv = rng.randint(-1000, 1000)
                if nv == cur[i][ax]:
                    continue
                cand[i][ax] = nv
                if not _valid(cand):
                    continue
            else:
                i, j = rng.sample(range(16), 2)
                cand = [list(p) for p in cur]
                dx1 = rng.randint(-3, 3); dy1 = rng.randint(-3, 3)
                dx2 = rng.randint(-3, 3); dy2 = rng.randint(-3, 3)
                if (dx1 == 0 and dy1 == 0) or (dx2 == 0 and dy2 == 0):
                    continue
                cand[i][0] += dx1; cand[i][1] += dy1
                cand[j][0] += dx2; cand[j][1] += dy2
                if not _valid(cand):
                    continue
            new_score = score_points(cand)
            delta = new_score - cur_score
            if delta >= 0:
                cur = cand
                cur_score = new_score
            else:
                if rng.random() < math.exp(delta / T):
                    cur = cand
                    cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_points = [list(p) for p in cur]
            T *= 0.997
        return cur

    def polish(pts):
        """Exhaustive single-point relocation to nearby sites; accept improvements."""
        nonlocal best_points, best_score
        cur = [list(p) for p in pts]
        cur_score = score_points(cur)
        if cur_score > best_score:
            best_score = cur_score
            best_points = [list(p) for p in cur]
        for i in range(16):
            if time.time() > deadline:
                break
            x0, y0 = cur[i]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    cand = [list(p) for p in cur]
                    cand[i] = [x0 + dx, y0 + dy]
                    if not _valid(cand):
                        continue
                    ns = score_points(cand)
                    if ns > cur_score:
                        cur = cand
                        cur_score = ns
                        if cur_score > best_score:
                            best_score = cur_score
                            best_points = [list(p) for p in cur]
        return cur

    def gen_starts(seed):
        rng = random.Random(seed)
        configs = []
        configs.append([[x, y] for x in range(4) for y in range(4)])
        configs.append([[x + 1, y + 1] for x in range(4) for y in range(4)])
        # 5x4 subset
        base = [[x, y] for x in range(5) for y in range(4)]
        idx = list(range(20)); rng.shuffle(idx)
        configs.append([base[k] for k in idx[:16]])
        # 4x5 subset
        base = [[x, y] for x in range(4) for y in range(5)]
        idx = list(range(20)); rng.shuffle(idx)
        configs.append([base[k] for k in idx[:16]])
        # random box
        pts = []
        while len(pts) < 16:
            p = [rng.randint(-20, 20), rng.randint(-20, 20)]
            if p not in pts:
                pts.append(p)
        configs.append(pts)
        # two rows of 8
        configs.append([[x, 0] for x in range(8)] + [[x, 3] for x in range(8)])
        # near-grid perturbations
        base = [[x, y] for x in range(4) for y in range(4)]
        p = base[:]
        i = rng.randint(0, 15)
        p[i] = [p[i][0] + rng.randint(-3, 3), p[i][1] + rng.randint(-3, 3)]
        configs.append(p)
        base = [[x, y] for x in range(4) for y in range(4)]
        p = base[:]
        i = rng.randint(0, 15); j = rng.randint(0, 15)
        if i != j:
            p[i] = [p[i][0] + rng.randint(-2, 2), p[i][1] + rng.randint(-2, 2)]
            p[j] = [p[j][0] + rng.randint(-2, 2), p[j][1] + rng.randint(-2, 2)]
        configs.append(p)
        return configs

    starts = gen_starts(42)
    for k, st in enumerate(starts):
        if time.time() > deadline:
            break
        sa_search(st, seed=100 + k, time_limit=1.5)

    # Final exhaustive polish from the best found
    if best_points is not None:
        if time.time() < deadline - 5.0:
            polish(best_points)

    return [list(p) for p in best_points] if best_points else [[x, y] for x in range(4) for y in range(4)]
# EVOLVE-BLOCK-END

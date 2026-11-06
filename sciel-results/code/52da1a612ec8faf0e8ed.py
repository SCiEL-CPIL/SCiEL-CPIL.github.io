import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 135  # leave margin before 150s wall-clock kill

    # ---- fast collinearity: map from line (a,b,c) -> list of point indices ----
    def line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = math.gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, 0)
        a = dy // g          # normal (a,b)
        b = -dx // g
        c = -(a * p1[0] + b * p1[1])
        return (a, b, c)

    def build_lines(points):
        n = len(points)
        lines = {}
        for i in range(n):
            for j in range(i + 1, n):
                k = line_key(points[i], points[j])
                lines.setdefault(k, []).append(i)
        return lines

    def max_collinear(points):
        lines = build_lines(points)
        m = 2
        for idxs in lines.values():
            if len(idxs) > m:
                m = len(idxs)
        return m

    def has_4_collinear(points):
        for idxs in build_lines(points).values():
            if len(idxs) >= 4:
                return True
        return False

    def get_triples(points):
        n = len(points)
        triples = []
        for idxs in build_lines(points).values():
            L = len(idxs)
            if L >= 3:
                for a in range(L):
                    for b in range(a + 1, L):
                        for c in range(b + 1, L):
                            triples.append((idxs[a], idxs[b], idxs[c]))
        return triples

    def exact_independence_number(points, time_limit):
        """Max subset with no 3 collinear, via branch & bound. Score to minimize."""
        n = len(points)
        triples = get_triples(points)
        pairs_with = [[] for _ in range(n)]
        for i, j, k in triples:
            pairs_with[i].append((j, k))
            pairs_with[j].append((i, k))
            pairs_with[k].append((i, j))

        # order by descending "conflict degree"
        deg = [len(pairs_with[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: -deg[i])
        pos = {old: new for new, old in enumerate(order)}
        new_pw = [[] for _ in range(n)]
        for i, j, k in triples:
            ni, nj, nk = pos[i], pos[j], pos[k]
            new_pw[ni].append((nj, nk))
            new_pw[nj].append((ni, nk))
            new_pw[nk].append((ni, nj))

        best = [0]
        deadline = start_time + time_limit

        # greedy lower bound: include a point if it doesn't create a triple
        def greedy_lb():
            mask = 0
            cnt = 0
            for idx in range(n):
                ok = True
                for j, k in new_pw[idx]:
                    if (mask & (1 << j)) and (mask & (1 << k)):
                        ok = False
                        break
                if ok:
                    mask |= (1 << idx)
                    cnt += 1
            return cnt

        best[0] = greedy_lb()

        def bb(idx, mask, cur):
            if cur + (n - idx) <= best[0]:
                return
            if time.time() > deadline:
                return
            if idx == n:
                if cur > best[0]:
                    best[0] = cur
                return
            # try include
            ok = True
            for j, k in new_pw[idx]:
                if (mask & (1 << j)) and (mask & (1 << k)):
                    ok = False
                    break
            if ok:
                bb(idx + 1, mask | (1 << idx), cur + 1)
            # try exclude
            bb(idx + 1, mask, cur)

        bb(0, 0, 0)
        return best[0]

    # ---- generators ----
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def dedup_fill(pts, seen):
        unique = []
        for p in pts:
            t = (p[0], p[1])
            if t not in seen:
                seen.add(t)
                unique.append([p[0], p[1]])
        while len(unique) < 20:
            p = random_point()
            t = (p[0], p[1])
            if t not in seen:
                seen.add(t)
                unique.append([p[0], p[1]])
        return unique[:20]

    def gen_parabola():
        pts, seen = [], set()
        # base parabola gives no-3-collinear; perturb to create dense triples
        for x in range(-10, 10):
            px = x + random.randint(-2, 2)
            py = x * x + random.randint(-4, 4)
            pts.append([max(-100, min(100, px)), max(-100, min(100, py))])
        return dedup_fill(pts, seen)

    def gen_grid():
        pts, seen = [], set()
        for x in range(5):
            for y in range(4):
                px = x * 14 + random.randint(-4, 4)
                py = y * 14 + random.randint(-4, 4)
                pts.append([max(-100, min(100, px)), max(-100, min(100, py))])
        return dedup_fill(pts, seen)

    def gen_line_cluster():
        pts, seen = [], set()
        for _ in range(random.randint(5, 7)):
            kind = random.random()
            if kind < 0.34:
                y = random.randint(-80, 80)
                x0 = random.randint(-80, 80)
                for i in range(random.randint(3, 4)):
                    x = x0 + i * random.randint(15, 25)
                    if -100 <= x <= 100:
                        pts.append([x, y])
            elif kind < 0.67:
                x = random.randint(-80, 80)
                y0 = random.randint(-80, 80)
                for i in range(random.randint(3, 4)):
                    y = y0 + i * random.randint(15, 25)
                    if -100 <= y <= 100:
                        pts.append([x, y])
            else:
                m = random.choice([1, -1, 2, -2])
                b = random.randint(-50, 50)
                x0 = random.randint(-80, 80)
                for i in range(random.randint(3, 4)):
                    x = x0 + i * random.randint(15, 25)
                    y = m * x + b
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        pts.append([x, y])
        return dedup_fill(pts, seen)

    def gen_random():
        return dedup_fill([random_point() for _ in range(20)], set())

    generators = [gen_parabola, gen_grid, gen_line_cluster, gen_random]

    # ---- local search (hill climbing) ----
    def local_search(points, budget):
        pts = [p[:] for p in points]
        cur = max_collinear(pts)
        T = 8.0
        cool = 0.999
        steps = 0
        while steps < 4000 and time.time() < budget:
            steps += 1
            T *= cool
            idx = random.randint(0, 19)
            old = pts[idx][:]
            r = random.random()
            if r < 0.4:
                np_ = [old[0] + random.randint(-8, 8), old[1] + random.randint(-8, 8)]
            else:
                # snap onto a line through two other points to create triples
                j, k = random.sample([x for x in range(20) if x != idx], 2)
                dx = pts[k][0] - pts[j][0]
                dy = pts[k][1] - pts[j][1]
                g = math.gcd(abs(dx), abs(dy))
                if g > 0:
                    sx, sy = dx // g, dy // g
                else:
                    sx, sy = 1, 0
                t = random.randint(-6, 6)
                np_ = [pts[j][0] + t * sx, pts[j][1] + t * sy]
            np_[0] = max(-100, min(100, np_[0]))
            np_[1] = max(-100, min(100, np_[1]))
            if np_ == old:
                continue
            if any(pts[i] == np_ for i in range(20) if i != idx):
                continue
            pts[idx] = np_
            if has_4_collinear(pts):
                pts[idx] = old
                continue
            newc = max_collinear(pts)
            d = newc - cur
            if d > 0 or random.random() < math.exp(d / max(T, 1e-9)):
                cur = newc
            else:
                pts[idx] = old
        return pts, cur

    best_points = None
    best_score = None

    gen_cycle = 0
    while time.time() - start_time < timeout:
        gen = generators[gen_cycle % len(generators)]
        gen_cycle += 1
        points = gen()
        if has_4_collinear(points) or len(set(map(tuple, points))) < 20:
            continue

        # local search to improve (raise collinearity density)
        ls_budget = min(start_time + timeout - 20, time.time() + 12)
        points, _ = local_search(points, ls_budget)
        if has_4_collinear(points) or len(set(map(tuple, points))) < 20:
            continue

        # exact score (branch & bound) with a shrinking time limit
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        if remaining < 3:
            break
        tlimit = min(remaining - 2, 25)
        score = exact_independence_number(points, tlimit)

        if best_score is None or score < best_score:
            best_score = score
            best_points = [p[:] for p in points]

    # fallback valid config (parabola: no 3 collinear, no 4 collinear)
    if best_points is None:
        best_points = [[x, x * x] for x in range(-10, 10)]

    if has_4_collinear(best_points) or len(set(map(tuple, best_points))) < 20:
        best_points = [[x, x * x] for x in range(-10, 10)]

    return best_points

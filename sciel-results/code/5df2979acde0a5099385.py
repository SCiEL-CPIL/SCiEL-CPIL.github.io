import random
import time
from itertools import combinations

def construct():
    t0 = time.time()
    hard_deadline = 140  # 10s safety margin before 150s

    def now():
        return time.time()

    def collinearity(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) == (r[0] - p[0]) * (q[1] - p[1])

    def triples_mask(points):
        """Return a list of 20-bit masks, one per collinear triple."""
        n = len(points)
        masks = []
        for a, b, c in combinations(range(n), 3):
            if collinearity(points[a], points[b], points[c]):
                masks.append((1 << a) | (1 << b) | (1 << c))
        return masks

    def has_four_collinear(points):
        n = len(points)
        # count collinear triples and see if any line contains 4
        # faster: for each pair, count points on the line
        for i in range(n):
            for j in range(i + 1, n):
                cnt = 2
                for k in range(n):
                    if k != i and k != j and collinearity(points[i], points[j], points[k]):
                        cnt += 1
                if cnt >= 4:
                    return True
        return False

    def exact_alpha(points):
        """Exact max subset with no three collinear via branch & bound on triple hypergraph."""
        n = len(points)
        masks = triples_mask(points)
        if not masks:
            return n
        # Build adjacency: adj[i] = set of points that with i form a triple with some other point
        # Better: for each point, list of (other1, other2) pairs
        # For B&B: process points in order, maintain selected set
        # Key optimization: for each point, precompute which other points it conflicts with
        # Point i conflicts with pair (j,k) if {i,j,k} is a triple
        # When we add point i, we must ensure no two of its partners are both selected

        # Sort points by degree (number of triples they're in) descending
        deg = [0] * n
        for m in masks:
            for bit in range(n):
                if m & (1 << bit):
                    deg[bit] += 1
        order = sorted(range(n), key=lambda x: -deg[x])
        pos = [0] * n
        for new_i, old_i in enumerate(order):
            pos[old_i] = new_i

        # Rebuild conflict structure in new order
        # conflicts[new_i] = list of (new_j, new_k) such that {old_i, old_j, old_k} is a triple
        conflicts = [[] for _ in range(n)]
        for m in masks:
            a, b, c = [], [], []
            for bit in range(n):
                if m & (1 << bit):
                    a.append(bit)
            na, nb, nc = pos[a[0]], pos[a[1]], pos[a[2]]
            conflicts[na].append((nb, nc))
            conflicts[nb].append((na, nc))
            conflicts[nc].append((na, nb))

        best = 0

        def bb(idx, selected_mask, cnt):
            nonlocal best
            # Upper bound
            if cnt + (n - idx) <= best:
                return
            if idx == n:
                if cnt > best:
                    best = cnt
                return
            # Try including
            ok = True
            for j, k in conflicts[idx]:
                if (selected_mask & (1 << j)) and (selected_mask & (1 << k)):
                    ok = False
                    break
            if ok:
                bb(idx + 1, selected_mask | (1 << idx), cnt + 1)
            # Try excluding
            bb(idx + 1, selected_mask, cnt)

        bb(0, 0, 0)
        return best

    def valid_config(points):
        if len(points) != 20:
            return False
        if len(set(tuple(p) for p in points)) < 20:
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        if has_four_collinear(points):
            return False
        return True

    def random_valid_point(existing):
        while True:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in existing:
                return p

    def initial_config(strategy):
        pts = []
        used = set()
        if strategy == 'parabola':
            # Points near a parabola
            for i in range(20):
                x = i - 10
                y = x * x // 2 + random.randint(-3, 3)
                x += random.randint(-1, 1)
                y = max(-100, min(100, y))
                x = max(-100, min(100, x))
                p = (x, y)
                if p not in used:
                    pts.append(p)
                    used.add(p)
            while len(pts) < 20:
                p = random_valid_point(used)
                pts.append(p)
                used.add(p)
        elif strategy == 'grid':
            for i in range(5):
                for j in range(4):
                    x = i * 20 + random.randint(-2, 2)
                    y = j * 20 + random.randint(-2, 2)
                    x = max(-100, min(100, x))
                    y = max(-100, min(100, y))
                    p = (x, y)
                    if p not in used:
                        pts.append(p)
                        used.add(p)
            while len(pts) < 20:
                p = random_valid_point(used)
                pts.append(p)
                used.add(p)
        elif strategy == 'lines':
            # Cluster on a few lines
            lines = []
            for _ in range(5):
                m = random.choice([0, 1, -1, 2, -2])
                b = random.randint(-40, 40)
                lines.append((m, b))
            for _ in range(20):
                m, b = random.choice(lines)
                x = random.randint(-60, 60)
                y = m * x + b
                y = max(-100, min(100, y))
                p = (x, y)
                if p not in used:
                    pts.append(p)
                    used.add(p)
            while len(pts) < 20:
                p = random_valid_point(used)
                pts.append(p)
                used.add(p)
        else:
            # Random
            for _ in range(20):
                p = random_valid_point(used)
                pts.append(p)
                used.add(p)

        pts = [list(p) for p in pts[:20]]
        # Fix 4-collinear issues
        for _ in range(500):
            if not has_four_collinear(pts):
                break
            # Find a 4-collinear line and move one point
            n = 20
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    cnt_pts = []
                    for k in range(n):
                        if k != i and k != j and collinearity(pts[i], pts[j], pts[k]):
                            cnt_pts.append(k)
                    if len(cnt_pts) >= 2:
                        # Move one of the collinear points
                        victim = random.choice(cnt_pts)
                        old = pts[victim][:]
                        for _ in range(100):
                            nx = old[0] + random.randint(-10, 10)
                            ny = old[1] + random.randint(-10, 10)
                            nx = max(-100, min(100, nx))
                            ny = max(-100, min(100, ny))
                            p = (nx, ny)
                            if p not in [tuple(q) for q in pts if q is not pts[victim]]:
                                pts[victim] = [nx, ny]
                                if not has_four_collinear(pts):
                                    moved = True
                                    break
                        if not moved:
                            pts[victim] = old
                        break
                if moved:
                    break
        return pts if valid_config(pts) else None

    def hill_climb(points, budget):
        """Local search: move one point at a time to reduce alpha."""
        pts = [p[:] for p in points]
        cur_alpha = exact_alpha(pts)
        best_alpha = cur_alpha
        best_pts = [p[:] for p in pts]

        steps = 0
        while now() < budget and steps < 500:
            steps += 1
            improved = False
            for i in range(20):
                if now() > budget:
                    break
                orig = pts[i][:]
                # Try moving point i to a new random location
                for _ in range(8):
                    if now() > budget:
                        break
                    if random.random() < 0.5:
                        nx = orig[0] + random.randint(-8, 8)
                        ny = orig[1] + random.randint(-8, 8)
                    else:
                        nx = random.randint(-100, 100)
                        ny = random.randint(-100, 100)
                    nx = max(-100, min(100, nx))
                    ny = max(-100, min(100, ny))
                    p = (nx, ny)
                    if p in [tuple(q) for q in pts if q is not pts[i]]:
                        continue
                    pts[i] = [nx, ny]
                    if has_four_collinear(pts):
                        pts[i] = orig
                        continue
                    new_alpha = exact_alpha(pts)
                    if new_alpha < cur_alpha:
                        cur_alpha = new_alpha
                        if cur_alpha < best_alpha:
                            best_alpha = cur_alpha
                            best_pts = [q[:] for q in pts]
                        improved = True
                        break
                    else:
                        pts[i] = orig
                if improved:
                    break
            if not improved:
                break
        return best_pts, best_alpha

    def sa_search(points, budget):
        """Simulated annealing."""
        pts = [p[:] for p in points]
        cur_alpha = exact_alpha(pts)
        best_alpha = cur_alpha
        best_pts = [p[:] for p in pts]
        T = 5.0

        iters = 0
        while now() < budget and iters < 300:
            iters += 1
            i = random.randint(0, 19)
            orig = pts[i][:]
            if random.random() < 0.6:
                nx = orig[0] + random.randint(-6, 6)
                ny = orig[1] + random.randint(-6, 6)
            else:
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
            nx = max(-100, min(100, nx))
            ny = max(-100, min(100, ny))
            p = (nx, ny)
            if p in [tuple(q) for q in pts if q is not pts[i]]:
                continue
            pts[i] = [nx, ny]
            if has_four_collinear(pts):
                pts[i] = orig
                continue
            new_alpha = exact_alpha(pts)
            delta = new_alpha - cur_alpha
            if delta <= 0 or random.random() < min(1.0, max(0.0, 1.0 - delta / (T + 0.01))):
                cur_alpha = new_alpha
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best_pts = [q[:] for q in pts]
            else:
                pts[i] = orig
            T *= 0.997

        return best_pts, best_alpha

    # Main
    best_pts = None
    best_alpha = 21

    strategies = ['parabola', 'grid', 'lines', 'random', 'parabola', 'grid', 'lines', 'random']
    for strat in strategies:
        if now() > hard_deadline - 30:
            break
        cfg = initial_config(strat)
        if cfg is None:
            continue
        a = exact_alpha(cfg)
        if a < best_alpha:
            best_alpha = a
            best_pts = [p[:] for p in cfg]

    if best_pts is None:
        cfg = initial_config('random')
        if cfg is None:
            # Fallback: simple valid config
            pts = []
            used = set()
            for _ in range(20):
                while True:
                    p = (random.randint(-100, 100), random.randint(-100, 100))
                    if p not in used:
                        used.add(p)
                        pts.append(list(p))
                        break
            for _ in range(200):
                if not has_four_collinear(pts):
                    break
                for i in range(20):
                    for j in range(i+1, 20):
                        cnt = [k for k in range(20) if k != i and k != j and collinearity(pts[i], pts[j], pts[k])]
                        if len(cnt) >= 2:
                            v = random.choice(cnt)
                            for _ in range(50):
                                nx = pts[v][0] + random.randint(-15, 15)
                                ny = pts[v][1] + random.randint(-15, 15)
                                nx = max(-100, min(100, nx))
                                ny = max(-100, min(100, ny))
                                if (nx, ny) not in [tuple(q) for q in pts if q is not pts[v]]:
                                    pts[v] = [nx, ny]
                                    if not has_four_collinear(pts):
                                        break
                            break
                    break
            best_pts = pts
            best_alpha = exact_alpha(pts)
        else:
            best_pts = cfg
            best_alpha = exact_alpha(cfg)

    # Refine with hill climbing
    if now() < hard_deadline - 10:
        hp, ha = hill_climb(best_pts, hard_deadline - 15)
        if ha < best_alpha:
            best_alpha = ha
            best_pts = hp

    # Refine with SA
    if now() < hard_deadline - 5:
        sp, sa = sa_search(best_pts, hard_deadline - 2)
        if sa < best_alpha:
            best_alpha = sa
            best_pts = sp

    # Final validation
    if not valid_config(best_pts):
        # Emergency fallback
        pts = []
        used = set()
        for _ in range(20):
            while True:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in used:
                    used.add(p)
                    pts.append(list(p))
                    break
        best_pts = pts

    return best_pts

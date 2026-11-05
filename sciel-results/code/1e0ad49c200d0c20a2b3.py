"""Rewritten: structural line-packing + hypergraph B&B + line-aware local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    g = gcd(abs(A), gcd(abs(B), abs(C))) if (A != 0 or B != 0) else 1
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _on_line(lk, p):
    return lk[0] * p[0] + lk[1] * p[1] + lk[2] == 0

def _count_collinear_triples(points):
    n = len(points)
    triples = 0
    for i in range(n):
        for j in range(i + 1, n):
            lk = _line_key(points[i], points[j])
            cnt = 0
            for k in range(j + 1, n):
                if _on_line(lk, points[k]):
                    cnt += 1
            if cnt >= 2:
                triples += 1
    return triples

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            lk = _line_key(points[i], points[j])
            cnt = 2
            for k in range(j + 1, n):
                if _on_line(lk, points[k]):
                    cnt += 1
                    if cnt >= 4:
                        return True
    return False

def _max_3free_subset(points, deadline=None):
    n = len(points)
    best = [0]

    # Precompute collinearity triplets
    collinear = [[[False] * n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            lk = _line_key(points[i], points[j])
            for k in range(j + 1, n):
                if _on_line(lk, points[k]):
                    collinear[i][j][k] = True
                    collinear[j][i][k] = True
                    collinear[k][i][j] = True
                    collinear[i][k][j] = True
                    collinear[j][k][i] = True
                    collinear[k][j][i] = True

    def dfs(idx, current, current_size):
        if deadline and time.time() > deadline:
            return
        if idx >= n:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + (n - idx) <= best[0]:
            return

        # Try adding idx
        can_add = True
        cur = current
        for a in range(len(cur)):
            for b in range(a + 1, len(cur)):
                if collinear[idx][cur[a]][cur[b]]:
                    can_add = False
                    break
            if not can_add:
                break

        if can_add:
            cur.append(idx)
            dfs(idx + 1, cur, current_size + 1)
            cur.pop()

        # Try not adding idx
        dfs(idx + 1, cur, current_size)

    dfs(0, [], 0)
    return best[0]

def _find_third_on_line(lk, used_set, rng):
    A, B, C = lk
    candidates = []
    if B != 0:
        for x in range(-100, 101):
            num = -A * x - C
            if num % B == 0:
                y = num // B
                if -100 <= y <= 100:
                    candidates.append((x, y))
    elif A != 0:
        for y in range(-100, 101):
            num = -B * y - C
            if num % A == 0:
                x = num // A
                if -100 <= x <= 100:
                    candidates.append((x, y))
    rng.shuffle(candidates)
    for cand in candidates:
        if cand not in used_set:
            return cand
    return None

def _try_line_packing(seed, deadline):
    """
    Deliberately build 3-point lines to maximize collinear triples.
    Greedy: pick pairs, extend to triples, ensure no 4-collinear.
    """
    rng = random.Random(seed)
    n = 20
    points = []
    used = set()

    def add_point(p):
        points.append(list(p))
        used.add(tuple(p))

    # Seed with 2 random points
    while len(points) < 2:
        x, y = rng.randint(-90, 90), rng.randint(-90, 90)
        if (x, y) not in used:
            add_point((x, y))

    # Greedy: repeatedly pick a pair, try to add a third on their line
    attempts = 0
    max_attempts = 3000
    while len(points) < n and attempts < max_attempts and time.time() < deadline - 10:
        attempts += 1
        i, j = rng.sample(range(len(points)), 2)
        lk = _line_key(points[i], points[j])

        # Count existing points on this line
        cnt = sum(1 for p in points if _on_line(lk, p))
        if cnt > 2:
            continue
        if cnt < 2:
            continue

        third = _find_third_on_line(lk, used, rng)
        if third is None:
            continue

        # Validate: adding third doesn't create 4-collinear on any other line
        valid = True
        for a in range(len(points)):
            for b in range(a + 1, len(points)):
                if a == i and b == j:
                    continue
                lk2 = _line_key(points[a], points[b])
                if _on_line(lk2, third):
                    cnt2 = sum(1 for p in points if _on_line(lk2, p))
                    if cnt2 >= 3:
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            continue

        add_point(third)

    # Fill remaining with random points that don't create 4-collinear
    while len(points) < n and time.time() < deadline - 10:
        x, y = rng.randint(-90, 90), rng.randint(-90, 90)
        if (x, y) in used:
            continue
        # Check no 4-collinear
        valid = True
        for a in range(len(points)):
            for b in range(a + 1, len(points)):
                lk = _line_key(points[a], points[b])
                if _on_line(lk, (x, y)):
                    cnt = sum(1 for p in points if _on_line(lk, p))
                    if cnt >= 3:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            add_point((x, y))

    if len(points) != n or _has_four_collinear(points):
        return None
    return points

def _local_search_line_aware(points, deadline):
    """
    Local search: move points to create more collinear triples while
    maintaining no-4-collinear.
    """
    points = [p[:] for p in points]
    current_triples = _count_collinear_triples(points)
    rng = random.Random(hash(tuple(map(tuple, points))))

    def try_move(i, new_pos):
        orig = points[i][:]
        points[i] = list(new_pos)
        if _has_four_collinear(points):
            points[i] = orig
            return False, current_triples
        nt = _count_collinear_triples(points)
        if nt <= current_triples:
            points[i] = orig
            return False, nt
        return True, nt

    while time.time() < deadline:
        improved = False
        n = len(points)
        for i in range(n):
            if time.time() > deadline:
                break
            # Try moving point i onto a line defined by pair (j,k)
            for j in range(n):
                if j == i:
                    continue
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    lk = _line_key(points[j], points[k])
                    if _on_line(lk, points[i]):
                        continue
                    # Check if this line has exactly 2 points (j,k)
                    cnt = sum(1 for m in range(n) if m != i and _on_line(lk, points[m]))
                    if cnt != 2:
                        continue
                    new_pos = _find_third_on_line(lk, set(tuple(p) for m in range(n) if m != i for p in [points[m]]), rng)
                    if new_pos is None:
                        continue
                    ok, nt = try_move(i, new_pos)
                    if ok:
                        current_triples = nt
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 118

    best_points = None
    best_triples = -1

    # Phase 1: Line-packing with multiple seeds
    for seed in range(120):
        if time.time() > deadline - 40:
            break
        pts = _try_line_packing(seed, deadline)
        if pts is None:
            continue
        t = _count_collinear_triples(pts)
        if t > best_triples:
            best_triples = t
            best_points = [p[:] for p in pts]

    # Phase 2: Local search
    if best_points is not None and time.time() < deadline - 20:
        improved = _local_search_line_aware(best_points, deadline - 5)
        if len(set(tuple(p) for p in improved)) == 20 and not _has_four_collinear(improved):
            t = _count_collinear_triples(improved)
            if t > best_triples:
                best_triples = t
                best_points = [p[:] for p in improved]

    # Phase 3: More local search passes
    if best_points is not None and time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            improved = _local_search_line_aware(best_points, deadline - 3)
            if len(set(tuple(p) for p in improved)) == 20 and not _has_four_collinear(improved):
                t = _count_collinear_triples(improved)
                if t > best_triples:
                    best_triples = t
                    best_points = [p[:] for p in improved]
                else:
                    break

    if best_points is not None:
        return best_points
    # Fallback: parabola
    return [[x, x * x] for x in range(-10, 10)]
# EVOLVE-BLOCK-END

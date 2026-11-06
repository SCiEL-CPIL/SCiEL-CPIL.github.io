import time
import random
import itertools
from math import gcd

DEADLINE = 143.0  # safety margin before 150s kill

def _line_key(p1, p2):
    """Canonical key for line through p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    a = -dy
    b = dx
    c = dx * y1 - dy * x1
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    g2 = gcd(abs(a), abs(b))
    if g2 > 1:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _build_lines(points):
    """Return line_map: key -> sorted list of point indices on that line."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return line_map

def _has_four_collinear(line_map):
    for pts in line_map.values():
        if len(pts) >= 4:
            return True
    return False

def _max_cap_free(points, time_limit=5.0):
    """Max subset with no three collinear via branch and bound (fast)."""
    n = len(points)
    start_time = time.time()

    line_map = _build_lines(points)
    # triples: list of (i,j,k) with i<j<k
    triples = []
    for pts in line_map.values():
        if len(pts) >= 3:
            pl = sorted(pts)
            for t in itertools.combinations(pl, 3):
                triples.append(t)
    if not triples:
        return n

    # triple_conflict[i] = set of (j,k) with j<k, both != i, {i,j,k} triple
    triple_conflict = [set() for _ in range(n)]
    for (i, j, k) in triples:
        triple_conflict[i].add((j, k))
        triple_conflict[j].add((i, k))
        triple_conflict[k].add((i, j))

    best = [0]
    best_set = [0]
    # Precompute pair masks for quick checking
    pair_mask = {}
    for (i, j, k) in triples:
        pair_mask[(i, j)] = 1 << k
        pair_mask[(j, k)] = 1 << i
        pair_mask[(i, k)] = 1 << j

    # For each i, list of (j, k, mask_of_j_and_k) sorted by j
    conflict_list = []
    for i in range(n):
        lst = []
        for (j, k) in triple_conflict[i]:
            if j > k:
                j, k = k, j
            lst.append((j, k, (1 << j) | (1 << k)))
        lst.sort()
        conflict_list.append(lst)

    def is_valid_add(current_mask, idx):
        for (j, k, m) in conflict_list[idx]:
            if current_mask & m == m:
                return False
        return True

    # Upper bound: count of remaining indices that can still be added
    # (no triple fully in S). Use a simple bound: remaining count.
    # Stronger: for each remaining candidate, it's addable only if not blocked.

    # Memoization: (next_idx, current_mask) -> best additional size
    memo = {}

    def dfs(current_mask, next_idx):
        if time.time() - start_time > time_limit:
            return 0
        cur_size = current_mask.bit_count()
        # Upper bound
        ub = cur_size + (n - next_idx)
        if ub <= best[0]:
            return 0
        # Try to build greedily to improve lower bound
        # Check memo
        key = (next_idx, current_mask)
        if key in memo:
            return memo[key]
        best_add = 0
        for idx in range(next_idx, n):
            if time.time() - start_time > time_limit:
                break
            if is_valid_add(current_mask, idx):
                r = dfs(current_mask | (1 << idx), idx + 1)
                if r > best_add:
                    best_add = r
            else:
                # skip
                pass
            # Also try excluding (implicit by continuing loop)
        # The loop already handles exclude by not taking idx.
        # But we need to also consider the case of excluding all remaining.
        # Actually the recursion: for each idx we either include (if valid) or move on.
        # The "move on" is the exclude branch. So best_add is max over choices.
        # But we also need the base: excluding everything gives 0.
        memo[key] = best_add
        total = cur_size + best_add
        if total > best[0]:
            best[0] = total
            best_set[0] = current_mask
        return best_add

    dfs(0, 0)
    return best[0]

def _score(points, time_limit=8.0):
    """Compute the max cap-free subset size. Returns 999 if invalid."""
    line_map = _build_lines(points)
    if _has_four_collinear(line_map):
        return 999
    return _max_cap_free(points, time_limit=time_limit)

def _generate_grid_perturbed(seed, scale=1):
    """Generate a perturbed 5x4 grid."""
    rng = random.Random(seed)
    base_x = list(range(5))
    base_y = list(range(4))
    points = []
    for i in range(5):
        for j in range(4):
            x = base_x[i] * scale + rng.randint(-3, 3)
            y = base_y[j] * (scale * 2) + rng.randint(-2, 2)
            x = max(-100, min(100, x))
            y = max(-100, min(100, y))
            points.append([x, y])
    if len(set(tuple(p) for p in points)) != 20:
        return None
    return points

def _generate_parabola_perturbed(seed):
    """Generate points near a parabola with perturbation."""
    rng = random.Random(seed)
    points = []
    for x in range(-10, 10):
        y = x * x + rng.randint(-2, 2)
        y = max(-100, min(100, y))
        points.append([x, y])
    if len(set(tuple(p) for p in points)) != 20:
        return None
    return points

def _generate_random(seed, spread=60):
    """Generate random points."""
    rng = random.Random(seed)
    points = []
    for _ in range(20):
        x = rng.randint(-spread, spread)
        y = rng.randint(-spread, spread)
        points.append([x, y])
    if len(set(tuple(p) for p in points)) != 20:
        return None
    return points

def _generate_lattice(seed):
    """Generate points on a lattice with jitter."""
    rng = random.Random(seed)
    points = []
    for i in range(5):
        for j in range(4):
            x = i * 8 + rng.randint(-2, 2)
            y = j * 12 + rng.randint(-2, 2)
            x = max(-100, min(100, x))
            y = max(-100, min(100, y))
            points.append([x, y])
    if len(set(tuple(p) for p in points)) != 20:
        return None
    return points

def _local_search(points, time_limit=15.0):
    """Local search: try moving points to reduce score."""
    start_time = time.time()
    current = [p[:] for p in points]
    current_score = _score(current)
    if current_score >= 999:
        return current, current_score

    best = [p[:] for p in current]
    best_score = current_score

    while time.time() - start_time < time_limit:
        improved = False
        # Try moving each point by a larger range
        for i in range(20):
            if time.time() - start_time > time_limit:
                break
            ox, oy = current[i]
            # Try moves in a larger neighborhood
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx = ox + dx
                    ny = oy + dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    current[i] = [nx, ny]
                    # Check distinctness
                    if len(set(tuple(p) for p in current)) != 20:
                        continue
                    s = _score(current, time_limit=3.0)
                    if s < best_score:
                        best_score = s
                        best = [p[:] for p in current]
                        improved = True
                    current[i] = [ox, oy]
        # Also try swapping two points' positions
        for i in range(20):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, 20):
                if time.time() - start_time > time_limit:
                    break
                current[i], current[j] = current[j][:], current[i][:]
                s = _score(current, time_limit=3.0)
                if s < best_score:
                    best_score = s
                    best = [p[:] for p in current]
                    improved = True
                current[i], current[j] = current[j][:], current[i][:]
        if not improved:
            break

    return best, best_score

def construct():
    start_time = time.time()

    best_points = None
    best_score = 999

    def try_config(pts, do_local=True):
        nonlocal best_points, best_score
        if pts is None:
            return
        if len(set(tuple(p) for p in pts)) != 20:
            return
        line_map = _build_lines(pts)
        if _has_four_collinear(line_map):
            return
        s = _score(pts, time_limit=8.0)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in pts]
            # Try local search if promising and time remains
            if s <= 18 and time.time() - start_time < DEADLINE - 25:
                remaining = min(12.0, DEADLINE - 20 - (time.time() - start_time))
                if remaining > 2.0:
                    refined, refined_score = _local_search(pts, time_limit=remaining)
                    if refined_score < best_score:
                        best_score = refined_score
                        best_points = refined

    # Strategy 1: Perturbed grids
    for seed in range(80):
        if time.time() - start_time > DEADLINE - 20:
            break
        for scale in [1, 2]:
            pts = _generate_grid_perturbed(seed * 10 + scale, scale)
            try_config(pts, do_local=True)

    # Strategy 2: Parabola perturbed
    for seed in range(30):
        if time.time() - start_time > DEADLINE - 15:
            break
        pts = _generate_parabola_perturbed(seed)
        try_config(pts, do_local=True)

    # Strategy 3: Random points
    for seed in range(50):
        if time.time() - start_time > DEADLINE - 10:
            break
        for spread in [40, 60, 80]:
            pts = _generate_random(seed * 10 + spread, spread)
            try_config(pts, do_local=True)

    # Strategy 4: Lattice
    for seed in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
        pts = _generate_lattice(seed)
        try_config(pts, do_local=True)

    # Fallback: parabola (always valid, score 20)
    if best_points is None or best_score >= 20:
        best_points = [[x, x * x] for x in range(-10, 10)]

    return best_points

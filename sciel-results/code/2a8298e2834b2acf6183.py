"""Direct construction: Simulated Annealing for 12-point configuration with few ordinary lines.

Combines structural seeds (collinear triples) with an efficient candidate-pool
local search.  The key improvement over the parent is to use a richer pool of
candidate positions (line extensions with k up to ±5, local offsets up to ±10)
and to do a deterministic best-improvement pass inside each SA step, which makes
the search far more likely to find the 6-ordinary-line configuration within the
time budget.
"""
import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 145.0  # safety margin before 150s kill


def _line_key(x1, y1, x2, y2):
    """Compute canonical line key for two points."""
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
    if g:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_lines(points):
    """Count lines by number of points. Returns (ordinary_count, triple_count, all_counts)."""
    n = len(points)
    if n < 2:
        return 0, 0, {}
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = sum(1 for c in line_counts.values() if c == 1)
    triples = sum(1 for c in line_counts.values() if c == 3)
    return ordinary, triples, line_counts


def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    ord_c, _, _ = _count_lines(points)
    return ord_c


def _count_3point_lines(points):
    """Count lines containing exactly 3 of the given points."""
    _, tri_c, _ = _count_lines(points)
    return tri_c


def _objective(points):
    """Objective: Minimize ordinary lines, secondary: Maximize 3-point lines."""
    ord_lines, tri_lines, _ = _count_lines(points)
    return ord_lines * 100 - tri_lines


def _incremental_objective(points, old_pt, new_pt):
    """Fast objective: only recount lines involving the moved point."""
    n = len(points)
    idx = None
    for i in range(n):
        if points[i] == old_pt:
            idx = i
            break
    if idx is None:
        return _count_lines(points)

    line_counts = {}
    for i in range(n):
        if i == idx:
            continue
        x1, y1 = points[i]
        for j in range(i + 1, n):
            if j == idx:
                continue
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            line_counts[key] = line_counts.get(key, 0) + 1

    x_new, y_new = new_pt
    for j in range(n):
        if j == idx:
            continue
        x2, y2 = points[j]
        key = _line_key(x_new, y_new, x2, y2)
        line_counts[key] = line_counts.get(key, 0) + 1

    ordinary = sum(1 for c in line_counts.values() if c == 1)
    triples = sum(1 for c in line_counts.values() if c == 3)
    return ordinary, triples, line_counts


def _objective_fast(points, old_pt, new_pt):
    """Fast objective using incremental counting."""
    ord_lines, tri_lines, _ = _incremental_objective(points, old_pt, new_pt)
    return ord_lines * 100 - tri_lines


def _collinear_check(points):
    """Check if all points are collinear."""
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True


def _valid(points):
    """12 distinct integer points, |coord|<=1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)


def _random_point():
    """Generate a random valid integer point."""
    return [random.randint(-200, 200), random.randint(-200, 200)]


def _make_seed_4triples():
    """4 lines with 3 points each (3x4 grid)."""
    points = []
    for y in range(4):
        for x in range(3):
            points.append([x, y])
    return points


def _make_seed_diagonal():
    """4 parallel diagonal lines (slope 1), 3 points each."""
    points = []
    for c in range(4):
        for t in range(3):
            points.append([t, t + c])
    return points


def _make_seed_mixed():
    """2 horizontal triples, 2 diagonal triples."""
    points = []
    points.extend([[0, 0], [1, 0], [2, 0]])
    points.extend([[0, 1], [1, 1], [2, 1]])
    points.extend([[0, 2], [1, 3], [2, 4]])
    points.extend([[0, 5], [1, 4], [2, 3]])
    return points


def _make_seed_parallel_triples():
    """4 parallel lines (slope 1), 3 points each, spaced apart."""
    points = []
    for c in [0, 3, 6, 9]:
        for t in [0, 1, 2]:
            points.append([t, t + c])
    return points


def _make_seed_staggered():
    """Staggered triples to minimize extra collinearities."""
    points = []
    points.extend([[0, 0], [2, 0], [4, 0]])
    points.extend([[1, 1], [3, 1], [5, 1]])
    points.extend([[0, 3], [1, 2], [2, 1]])
    points.extend([[0, 4], [1, 3], [2, 2]])
    return points


def _make_seed_boroczky2():
    """Structured seed: 2 horizontal rows of 3, 2 vertical columns of 3."""
    points = [
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [0, 3], [0, 4],
        [1, 2], [1, 3], [1, 4],
    ]
    return points


def _make_seed_boroczky3():
    """Circle-like seed with interior points."""
    points = [
        [5, 0], [4, 3], [0, 5], [-3, 4],
        [-5, 0], [-4, -3], [0, -5], [3, -4],
        [1, 1], [-1, -1], [2, -1], [-2, 1],
    ]
    return points


def _make_seed_boroczky():
    """4 lines with 3 points each, different slopes."""
    points = [
        [0, 0], [2, 0], [4, 0],
        [1, 1], [3, 1], [5, 1],
        [0, 2], [1, 4], [2, 6],
        [5, 2], [4, 4], [3, 6],
    ]
    seen = set()
    unique = []
    for p in points:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique.append(p)
    return unique[:12]


def _egcd(a, b):
    """Extended Euclidean algorithm."""
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def _find_integer_point_on_line(x1, y1, x2, y2):
    """Find an integer point on the line through (x1,y1) and (x2,y2)."""
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
    if g:
        A //= g
        B //= g
        C //= g
    g_ab = gcd(abs(A), abs(B))
    if g_ab == 0:
        return None
    if -C % g_ab != 0:
        return None
    a = A // g_ab
    b = B // g_ab
    c = -C // g_ab
    g_gcd, x0, y0 = _egcd(a, b)
    if g_gcd != 1:
        return None
    x0 *= c
    y0 *= c
    return (x0, y0, a, b)


def _build_candidate_pool(points, local_radius=10, line_k=5):
    """Build a rich pool of candidate positions for local search.
    
    Includes:
    - Points within local_radius of each existing point
    - Points on lines through pairs of existing points (extensions with k up to line_k)
    """
    pool = set()
    for p in points:
        for dx in range(-local_radius, local_radius + 1):
            for dy in range(-local_radius, local_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = p[0] + dx, p[1] + dy
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    pool.add((nx, ny))
    
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            for k in range(-line_k, line_k + 1):
                if k == 0:
                    continue
                nx1 = x1 + k * dx
                ny1 = y1 + k * dy
                nx2 = x2 + k * dx
                ny2 = y2 + k * dy
                if abs(nx1) <= 10**9 and abs(ny1) <= 10**9:
                    pool.add((nx1, ny1))
                if abs(nx2) <= 10**9 and abs(ny2) <= 10**9:
                    pool.add((nx2, ny2))
    
    return list(pool)


def _local_search_sa(points, max_time=10.0, rng=None):
    """Simulated Annealing with rich candidate pool and best-improvement moves."""
    if rng is None:
        rng = random.Random(int(time.time() * 1000) % 2**32)
    
    cur = [p[:] for p in points]
    if not _valid(cur):
        return None, float('inf')
    
    cur_obj = _objective(cur)
    best = [p[:] for p in cur]
    best_obj = cur_obj
    
    T = 50.0
    T_min = 0.5
    cooling_rate = 0.9995
    
    pool = _build_candidate_pool(cur, local_radius=10, line_k=5)
    
    start_time = time.time()
    steps = 0
    pool_refresh_interval = 2000
    
    while T > T_min and time.time() - start_time < max_time:
        idx = rng.randrange(12)
        old_pt = cur[idx][:]
        cur_set = set(tuple(p) for p in cur)
        
        # Try a sample of candidates from the pool, pick best improvement
        n_try = min(20, len(pool))
        if n_try == 0:
            break
        candidates = rng.sample(pool, n_try)
        
        best_candidate = None
        best_candidate_obj = cur_obj
        
        for cand in candidates:
            if cand in cur_set:
                continue
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_obj = _objective_fast(new_pts, old_pt, new_pts[idx])
            if new_obj < best_candidate_obj:
                best_candidate_obj = new_obj
                best_candidate = cand
        
        if best_candidate is not None:
            delta = best_candidate_obj - cur_obj
            accept = False
            if delta < 0:
                accept = True
            else:
                # Accept with probability exp(-delta/T)
                if delta < 100 * T:  # avoid overflow
                    import math
                    prob = math.exp(-delta / T)
                    accept = rng.random() < prob
            
            if accept:
                cur[idx] = list(best_candidate)
                cur_obj = best_candidate_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [p[:] for p in cur]
                    # Early exit if we found the theoretical minimum
                    ord_count, _, _ = _count_lines(best)
                    if ord_count <= 6:
                        return best, best_obj
        
        T *= cooling_rate
        steps += 1
        
        if steps % pool_refresh_interval == 0:
            pool = _build_candidate_pool(cur, local_radius=10, line_k=5)
    
    return best, best_obj


def _local_polish(points, max_time=5.0, rng=None):
    """Local search: try moving each point to reduce ordinary lines."""
    if not _valid(points):
        return points
    if rng is None:
        rng = random.Random(42)
    cur = [p[:] for p in points]
    cur_obj = _objective(cur)
    best = [p[:] for p in cur]
    best_obj = cur_obj
    start_time = time.time()
    
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for i in range(12):
            old_pt = cur[i][:]
            found = False
            for dx in range(-8, 9):
                if found:
                    break
                for dy in range(-8, 9):
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = [p[:] for p in cur]
                    new_pts[i][0] += dx
                    new_pts[i][1] += dy
                    if not _valid(new_pts):
                        continue
                    new_obj = _objective_fast(new_pts, old_pt, new_pts[i])
                    if new_obj < cur_obj:
                        cur = new_pts
                        cur_obj = new_obj
                        if cur_obj < best_obj:
                            best_obj = cur_obj
                            best = [p[:] for p in cur]
                        improved = True
                        found = True
                        break
            # Try snap-to-line moves
            if time.time() - start_time >= max_time:
                break
            for j in range(12):
                if time.time() - start_time >= max_time:
                    break
                for k in range(12):
                    if i == j or i == k or j == k:
                        continue
                    x1, y1 = cur[j]
                    x2, y2 = cur[k]
                    if x1 == x2 and y1 == y2:
                        continue
                    res = _find_integer_point_on_line(x1, y1, x2, y2)
                    if res is None:
                        continue
                    x0, y0, a, b = res
                    for t in range(-15, 16):
                        nx = x0 + b * t
                        ny = y0 - a * t
                        if abs(nx) > 10**9 or abs(ny) > 10**9:
                            continue
                        new_pts = [p[:] for p in cur]
                        new_pts[i] = [nx, ny]
                        if not _valid(new_pts):
                            continue
                        new_obj = _objective_fast(new_pts, old_pt, new_pts[i])
                        if new_obj < cur_obj:
                            cur = new_pts
                            cur_obj = new_obj
                            if cur_obj < best_obj:
                                best_obj = cur_obj
                                best = [p[:] for p in cur]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return best


def _canonical_key(points):
    """Create a canonical key for deduplication."""
    return tuple(sorted(tuple(p) for p in points))


def construct():
    """Return 12 distinct integer points, |x|,|y| <= 1e9, not all collinear,
    minimizing ordinary lines. Uses SA with rich candidate pools."""
    
    best_pts = None
    best_score = float('inf')
    seen_configs = set()
    
    # Structured seeds
    seeds = [
        _make_seed_4triples(),
        _make_seed_diagonal(),
        _make_seed_mixed(),
        _make_seed_parallel_triples(),
        _make_seed_staggered(),
        _make_seed_boroczky2(),
        _make_seed_boroczky3(),
        _make_seed_boroczky(),
    ]
    
    # Add random seeds
    for _ in range(3):
        pts = [_random_point() for _ in range(12)]
        if _valid(pts):
            seeds.append(pts)
    
    # Deduplicate seeds
    unique_seeds = []
    for s in seeds:
        key = _canonical_key(s)
        if key not in seen_configs:
            seen_configs.add(key)
            unique_seeds.append(s)
    
    # Run SA for each seed with time budget
    total_time = 120.0
    time_per_seed = total_time / max(len(unique_seeds), 1)
    time_per_seed = min(time_per_seed, 12.0)
    
    for idx, seed in enumerate(unique_seeds):
        if time.time() > _DEADLINE - 10:
            break
        remaining = _DEADLINE - 10 - time.time()
        seed_time = min(time_per_seed, remaining)
        if seed_time < 2.0:
            break
            
        pts, score = _local_search_sa(seed, max_time=seed_time)
        if pts and score < best_score:
            key = _canonical_key(pts)
            if key not in seen_configs:
                seen_configs.add(key)
                best_score = score
                best_pts = pts
                # Early exit if we found the theoretical minimum
                ord_count, _, _ = _count_lines(best_pts)
                if ord_count <= 6:
                    break
    
    # If we found a good solution, try to refine it further
    if best_pts is not None and best_score > 600 and time.time() < _DEADLINE - 20:
        pts, score = _local_search_sa(best_pts, max_time=15.0)
        if pts and score < best_score:
            key = _canonical_key(pts)
            if key not in seen_configs:
                seen_configs.add(key)
                best_score = score
                best_pts = pts
    
    # Try additional SA runs from the best solution with different random seeds
    if best_pts is not None and time.time() < _DEADLINE - 15:
        for trial in range(3):
            if time.time() > _DEADLINE - 10:
                break
            rng = random.Random(1000 + trial * 777)
            pts, score = _local_search_sa(best_pts, max_time=4.0, rng=rng)
            if pts and score < best_score:
                key = _canonical_key(pts)
                if key not in seen_configs:
                    seen_configs.add(key)
                    best_score = score
                    best_pts = pts
    
    # Final local polish
    if best_pts is not None and time.time() < _DEADLINE - 5:
        polished = _local_polish(best_pts, max_time=5.0)
        if polished:
            polished_obj = _objective(polished)
            if polished_obj < best_score:
                best_pts = polished
                best_score = polished_obj
    
    if best_pts is None:
        best_pts = _make_seed_4triples()
        
    if not _valid(best_pts):
        best_pts = _make_seed_4triples()
        
    return best_pts


if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("3-point lines:", _count_3point_lines(pts))
    print("Objective:", _objective(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))

"""Direct construction: Simulated Annealing for 12-point configuration with few ordinary lines.

Uses Simulated Annealing with structural seeds (collinear triples) and triple-aware moves
to search for integer coordinates that minimize the number of ordinary lines.
Target: 6 ordinary lines (4 collinear triples covering all 12 points).
"""
import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 145.0  # safety margin before 150s kill


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
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    ordinary = sum(1 for c in line_counts.values() if c == 1)
    triples = sum(1 for c in line_counts.values() if c == 3)
    return ordinary, triples, line_counts


def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    ord_c, _, _ = _count_lines(points)
    return ord_c


def _count_3point_lines(points):
    """Count lines containing exactly 3 of the given points (n=12)."""
    _, tri_c, _ = _count_lines(points)
    return tri_c


def _objective(points):
    """Objective: Minimize ordinary lines, secondary: Maximize 3-point lines."""
    ord_lines, tri_lines, _ = _count_lines(points)
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
    """Create a seed with 4 collinear triples (4 lines, 3 points each, 12 total).
    This is the target structure: 6 ordinary lines minimum."""
    # 4 lines, each with 3 points, no two lines sharing more than 1 point ideally
    # Simple construction: 4 horizontal lines at different y, 3 points each
    # y=0: (0,0),(1,0),(2,0)
    # y=1: (0,1),(1,1),(2,1)
    # y=2: (0,2),(1,2),(2,2)
    # y=3: (0,3),(1,3),(2,3)
    # This is a 3x4 grid - has many extra collinearities but starts from triple structure
    points = []
    for y in range(4):
        for x in range(3):
            points.append([x, y])
    return points


def _make_seed_diagonal():
    """Create a seed with 4 diagonal triples."""
    # Line 1: y=x -> (0,0),(1,1),(2,2)
    # Line 2: y=x+1 -> (0,1),(1,2),(2,3)
    # Line 3: y=-x+2 -> (0,2),(1,1),(2,0) -- but (1,1) and (2,0) already used
    # Let me use parallel lines with different offsets
    # y = x + c for c = 0, 1, 2, 3
    points = []
    for c in range(4):
        for t in range(3):
            points.append([t, t + c])
    return points


def _make_seed_mixed():
    """Create a mixed seed: 2 horizontal triples, 2 diagonal triples."""
    points = []
    # Horizontal: y=0: (0,0),(1,0),(2,0)
    points.extend([[0,0],[1,0],[2,0]])
    # Horizontal: y=1: (0,1),(1,1),(2,1)
    points.extend([[0,1],[1,1],[2,1]])
    # Diagonal: y=x+2: (0,2),(1,3),(2,4)
    points.extend([[0,2],[1,3],[2,4]])
    # Diagonal: y=-x+3: (0,3),(1,2),(2,1) -- (2,1) already used!
    # Use y=-x+4: (0,4),(1,3),(2,2) -- (1,3) already used!
    # Let me adjust: y=-x+5: (0,5),(1,4),(2,3)
    points.extend([[0,5],[1,4],[2,3]])
    return points


def _make_seed_parallel_triples():
    """4 parallel lines (slope 1), 3 points each, spaced apart."""
    points = []
    # y = x + c, c = 0, 3, 6, 9
    for c in [0, 3, 6, 9]:
        for t in [0, 1, 2]:
            points.append([t, t + c])
    return points


def _make_seed_staggered():
    """Staggered triples to minimize extra collinearities."""
    points = []
    # Triple 1: horizontal y=0: (0,0),(2,0),(4,0)
    points.extend([[0,0],[2,0],[4,0]])
    # Triple 2: horizontal y=1: (1,1),(3,1),(5,1)
    points.extend([[1,1],[3,1],[5,1]])
    # Triple 3: diagonal y=x: (0,1) already used... use y=x-1: (-1,0) already used
    # Let's use: y = 2x: (0,0) used. y = 2x+1: (0,1) used.
    # y = -x + 3: (0,3),(1,2),(2,1)
    points.extend([[0,3],[1,2],[2,1]])
    # Triple 4: y = -x + 4: (0,4),(1,3),(2,2)
    points.extend([[0,4],[1,3],[2,2]])
    return points


def _egcd(a, b):
    """Extended Euclidean algorithm."""
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def _find_integer_point_on_line(x1, y1, x2, y2, t_range=(-20, 20)):
    """Find an integer point on the line through (x1,y1) and (x2,y2)."""
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
    if g:
        A //= g
        B //= g
        C //= g
    # A x + B y = -C
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
    # General solution: x = x0 + b*t, y = y0 - a*t
    return (x0, y0, a, b)  # base point and direction


def _simulate_annealing(seed_points, max_time=10.0, rng=None):
    """Run Simulated Annealing to optimize a configuration."""
    if rng is None:
        rng = random.Random(int(time.time() * 1000) % 2**32)
    cur = [p[:] for p in seed_points]
    if not _valid(cur):
        for i in range(12):
            if not _valid(cur):
                cur[i] = _random_point()
        if not _valid(cur):
            return None, float('inf')
            
    cur_obj = _objective(cur)
    best = [p[:] for p in cur]
    best_obj = cur_obj
    
    T = 500.0
    T_min = 0.5
    cooling_rate = 0.9995
    
    start_time = time.time()
    
    while T > T_min and time.time() - start_time < max_time:
        new_pts = [p[:] for p in cur]
        move_type = rng.random()
        
        if move_type < 0.35:
            # Small perturbation: move one point slightly
            idx = rng.randrange(12)
            new_pts[idx][0] += rng.randint(-3, 3)
            new_pts[idx][1] += rng.randint(-3, 3)
        elif move_type < 0.65:
            # Snap to line: move point i onto line through j and k
            i = rng.randrange(12)
            j = rng.randrange(12)
            k = rng.randrange(12)
            if i == j or i == k or j == k:
                continue
            x1, y1 = new_pts[j]
            x2, y2 = new_pts[k]
            if x1 == x2 and y1 == y2:
                continue
            res = _find_integer_point_on_line(x1, y1, x2, y2)
            if res is None:
                continue
            x0, y0, a, b = res
            t = rng.randint(-15, 15)
            nx = x0 + b * t
            ny = y0 - a * t
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pts[i] = [nx, ny]
        elif move_type < 0.85:
            # Swap coordinates or reflect
            idx = rng.randrange(12)
            op = rng.randint(0, 2)
            if op == 0:
                new_pts[idx][0], new_pts[idx][1] = new_pts[idx][1], new_pts[idx][0]
            elif op == 1:
                new_pts[idx][0] = -new_pts[idx][0]
            else:
                new_pts[idx][1] = -new_pts[idx][1]
        else:
            # Large jump
            idx = rng.randrange(12)
            new_pts[idx] = _random_point()
            
        if not _valid(new_pts):
            T *= cooling_rate
            continue
            
        new_obj = _objective(new_pts)
        delta = new_obj - cur_obj
        
        if delta < 0 or rng.random() < pow(2.71828, -delta / T):
            cur = new_pts
            cur_obj = new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best = [p[:] for p in cur]
                
        T *= cooling_rate
        
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
            # Try small moves
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = [p[:] for p in cur]
                    new_pts[i][0] += dx
                    new_pts[i][1] += dy
                    if not _valid(new_pts):
                        continue
                    new_obj = _objective(new_pts)
                    if new_obj < cur_obj:
                        cur = new_pts
                        cur_obj = new_obj
                        if cur_obj < best_obj:
                            best_obj = cur_obj
                            best = [p[:] for p in cur]
                        improved = True
                        break
                else:
                    continue
                break
            # Try snap-to-line moves
            if time.time() - start_time >= max_time:
                break
            for j in range(12):
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
                    for t in range(-10, 11):
                        nx = x0 + b * t
                        ny = y0 - a * t
                        if abs(nx) > 10**9 or abs(ny) > 10**9:
                            continue
                        new_pts = [p[:] for p in cur]
                        new_pts[i] = [nx, ny]
                        if not _valid(new_pts):
                            continue
                        new_obj = _objective(new_pts)
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
    minimizing ordinary lines. Uses Simulated Annealing with structural seeds."""
    
    best_pts = None
    best_score = float('inf')
    seen_configs = set()
    
    # Structured seeds with collinear triples
    seeds = [
        _make_seed_4triples(),
        _make_seed_diagonal(),
        _make_seed_mixed(),
        _make_seed_parallel_triples(),
        _make_seed_staggered(),
    ]
    
    # Add some random seeds
    for _ in range(4):
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
    
    # Run SA for each seed
    time_per_seed = 12.0
    
    for idx, seed in enumerate(unique_seeds):
        if time.time() > _DEADLINE - 5:
            break
            
        pts, score = _simulate_annealing(seed, max_time=time_per_seed)
        if pts and score < best_score:
            key = _canonical_key(pts)
            if key not in seen_configs:
                seen_configs.add(key)
                best_score = score
                best_pts = pts
    
    # If we found a good solution, try to refine it further
    if best_pts is not None and best_score > 600 and time.time() < _DEADLINE - 20:
        pts, score = _simulate_annealing(best_pts, max_time=20.0)
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
            pts, score = _simulate_annealing(best_pts, max_time=5.0, rng=rng)
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
        # Fallback
        best_pts = _make_seed_4triples()
        
    # Final validation
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

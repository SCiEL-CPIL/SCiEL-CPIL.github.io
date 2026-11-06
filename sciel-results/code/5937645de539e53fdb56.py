import time
import random
import itertools
from math import gcd

DEADLINE = 140.0  # Safety margin

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

def _max_cap_free(points, time_limit=2.0):
    """Max subset with no three collinear via branch and bound."""
    n = len(points)
    start_time = time.time()

    line_map = _build_lines(points)
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

    memo = {}

    def dfs(current_mask, next_idx):
        if time.time() - start_time > time_limit:
            return 0
        cur_size = current_mask.bit_count()
        ub = cur_size + (n - next_idx)
        if ub <= best[0]:
            return 0
        key = (next_idx, current_mask)
        if key in memo:
            return memo[key]
        
        best_add = 0
        # Heuristic: try to include points that are "constrained" first?
        # Or just standard order.
        for idx in range(next_idx, n):
            if time.time() - start_time > time_limit:
                break
            if is_valid_add(current_mask, idx):
                r = dfs(current_mask | (1 << idx), idx + 1)
                if r > best_add:
                    best_add = r
            # Implicitly excludes idx by continuing loop
        memo[key] = best_add
        total = cur_size + best_add
        if total > best[0]:
            best[0] = total
        return best_add

    dfs(0, 0)
    return best[0]

def _score(points, time_limit=2.0):
    """Compute the max cap-free subset size. Returns 999 if invalid."""
    line_map = _build_lines(points)
    if _has_four_collinear(line_map):
        return 999
    return _max_cap_free(points, time_limit=time_limit)

def _generate_initial(seed, box_size=15):
    """Generate random points in a small box to increase density."""
    rng = random.Random(seed)
    points = []
    used = set()
    while len(points) < 20:
        x = rng.randint(-box_size, box_size)
        y = rng.randint(-box_size, box_size)
        if (x, y) not in used:
            used.add((x, y))
            points.append([x, y])
    return points

def _local_search(points, time_limit=10.0, search_radius=3):
    """Local search to minimize score."""
    start_time = time.time()
    current = [p[:] for p in points]
    current_score = _score(current, time_limit=1.0)
    
    if current_score >= 999:
        # If invalid, try to fix by moving points randomly until valid
        # For now, just return as is if we can't fix quickly, 
        # but the caller should handle invalids.
        # Actually, let's try a few random moves to fix validity.
        for _ in range(50):
            i = random.randint(0, 19)
            dx = random.randint(-search_radius, search_radius)
            dy = random.randint(-search_radius, search_radius)
            nx, ny = current[i][0] + dx, current[i][1] + dy
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                current[i] = [nx, ny]
                if len(set(tuple(p) for p in current)) == 20:
                    s = _score(current, time_limit=0.5)
                    if s < 999:
                        current_score = s
                        break

    best = [p[:] for p in current]
    best_score = current_score

    while time.time() - start_time < time_limit:
        improved = False
        # Shuffle points to avoid bias
        indices = list(range(20))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            ox, oy = current[i]
            # Try moves in a small neighborhood
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
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
                    s = _score(current, time_limit=0.5)
                    if s < 999 and s < current_score:
                        current_score = s
                        improved = True
                        if s < best_score:
                            best_score = s
                            best = [p[:] for p in current]
                    # Revert
                    current[i] = [ox, oy]
                    if time.time() - start_time > time_limit:
                        break
                if time.time() - start_time > time_limit:
                    break
            if time.time() - start_time > time_limit:
                break
        if not improved:
            # Stagnation: try larger jumps or break
            break

    return best, best_score

def construct():
    start_time = time.time()
    best_points = None
    best_score = 999
    
    # Try multiple seeds
    for seed in range(10):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        initial_points = _generate_initial(seed, box_size=10 + seed)
        # Quick validity check
        lm = _build_lines(initial_points)
        if _has_four_collinear(lm):
            # Try to fix
            # Just skip this seed for now, or do a quick repair
            # For simplicity, skip if invalid initially, rely on other seeds
            continue
            
        pts, score = _local_search(initial_points, time_limit=10.0)
        
        if score < best_score:
            best_score = score
            best_points = pts
            
        if best_score <= 10: # Heuristic lower bound guess, if we get very good, stop
            break

    if best_points is None:
        # Fallback to a known valid set (e.g. parabola)
        # Parabola has no 3 collinear, so score is 20.
        # We want lower than 20.
        # Let's return the static set from the original if nothing better found
        points = [
            [0, 0], [1, 0], [2, 0],
            [3, 1], [4, 1], [5, 1],
            [6, 2], [7, 2], [8, 2],
            [0, 3], [1, 3], [2, 3],
            [3, 4], [4, 4], [5, 4],
            [6, 5], [7, 5], [8, 5],
            [100, 100],
            [100, -100]
        ]
        # Verify it
        lm = _build_lines(points)
        if _has_four_collinear(lm):
             # Should not happen
             return [[x, x*x] for x in range(-10, 10)]
        return points
    else:
        return best_points

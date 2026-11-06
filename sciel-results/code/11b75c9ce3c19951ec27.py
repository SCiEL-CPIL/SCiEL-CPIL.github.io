import time
import random
from math import gcd

DEADLINE = 145

def _line_key(p1, p2):
    """Get a canonical representation of the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return ('v', x1)
    elif dy == 0:
        return ('h', y1)
    else:
        g = gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        if dx_n < 0:
            dx_n = -dx_n
            dy_n = -dy_n
        c = dy_n * x1 - dx_n * y1
        return ('s', dx_n, dy_n, c)

def _build_lines(points):
    """Build line -> set of point indices. Single pass."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _check_no_four_collinear(lines):
    """Check no line has 4+ points."""
    for pts in lines.values():
        if len(pts) >= 4:
            return False
    return True

def _get_triple_bitmasks(lines):
    """Precompute collinear_mask: for each i, for each j, bitmask of k such that {i,j,k} collinear."""
    n = 20
    collinear_mask = [[0] * n for _ in range(n)]
    for pts in lines.values():
        if len(pts) >= 3:
            pts_list = sorted(pts)
            m = len(pts_list)
            for a in range(m):
                for b in range(a + 1, m):
                    for c in range(b + 1, m):
                        ia, ib, ic = pts_list[a], pts_list[b], pts_list[c]
                        collinear_mask[ia][ib] |= (1 << ic)
                        collinear_mask[ia][ic] |= (1 << ib)
                        collinear_mask[ib][ia] |= (1 << ic)
                        collinear_mask[ib][ic] |= (1 << ia)
                        collinear_mask[ic][ia] |= (1 << ib)
                        collinear_mask[ic][ib] |= (1 << ia)
    return collinear_mask

def _max_cap_free(points, collinear_mask, deadline):
    """Find largest subset with no three collinear using bitmask branch and bound."""
    n = len(points)
    best = [0]
    
    def backtrack(subset_mask, start, current_size):
        if time.time() > deadline:
            return
        if current_size > best[0]:
            best[0] = current_size
        # Upper bound
        if current_size + (n - start) <= best[0]:
            return
        for i in range(start, n):
            if time.time() > deadline:
                return
            if subset_mask & (1 << i):
                continue
            # Check validity: for each j in subset, collinear_mask[i][j] must not intersect subset
            valid = True
            temp = subset_mask
            while temp:
                j_bit = temp & (-temp)
                j = j_bit.bit_length() - 1
                if collinear_mask[i][j] & subset_mask:
                    valid = False
                    break
                temp -= j_bit
            if valid:
                backtrack(subset_mask | (1 << i), i + 1, current_size + 1)
    
    backtrack(0, 0, 0)
    return best[0]

def _compute_score(points, deadline):
    """Compute score: largest subset with no three collinear."""
    lines = _build_lines(points)
    if not _check_no_four_collinear(lines):
        return 20
    collinear_mask = _get_triple_bitmasks(lines)
    return _max_cap_free(points, collinear_mask, deadline)

def _is_valid(points, deadline):
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 100 or abs(y) > 100:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    lines = _build_lines(points)
    return _check_no_four_collinear(lines)

def _gen_perturbed_grid(seed):
    """Generate points on a perturbed grid structure to create many collinear triples."""
    rng = random.Random(seed)
    # Use a grid-like structure with perturbations
    # Place points in clusters that share lines
    points = []
    # Strategy: place points near a few lines to create many triples
    # Use a 5x4 grid base with perturbations
    base_xs = [-40, -20, 0, 20, 40]
    base_ys = [-30, -10, 10, 30]
    
    candidates = []
    for x in base_xs:
        for y in base_ys:
            # Add perturbation
            px = x + rng.randint(-5, 5)
            py = y + rng.randint(-5, 5)
            px = max(-100, min(100, px))
            py = max(-100, min(100, py))
            candidates.append([px, py])
    
    # Shuffle and take 20
    rng.shuffle(candidates)
    points = candidates[:20]
    
    # Ensure distinct
    seen = set()
    unique = []
    for p in points:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique.append(p)
    while len(unique) < 20:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique.append([x, y])
    return unique[:20]

def _gen_parabola_perturbed(seed):
    """Generate points on a perturbed parabola."""
    rng = random.Random(seed)
    points = []
    xs = sorted(rng.sample(range(-50, 51), 20))
    for x in xs:
        y = x * x // 10 + rng.randint(-5, 5)
        y = max(-100, min(100, y))
        points.append([x, y])
    seen = set()
    unique = []
    for p in points:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique.append(p)
    while len(unique) < 20:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique.append([x, y])
    return unique[:20]

def _local_search(points, deadline):
    """Local search: try moving each point by small amounts."""
    best = [list(p) for p in points]
    best_score = _compute_score(best, deadline)
    
    for iteration in range(30):
        if time.time() > deadline - 20:
            break
        improved = False
        # Randomize order of points to try
        indices = list(range(20))
        random.shuffle(indices)
        for i in indices:
            if time.time() > deadline - 20:
                break
            orig = best[i][:]
            # Try various move sizes
            for radius in [1, 2, 3, 5, 8]:
                if time.time() > deadline - 20:
                    break
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = orig[0] + dx, orig[1] + dy
                        if abs(nx) > 100 or abs(ny) > 100:
                            continue
                        # Check no collision
                        collides = False
                        for k in range(20):
                            if k != i and best[k][0] == nx and best[k][1] == ny:
                                collides = True
                                break
                        if collides:
                            continue
                        candidate = [list(p) for p in best]
                        candidate[i] = [nx, ny]
                        if _is_valid(candidate, deadline):
                            s = _compute_score(candidate, deadline)
                            if s < best_score:
                                best_score = s
                                best = candidate
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    random.seed(42)
    
    best_points = [[x, x * x] for x in range(-10, 10)]  # Baseline parabola
    best_score = 20
    
    # Strategy 1: Perturbed grids (measured to work well)
    for seed in range(100):
        if time.time() > deadline - 40:
            break
        pts = _gen_perturbed_grid(seed)
        if _is_valid(pts, deadline):
            s = _compute_score(pts, deadline)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in pts]
    
    # Strategy 2: Perturbed parabolas
    for seed in range(50):
        if time.time() > deadline - 40:
            break
        pts = _gen_parabola_perturbed(seed)
        if _is_valid(pts, deadline):
            s = _compute_score(pts, deadline)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in pts]
    
    # Strategy 3: Random points
    for gen in range(50):
        if time.time() > deadline - 40:
            break
        points = []
        while len(points) < 20:
            x = random.randint(-60, 60)
            y = random.randint(-60, 60)
            if (x, y) not in [tuple(p) for p in points]:
                points.append([x, y])
        if _is_valid(points, deadline):
            s = _compute_score(points, deadline)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in points]
    
    # Local search from best found
    if time.time() < deadline - 25:
        best_points, best_score = _local_search(best_points, deadline)
    
    # Final local search passes if time remains
    while time.time() < deadline - 25:
        improved_pts, improved_score = _local_search(best_points, deadline)
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_pts
        else:
            break
    
    return [list(p) for p in best_points]

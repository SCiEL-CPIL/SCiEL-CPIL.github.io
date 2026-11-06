"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Map each line to the set of point indices on it
    line_map = defaultdict(set)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            # Normalize the direction
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            # Normalize sign: make first non-zero positive
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            # Line is determined by direction and a point on it
            # Use the "canonical" point: the one with smallest (x, y) among points on this line in this direction
            # Actually, for counting, we just need to group by line.
            # A line can be represented by (slope, intercept) but that's messy with integers.
            # Better: for each pair, compute the line in normalized form.
            # Line through (xi,yi) and (xj,yj) with direction (dx_n, dy_n):
            # The line can be identified by (dy_n * x - dx_n * y = c) where c = dy_n*xi - dx_n*yi
            c = dy_n * xi - dx_n * yi
            line_key = (dx_n, dy_n, c)
            line_map[line_key].add(i)
            line_map[line_key].add(j)
    
    ordinary = 0
    for key, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _evaluate(points):
    """Return the number of ordinary lines (lower is better)."""
    return _count_ordinary_lines(points)

def _try_move(points, idx, new_pt):
    """Try moving point idx to new_pt; return new ordinary count if improved, else None."""
    # Quick check: new_pt must not coincide with any other point
    for j in range(len(points)):
        if j != idx and points[j] == new_pt:
            return None
    old = points[idx]
    points[idx] = new_pt
    cnt = _count_ordinary_lines(points)
    points[idx] = old
    return cnt

def _local_search(points, max_iters=50000, deadline=None):
    """Local search: move one point at a time to reduce ordinary lines."""
    n = len(points)
    best = points[:]
    best_cnt = _count_ordinary_lines(best)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        if deadline and time.time() > deadline:
            break
        improved = False
        for idx in range(n):
            if deadline and time.time() > deadline:
                break
            # Try moving point idx to various positions
            xi, yi = best[idx]
            # Candidate positions: nearby lattice points
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = [xi + dx, yi + dy]
                    # Check bounds (we'll scale later, but keep within reasonable range)
                    if abs(new_pt[0]) > 100 or abs(new_pt[1]) > 100:
                        continue
                    # Check not coinciding
                    valid = True
                    for j in range(n):
                        if j != idx and best[j] == new_pt:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Evaluate
                    best[idx] = new_pt
                    cnt = _count_ordinary_lines(best)
                    best[idx] = [xi, yi]
                    if cnt < best_cnt:
                        best[idx] = new_pt
                        best_cnt = cnt
                        improved = True
                        iters += 1
                        if deadline and time.time() > deadline:
                            return best, best_cnt
            # Also try some random distant moves
            for _ in range(3):
                if deadline and time.time() > deadline:
                    break
                rx = random.randint(-50, 50)
                ry = random.randint(-50, 50)
                new_pt = [rx, ry]
                valid = True
                for j in range(n):
                    if j != idx and best[j] == new_pt:
                        valid = False
                        break
                if not valid:
                    continue
                best[idx] = new_pt
                cnt = _count_ordinary_lines(best)
                best[idx] = [xi, yi]
                if cnt < best_cnt:
                    best[idx] = new_pt
                    best_cnt = cnt
                    improved = True
                    iters += 1
        iters += 1
    return best, best_cnt

def _pair_swap_search(points, max_iters=10000, deadline=None):
    """Try swapping pairs of points with random new positions."""
    n = len(points)
    best = points[:]
    best_cnt = _count_ordinary_lines(best)
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        # Pick two random indices
        i, j = random.sample(range(n), 2)
        old_i, old_j = best[i][:], best[j][:]
        # Try random new positions
        for _ in range(5):
            ni = [random.randint(-30, 30), random.randint(-30, 30)]
            nj = [random.randint(-30, 30), random.randint(-30, 30)]
            if ni == nj:
                continue
            # Check no collisions
            valid = True
            for k in range(n):
                if k != i and k != j:
                    if best[k] == ni or best[k] == nj:
                        valid = False
                        break
                if k == i and ni == old_j:
                    valid = False
                if k == j and nj == old_i:
                    valid = False
            if not valid:
                continue
            best[i] = ni
            best[j] = nj
            cnt = _count_ordinary_lines(best)
            best[i] = old_i
            best[j] = old_j
            if cnt < best_cnt:
                best[i] = ni
                best[j] = nj
                best_cnt = cnt
    return best, best_cnt

def _make_grid(w, h, offset=(0, 0)):
    """Create a w x h grid."""
    pts = []
    for x in range(w):
        for y in range(h):
            pts.append([x + offset[0], y + offset[1]])
    return pts

def _make_hexagonal(rows=4):
    """Create a hexagonal-ish arrangement."""
    pts = []
    for r in range(rows):
        for c in range(rows):
            x = c + (r % 2)
            y = r
            pts.append([x, y])
    return pts[:12]

def _make_near_pencil():
    """11 points on a line, 1 off."""
    pts = [[i, 0] for i in range(11)]
    pts.append([5, 1])
    return pts

def _make_staggered():
    """Staggered rows."""
    pts = []
    for r in range(4):
        for c in range(3):
            x = c + (r % 2) * 2
            y = r * 2
            pts.append([x, y])
    return pts

def _make_diagonal_grid():
    """Points on diagonal lines."""
    pts = []
    for i in range(4):
        for j in range(3):
            pts.append([i + j, i - j])
    return pts

def _scale_points(points, factor=100):
    """Scale points to use more of the coordinate space."""
    return [[x * factor, y * factor] for x, y in points]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    random.seed(42)
    
    # Generate diverse initial configurations
    seeds = []
    seeds.append(_make_grid(4, 3))
    seeds.append(_make_grid(3, 4))
    seeds.append(_make_hexagonal())
    seeds.append(_make_near_pencil())
    seeds.append(_make_staggered())
    seeds.append(_make_diagonal_grid())
    
    # Add some random configurations
    for _ in range(5):
        pts = []
        while len(pts) < 12:
            p = [random.randint(-20, 20), random.randint(-20, 20)]
            if p not in pts:
                pts.append(p)
        seeds.append(pts)
    
    best_pts = None
    best_cnt = float('inf')
    
    for seed in seeds:
        if time.time() > deadline:
            break
        # Evaluate seed
        cnt = _count_ordinary_lines(seed)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = seed[:]
        
        # Local search from this seed
        pts, cnt = _local_search(seed, max_iters=20000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
        
        # Pair swap from best so far
        if best_pts:
            pts, cnt = _pair_swap_search(best_pts, max_iters=5000, deadline=deadline)
            if cnt < best_cnt:
                best_cnt = cnt
                best_pts = pts[:]
    
    # Final refinement: more local search from best
    if best_pts and time.time() < deadline - 10:
        pts, cnt = _local_search(best_pts, max_iters=50000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
    
    # Ensure points are distinct and within bounds
    # Scale if needed to use the full coordinate range
    if best_pts is None:
        best_pts = _make_grid(4, 3)
    
    # Verify distinctness
    seen = set()
    final = []
    for p in best_pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            final.append(p)
    
    # If we have fewer than 12 (shouldn't happen), add more
    while len(final) < 12:
        p = [random.randint(-10**9, 10**9), random.randint(-10**9, 10**9)]
        if tuple(p) not in seen:
            seen.add(tuple(p))
            final.append(p)
    
    # Scale to use full coordinate space (optional, helps with distinctness)
    # Actually, keep them small to avoid issues
    # Just return as is, they're already integers within bounds
    
    return final
# EVOLVE-BLOCK-END

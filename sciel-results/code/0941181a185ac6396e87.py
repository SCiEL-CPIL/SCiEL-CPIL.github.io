"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from math import gcd

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_map = defaultdict(set)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            c = dy_n * xi - dx_n * yi
            line_key = (dx_n, dy_n, c)
            line_map[line_key].add(i)
            line_map[line_key].add(j)
    
    ordinary = 0
    for key, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _get_lines(points):
    """Return dict mapping line_key -> set of point indices on that line."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            c = dy_n * xi - dx_n * yi
            line_key = (dx_n, dy_n, c)
            line_map[line_key].add(i)
            line_map[line_key].add(j)
    return line_map

def _count_triples(points):
    """Count number of collinear triples (i,j,k) with i<j<k."""
    n = len(points)
    line_map = _get_lines(points)
    triples = 0
    for key, pts in line_map.items():
        k = len(pts)
        if k >= 3:
            triples += k * (k - 1) * (k - 2) // 6
    return triples

def _evaluate_combined(points):
    """Combined score: primary is ordinary lines, secondary is -triples."""
    ordinary = _count_ordinary_lines(points)
    triples = _count_triples(points)
    return (ordinary, -triples)

def _try_place_on_line(idx, points, line_key, line_map, max_search=30):
    """
    Try to place point idx on the given line (identified by line_key) such that
    it lands on an integer point not already occupied. Returns (new_point, new_count) 
    if improvement found, else (None, None).
    """
    n = len(points)
    pts_on_line = line_map[line_key]
    if idx in pts_on_line or len(pts_on_line) < 2:
        return None, None
    
    # Get direction from the line key
    dx_n, dy_n, c = line_key
    
    # Find two points on the line to get a base
    pts_list = list(pts_on_line)
    p1 = points[pts_list[0]]
    p2 = points[pts_list[1]]
    
    # Use p1 as a particular solution
    x0, y0 = p1[0], p1[1]
    
    # Find existing t values for points on the line
    t_values = set()
    for pidx in pts_on_line:
        px, py = points[pidx]
        if dx_n != 0:
            if (px - x0) % dx_n == 0:
                t_values.add((px - x0) // dx_n)
        elif dy_n != 0:
            if (py - y0) % dy_n == 0:
                t_values.add((py - y0) // dy_n)
    
    existing_points = set(tuple(p) for p in points)
    
    # Generate candidate t values
    candidate_ts = set()
    for t in t_values:
        for dt in range(-max_search, max_search + 1):
            candidate_ts.add(t + dt)
    
    # Also try t values near 0
    for dt in range(-max_search, max_search + 1):
        candidate_ts.add(dt)
    
    # Sort candidates by distance to current point
    xi, yi = points[idx]
    def dist_to_current(t):
        nx = x0 + t * dx_n
        ny = y0 + t * dy_n
        return (nx - xi) ** 2 + (ny - yi) ** 2
    
    candidate_ts = sorted(candidate_ts, key=dist_to_current)
    
    for t in candidate_ts[:100]:  # Try closest 100
        nx = x0 + t * dx_n
        ny = y0 + t * dy_n
        
        if abs(nx) > 10**9 or abs(ny) > 10**9:
            continue
        
        new_pt = (nx, ny)
        if new_pt in existing_points:
            continue
        
        # Check if this is on the line
        if dy_n * nx - dx_n * ny != c:
            continue
        
        # Try this move
        old_pt = points[idx][:]
        points[idx] = [nx, ny]
        cnt = _count_ordinary_lines(points)
        points[idx] = old_pt
        
        return [nx, ny], cnt
    
    return None, None

def _smart_local_search(points, max_iters=50000, deadline=None):
    """
    Smart local search: for each point, try placing it on an existing line
    (one that already has 2+ other points) to create a 3+ point line,
    which reduces ordinary line count.
    """
    n = len(points)
    best = [p[:] for p in points]
    best_cnt = _count_ordinary_lines(best)
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        if deadline and time.time() > deadline:
            break
        improved = False
        lines = _get_lines(best)
        
        for idx in range(n):
            if deadline and time.time() > deadline:
                break
            xi, yi = best[idx]
            
            # Strategy 1: Try to place on an existing line with 2+ other points
            for line_key, pts_on_line in lines.items():
                if idx in pts_on_line:
                    continue
                if len(pts_on_line) < 2:
                    continue
                
                new_pt, cnt = _try_place_on_line(idx, best, line_key, lines, max_search=15)
                if new_pt is not None and cnt < best_cnt:
                    best[idx] = new_pt
                    best_cnt = cnt
                    improved = True
                    iters += 1
                    lines = _get_lines(best)
                    break
            
            # Strategy 2: Try nearby lattice points with larger radius
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = [xi + dx, yi + dy]
                    if abs(new_pt[0]) > 200 or abs(new_pt[1]) > 200:
                        continue
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
                        lines = _get_lines(best)
            
            # Strategy 3: Random moves with wider range
            for _ in range(15):
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
                    lines = _get_lines(best)
        
        iters += 1
    return best, best_cnt

def _pair_swap_search(points, max_iters=15000, deadline=None):
    """Try swapping pairs of points with random new positions."""
    n = len(points)
    best = [p[:] for p in points]
    best_cnt = _count_ordinary_lines(best)
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        i, j = random.sample(range(n), 2)
        old_i, old_j = best[i][:], best[j][:]
        for _ in range(10):
            ni = [random.randint(-60, 60), random.randint(-60, 60)]
            nj = [random.randint(-60, 60), random.randint(-60, 60)]
            if ni == nj:
                continue
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
    pts = []
    for x in range(w):
        for y in range(h):
            pts.append([x + offset[0], y + offset[1]])
    return pts

def _make_three_parallel_lines():
    """
    4 points on each of 3 parallel lines.
    """
    pts = []
    for y in [0, 1, 2]:
        for x in [0, 1, 2, 3]:
            pts.append([x, y])
    return pts

def _make_tetrahedral():
    """
    Points arranged to maximize collinear triples.
    """
    pts = [[0, 0], [1, 0], [0, 1], [1, 1]]
    pts.append([2, 0])    
    pts.append([0, 2])    
    pts.append([2, 1])    
    pts.append([1, 2])    
    pts.append([2, 2])    
    pts.append([2, -1])   
    return pts

def _make_optimal_structure():
    """
    Try to construct a configuration close to optimal.
    """
    pts = []
    for x in range(4):
        for y in range(3):
            pts.append([x, y])
    return pts

def _make_hexagonal():
    pts_hex = []
    for r in range(4):
        for c in range(3):
            x = c + (r % 2)
            y = r
            pts_hex.append([x, y])
    return pts_hex

def _make_near_pencil():
    pts_pencil = [[i, 0] for i in range(11)]
    pts_pencil.append([5, 1])
    return pts_pencil

def _make_staggered():
    pts = []
    for i in range(4):
        for j in range(3):
            x = i * 2 + (j % 2)
            y = j * 2
            pts.append([x, y])
    return pts

def _make_diagonal_grid():
    pts = []
    for i in range(3):
        for j in range(4):
            pts.append([i + j, i])
    return pts

def _make_hesse_like():
    """
    Inspired by Hesse configuration: 9 points with 12 lines of 3 points.
    For 12 points, we extend this idea.
    """
    pts = [
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [3, 0], [3, 1], [3, 2]
    ]
    return pts

def _make_cross_pattern():
    """
    Cross pattern with extra points to create more collinearities.
    """
    pts = [
        [0, 0], [1, 0], [-1, 0], [2, 0],
        [0, 1], [0, -1], [0, 2],
        [1, 1], [-1, -1], [1, -1],
        [-1, 1], [2, 1]
    ]
    return pts

def _make_lattice_subgrid():
    """
    Subgrid of a larger lattice with strategic point removal.
    """
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2], [3, 2],
        [1, 3]
    ]
    return pts

def _scale_points(pts, scale=10):
    return [[x * scale, y * scale] for x, y in pts]

def _generate_lattice_candidates(size=20):
    """Generate diverse candidate configurations from a lattice."""
    candidates = []
    
    # Random subsets of a grid
    for _ in range(20):
        pts = []
        while len(pts) < 12:
            x = random.randint(-size//2, size//2)
            y = random.randint(-size//2, size//2)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        candidates.append(pts)
    
    # Structured subsets
    for _ in range(10):
        pts = []
        base_x = random.randint(-size//4, size//4)
        base_y = random.randint(-size//4, size//4)
        for i in range(12):
            x = base_x + random.randint(-size//3, size//3)
            y = base_y + random.randint(-size//3, size//3)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        while len(pts) < 12:
            p = [random.randint(-size, size), random.randint(-size, size)]
            if p not in pts:
                pts.append(p)
        candidates.append(pts)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    random.seed(42)
    
    # Generate diverse initial configurations
    seeds = []
    seeds.append(_make_grid(4, 3))
    seeds.append(_make_grid(3, 4))
    seeds.append(_make_three_parallel_lines())
    seeds.append(_make_tetrahedral())
    seeds.append(_make_optimal_structure())
    seeds.append(_make_hexagonal())
    seeds.append(_make_near_pencil())
    seeds.append(_make_staggered())
    seeds.append(_make_diagonal_grid())
    seeds.append(_make_hesse_like())
    seeds.append(_make_cross_pattern())
    seeds.append(_make_lattice_subgrid())
    
    # Scaled versions
    seeds.append(_scale_points(_make_grid(4, 3), 5))
    seeds.append(_scale_points(_make_three_parallel_lines(), 5))
    seeds.append(_scale_points(_make_hesse_like(), 3))
    
    # Random configurations from lattice
    seeds.extend(_generate_lattice_candidates(size=25))
    
    best_pts = None
    best_cnt = float('inf')
    
    for seed in seeds:
        if time.time() > deadline:
            break
        cnt = _count_ordinary_lines(seed)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = [p[:] for p in seed]
        
        # Smart local search
        pts, cnt = _smart_local_search(seed, max_iters=15000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
        
        # Pair swap
        if best_pts and time.time() < deadline:
            pts, cnt = _pair_swap_search(best_pts, max_iters=3000, deadline=deadline)
            if cnt < best_cnt:
                best_cnt = cnt
                best_pts = pts[:]
    
    # Final refinement with more iterations
    if best_pts and time.time() < deadline - 10:
        pts, cnt = _smart_local_search(best_pts, max_iters=50000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
        
        pts, cnt = _pair_swap_search(best_pts, max_iters=10000, deadline=deadline)
        if cnt < best_cnt:
            best_cnt = cnt
            best_pts = pts[:]
    
    # Final polish: try to improve by small perturbations
    if best_pts and time.time() < deadline - 5:
        n = len(best_pts)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            lines = _get_lines(best_pts)
            for idx in range(n):
                if time.time() > deadline - 2:
                    break
                xi, yi = best_pts[idx]
                # Try very small moves
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = [xi + dx, yi + dy]
                        valid = True
                        for j in range(n):
                            if j != idx and best_pts[j] == new_pt:
                                valid = False
                                break
                        if not valid:
                            continue
                        best_pts[idx] = new_pt
                        cnt = _count_ordinary_lines(best_pts)
                        best_pts[idx] = [xi, yi]
                        if cnt < best_cnt:
                            best_pts[idx] = new_pt
                            best_cnt = cnt
                            improved = True
    
    if best_pts is None:
        best_pts = _make_grid(4, 3)
    
    # Verify distinctness
    seen = set()
    final = []
    for p in best_pts:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            final.append(p)
    
    while len(final) < 12:
        p = [random.randint(-10**9, 10**9), random.randint(-10**9, 10**9)]
        if tuple(p) not in seen:
            seen.add(tuple(p))
            final.append(p)
    
    return final
# EVOLVE-BLOCK-END

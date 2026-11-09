"""Construct 12 integer points minimizing ordinary (2-point) lines."""
# EVOLVE-BLOCK-START
import time
from math import gcd

# ---- Fast line counting using (dy, -dx, c) canonical keys ----
def _count_ordinary(pts):
    """Return (n_ordinary, lines_dict) for the given 12 points."""
    n = len(pts)
    lines = {}
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dy = yj - yi
            dx = xj - xi
            if dx == 0 and dy == 0:
                continue
            # Canonical direction: (dy, -dx)
            a, b = dy, -dx
            # Normalize so a > 0 or (a == 0 and b > 0)
            if a < 0 or (a == 0 and b < 0):
                a, b = -a, -b
            g = gcd(abs(a), abs(b)) if (a or b) else 1
            a //= g
            b //= g
            c = a * xi + b * yi
            key = (a, b, c)
            lines[key] = lines.get(key, 0) + 1
    n_ord = sum(1 for v in lines.values() if v == 1)
    return n_ord, lines

def _not_all_collinear(pts):
    x0, y0 = pts[0]
    for j in range(1, len(pts)):
        xj, yj = pts[j]
        for k in range(j + 1, len(pts)):
            xk, yk = pts[k]
            if (xj - x0) * (yk - y0) != (yj - y0) * (xk - x0):
                return True
    return False

# ---- Generate candidate hexagon-based starting configurations ----
def _generate_hex_startings():
    """Generate multiple hexagon+center configurations with integer coords."""
    starts = []
    # Various hexagon shapes and scales
    for scale in [1, 2, 3, 5, 8, 13]:
        # Regular hexagon approximated on integer grid
        # 6 vertices + center (7 points), then add 5 more strategically
        s = scale
        hex_pts = [
            [s, 0],
            [s // 2, s * 7 // 4],
            [-s // 2, s * 7 // 4],
            [-s, 0],
            [-s // 2, -s * 7 // 4],
            [s // 2, -s * 7 // 4],
            [0, 0],  # center
        ]
        # Add 5 more points: use points on the lines through hexagon vertices
        # to maximize collinearities
        # Points along each of the 6 vertex-center lines at 2x distance
        extra_pts = []
        for i in range(6):
            hx, hy = hex_pts[i]
            # Extend beyond the hexagon vertex
            ex, ey = hx * 2, hy * 2
            extra_pts.append([ex, ey])
            if len(extra_pts) >= 5:
                break
        # Fill remaining spots
        while len(extra_pts) < 5:
            extra_pts.append([0, s * 3])
        all_pts = hex_pts + extra_pts[:5]
        # Ensure exactly 12 distinct points
        seen = set()
        unique = []
        for p in all_pts:
            key = (p[0], p[1])
            if key not in seen and len(unique) < 12:
                seen.add(key)
                unique.append([p[0], p[1]])
        if len(unique) == 12 and _not_all_collinear(unique):
            starts.append([p[:] for p in unique])
    
    # Also try a different hexagon orientation (flat-top)
    for scale in [1, 2, 3, 4]:
        s = scale
        hex_pts = [
            [0, s],
            [s * 7 // 4, s // 2],
            [s * 7 // 4, -s // 2],
            [0, -s],
            [-s * 7 // 4, -s // 2],
            [-s * 7 // 4, s // 2],
            [0, 0],
        ]
        extra_pts = []
        for i in range(6):
            hx, hy = hex_pts[i]
            ex, ey = hx * 2, hy * 2
            extra_pts.append([ex, ey])
            if len(extra_pts) >= 5:
                break
        while len(extra_pts) < 5:
            extra_pts.append([s * 3, 0])
        all_pts = hex_pts + extra_pts[:5]
        seen = set()
        unique = []
        for p in all_pts:
            key = (p[0], p[1])
            if key not in seen and len(unique) < 12:
                seen.add(key)
                unique.append([p[0], p[1]])
        if len(unique) == 12 and _not_all_collinear(unique):
            starts.append([p[:] for p in unique])
    
    # Try a "3 lines of 4" configuration (3 parallel lines, 4 points each)
    # This gives each line 6 pairs, all on the line, reducing ordinary lines
    for base in range(-10, 11):
        pts = []
        for row in range(3):
            y = base + row * 3
            for col in range(4):
                pts.append([base + col * 2, y])
        if len(pts) == 12 and _not_all_collinear(pts):
            starts.append([p[:] for p in pts])
        if len(starts) >= 30:
            break
    if len(starts) < 30:
        for base in range(-20, 21):
            pts = []
            for row in range(3):
                y = base + row * 4
                for col in range(4):
                    pts.append([base + col * 3, y])
            if len(pts) == 12 and _not_all_collinear(pts):
                starts.append([p[:] for p in pts])
            if len(starts) >= 30:
                break
    
    return starts

# ---- Greedy local search: move one point to a candidate location ----
def _local_search(pts, time_limit):
    """Greedy local search: try moving each point to candidate locations."""
    start_time = time.time()
    n_ord, _ = _count_ordinary(pts)
    
    # Build candidate positions: intersections of pairs of lines through existing points
    # For efficiency, limit candidates
    def get_candidates(pts):
        candidates = set()
        # Intersections of lines formed by pairs of points
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi = pts[i]
                xj, yj = pts[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    xk, yk = pts[k]
                    for l in range(k + 1, n):
                        if l == i or l == j:
                            continue
                        xl, yl = pts[l]
                        dx2 = xl - xk
                        dy2 = yl - yk
                        if dx2 == 0 and dy2 == 0:
                            continue
                        # Solve intersection of line (i,j) and line (k,l)
                        # Line 1: (x - xi)/dx = (y - yi)/dy
                        # Line 2: (x - xk)/dx2 = (y - yk)/dy2
                        # Cross product check
                        denom = dx * dy2 - dy * dx2
                        if denom == 0:
                            continue  # parallel
                        # Intersection
                        # x = (xi*dy*dx2 - yi*dx*dy2 - xk*dy2*dx + yk*dx2*dy) / (dx*dy2 - dy*dx2)
                        # Using Cramer's rule:
                        # dy*x - dx*y = dy*xi - dx*yi = c1
                        # dy2*x - dx2*y = dy2*xk - dx2*yk = c2
                        c1 = dy * xi - dx * yi
                        c2 = dy2 * xk - dx2 * yk
                        # x = (c1*dx2 - c2*dx) / (dy*dx2 - dy2*dx)  -- wait
                        # From: dy*x - dx*y = c1
                        #       dy2*x - dx2*y = c2
                        # Multiply first by dx2, second by dx:
                        # dy*dx2*x - dx*dx2*y = c1*dx2
                        # dy2*dx*x - dx2*dx*y = c2*dx
                        # Subtract: (dy*dx2 - dy2*dx)*x = c1*dx2 - c2*dx
                        denom_x = dy * dx2 - dy2 * dx
                        if denom_x == 0:
                            continue
                        num_x = c1 * dx2 - c2 * dx
                        if num_x % denom_x != 0:
                            continue
                        ix = num_x // denom_x
                        # y from first equation: dy*x - dx*y = c1 => y = (dy*x - c1)/dx if dx != 0
                        if dx != 0:
                            num_y = dy * ix - c1
                            if num_y % dx != 0:
                                continue
                            iy = num_y // dx
                        elif dy != 0:
                            iy = (ix - xi) * dy // dx if dx != 0 else yi
                            # Actually if dx==0, x is fixed = xi
                            if ix != xi:
                                continue
                            iy = yi  # vertical line
                        else:
                            continue
                        if -10**9 <= ix <= 10**9 and -10**9 <= iy <= 10**9:
                            candidates.add((ix, iy))
        return candidates
    
    # Limit candidate generation for speed
    candidate_pool = get_candidates(pts)
    # Also add some random points
    import random
    rng = random.Random(12345)
    for _ in range(200):
        rx = rng.randint(-100, 100)
        ry = rng.randint(-100, 100)
        candidate_pool.add((rx, ry))
    
    # Greedy: repeatedly try to move the point that gives the best improvement
    max_iter = 500
    for iteration in range(max_iter):
        if time.time() > time_limit:
            break
        best_improvement = 0
        best_move = None  # (idx, new_x, new_y, new_n_ord)
        
        for idx in range(12):
            old_x, old_y = pts[idx]
            for (cx, cy) in candidate_pool:
                if cx == old_x and cy == old_y:
                    continue
                # Check duplicate
                is_dup = False
                for i, (px, py) in enumerate(pts):
                    if i != idx and px == cx and py == cy:
                        is_dup = True
                        break
                if is_dup:
                    continue
                pts[idx] = [cx, cy]
                if not _not_all_collinear(pts):
                    pts[idx] = [old_x, old_y]
                    continue
                new_n_ord, _ = _count_ordinary(pts)
                pts[idx] = [old_x, old_y]
                improvement = n_ord - new_n_ord
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = (idx, cx, cy, new_n_ord)
        
        if best_move is None:
            break
        idx, cx, cy, new_n_ord = best_move
        pts[idx] = [cx, cy]
        n_ord = new_n_ord
        if n_ord <= 6:
            break
    
    return pts, n_ord

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Generate starting configurations
    starts = _generate_hex_startings()
    
    if not starts:
        # Fallback: 3x4 grid
        starts = [[[x, y] for x in range(4) for y in range(3)]]
    
    # Evaluate all starts, keep the best few
    scored = []
    for s in starts:
        n_ord, _ = _count_ordinary(s)
        scored.append((n_ord, s))
    scored.sort(key=lambda x: x[0])
    
    # Take top starting points and refine each
    best_pts = None
    best_score = float('inf')
    
    # Refine the top 5 starts
    num_refine = min(5, len(scored))
    time_per_refine = 25  # seconds per refinement
    
    for i in range(num_refine):
        if time.time() > deadline - 10:
            break
        n_ord, pts = scored[i]
        pts = [p[:] for p in pts]
        refined_pts, refined_score = _local_search(pts, deadline - 10)
        if refined_score < best_score:
            best_score = refined_score
            best_pts = [p[:] for p in refined_pts]
        if best_score <= 6:
            break
    
    # If no good result yet, do a broader search
    if best_score > 8:
        # Try more starting configurations with quick evaluation
        for s in scored[5:]:
            if time.time() > deadline - 5:
                break
            n_ord, _ = _count_ordinary(s)
            if n_ord < best_score:
                best_score = n_ord
                best_pts = [p[:] for p in s]
    
    # Final validation
    assert len(best_pts) == 12
    assert all(isinstance(p, list) and len(p) == 2 for p in best_pts)
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in best_pts)
    assert _not_all_collinear(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END

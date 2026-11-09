"""Search for 20 integer points with no four collinear and a small no-three-collinear subset using constructive + local search."""
import time
import random
from math import gcd

_DEADLINE_SECONDS = 140.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE_SECONDS - _elapsed()


def _collinear(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0


def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
            if g:
                dxn, dyn = dx // g, dy // g
            else:
                dxn, dyn = dx, dy
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn, dyn = -dxn, -dyn
            cnt = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if (points[k][0] - points[i][0]) * dyn == (points[k][1] - points[i][1]) * dxn:
                    cnt += 1
                    if cnt >= 2:
                        return True
    return False


def _max_no_three_collinear(points, timeout=None):
    """Faster branch and bound for largest subset with no three collinear."""
    n = len(points)
    
    # Precompute collinearity: for each pair, which points are collinear with them
    collinear_pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            collinear_pairs[(i, j)] = set()
            for k in range(n):
                if k != i and k != j and _collinear(points[i], points[j], points[k]):
                    collinear_pairs[(i, j)].add(k)
    
    best = [0]
    chosen = [0] * n
    deadline = time.time() + 5.0 if timeout is None else timeout + 5.0
    
    def can_add(i):
        # i must not be collinear with any pair already chosen
        for a in range(n):
            if not chosen[a]:
                continue
            for b in range(a + 1, n):
                if chosen[b] and i in collinear_pairs.get((a, b), set()):
                    return False
        return True
    
    def branch(start, size):
        if time.time() > deadline:
            return
        if size > best[0]:
            best[0] = size
        if start >= n:
            return
        # Upper bound: can we still beat best?
        if size + (n - start) <= best[0]:
            return
        # Try include first (greedy heuristic)
        if can_add(start):
            chosen[start] = 1
            branch(start + 1, size + 1)
            chosen[start] = 0
        # Try exclude
        branch(start + 1, size)
    
    branch(0, 0)
    return best[0]


def _valid(points):
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
            return False
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
    if _has_four_collinear(points):
        return False
    return True


def _construct_parabola_perturbed(rng, seed_offset=0):
    """Construct points on a parabola with small perturbations to avoid 4-in-line."""
    pts = []
    used = set()
    # Use points on y = x^2 with perturbations
    x_values = list(range(-10, 10))
    rng.shuffle(x_values)
    
    for x in x_values:
        base_y = x * x
        found = False
        for _ in range(30):
            px = x + rng.randint(-3, 3)
            py = base_y + rng.randint(-3, 3)
            if -100 <= px <= 100 and -100 <= py <= 100 and (px, py) not in used:
                # Check no four collinear with existing points
                pts.append([px, py])
                if not _has_four_collinear(pts):
                    used.add((px, py))
                    found = True
                    break
                else:
                    pts.pop()
        if not found:
            # Try base point
            if (x, base_y) not in used and -100 <= base_y <= 100:
                pts.append([x, base_y])
                if not _has_four_collinear(pts):
                    used.add((x, base_y))
                else:
                    pts.pop()
            # If still not found, try nearby
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    px, py = x + dx, base_y + dy
                    if -100 <= px <= 100 and -100 <= py <= 100 and (px, py) not in used:
                        pts.append([px, py])
                        if not _has_four_collinear(pts):
                            used.add((px, py))
                            found = True
                            break
                        else:
                            pts.pop()
                if found:
                    break
    
    return pts if len(pts) == 20 else None


def _construct_random_valid(rng):
    """Construct 20 random points ensuring no four collinear."""
    pts = []
    used = set()
    attempts = 0
    while len(pts) < 20 and attempts < 10000:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if (x, y) in used:
            attempts += 1
            continue
        pts.append([x, y])
        if not _has_four_collinear(pts):
            used.add((x, y))
        else:
            pts.pop()
        attempts += 1
    return pts if len(pts) == 20 else None


def _local_search(points, rng, max_iter=200):
    """Local search: try moving points to reduce the no-three-collinear score."""
    pts = [p[:] for p in points]
    current_score = _max_no_three_collinear(pts, timeout=time.time())
    best_score = current_score
    best_pts = [p[:] for p in pts]
    
    for _ in range(max_iter):
        if _remaining() < 15.0:
            break
        
        # Pick a random point to move
        i = rng.randint(0, 19)
        orig_i = pts[i][:]
        improved = False
        
        for _ in range(20):
            new_x = rng.randint(-100, 100)
            new_y = rng.randint(-100, 100)
            if (new_x, new_y) in [tuple(p) for p in pts if p != orig_i]:
                continue
            pts[i] = [new_x, new_y]
            
            if not _has_four_collinear(pts):
                s = _max_no_three_collinear(pts, timeout=time.time())
                if s < best_score:
                    best_score = s
                    best_pts = [p[:] for p in pts]
                    current_score = s
                    improved = True
                    break
        
        if not improved:
            pts[i] = orig_i
    
    return best_pts, best_score


def construct():
    global _START
    _START = time.time()
    
    best_points = None
    best_score = 999
    
    rng = random.Random(42)
    
    # Strategy 1: Parabola perturbed (multiple seeds)
    for seed in range(50):
        if _remaining() < 20.0:
            break
        pts = _construct_parabola_perturbed(rng, seed)
        if pts is not None and not _has_four_collinear(pts):
            s = _max_no_three_collinear(pts, timeout=time.time())
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in pts]
                if best_score <= 12:
                    break
    
    # Strategy 2: Random valid points
    for attempt in range(30):
        if _remaining() < 20.0:
            break
        pts = _construct_random_valid(rng)
        if pts is not None:
            s = _max_no_three_collinear(pts, timeout=time.time())
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in pts]
                if best_score <= 12:
                    break
    
    # Strategy 3: Local search from best found
    if best_points is not None and _remaining() > 30.0:
        improved_pts, improved_score = _local_search(best_points, rng, max_iter=100)
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_pts
    
    # Fallback: simple parabola
    if best_points is None:
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    return best_points

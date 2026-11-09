"""Spectral-aware local search for (16,4) configurations.
Uses structured seeds and optimized incremental scoring with running totals."""
import time
import math
import random
from collections import defaultdict

_START = time.time()
_DEADLINE = _START + 118.0
_BOUND = 1000


def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx, dy = x2 - x1, y2 - y1
    g = _gcd(dx, dy)
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return (dx, dy, x1 * dy - y1 * dx)


def _build_ag24():
    """Construct 16 points in AG(2,4) with 16 lines of 4 points each."""
    points = [[x, y] for x in range(4) for y in range(4)]
    return points


def _seed_staggered():
    pts = []
    for row in range(4):
        offset = row
        for col in range(4):
            pts.append([col + offset, row])
    return pts


def _seed_hexagonal():
    pts = []
    pts.append([0, 0])
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
        pts.append([dx, dy])
    ring2 = [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 1), (-2, -1), (1, 2), (-1, -2)]
    for p in ring2:
        pts.append(list(p))
    return pts[:16]


def _seed_3x3_plus():
    """3x3 grid plus 7 additional points to create more 4-collinear lines."""
    pts = [[x, y] for x in range(3) for y in range(3)]
    # Add points to create diagonal and other 4-lines
    pts.append([3, 0])
    pts.append([0, 3])
    pts.append([3, 3])
    pts.append([-1, 1])
    pts.append([1, -1])
    pts.append([2, -1])
    pts.append([-1, 2])
    return pts[:16]


def _seed_lattice():
    """A lattice-based seed with structured collinearities."""
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i * 2, j * 2 + i])
    return pts


def _compute_line_pts(pts):
    """Compute line_pts: line_key -> set of point indices."""
    n = len(pts)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            if k not in line_pts:
                line_pts[k] = set()
            line_pts[k].add(i)
            line_pts[k].add(j)
    return line_pts


def _score_from_line_pts(line_pts):
    """Compute (n4, n5, n3) from line_pts dict."""
    n4 = n5 = n3 = 0
    for s in line_pts.values():
        c = len(s)
        if c == 4:
            n4 += 1
        elif c >= 5:
            n5 += 1
        elif c == 3:
            n3 += 1
    return n4, n5, n3


def _valid_from_line_pts(line_pts):
    """Check validity: no line has 5+ points."""
    for s in line_pts.values():
        if len(s) >= 5:
            return False
    return True


def _score_fast(points):
    """Fast score: count lines with exactly 4 points, lines with >=5 points, lines with 3."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
    n4 = n5 = n3 = 0
    for c in counts.values():
        if c == 6:
            n4 += 1
        elif c >= 10:
            n5 += 1
        elif c == 3:
            n3 += 1
    return n4, n5, n3


def _valid(points):
    """Check validity: no five collinear."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
            if counts[k] >= 10:
                return False
    return True


def _incremental_score(pts, line_pts, i, new_pos, cur_n4, cur_n5, cur_n3):
    """Compute new score after moving point i to new_pos, using incremental updates.
    Returns (n4, n5, n3) or None if invalid (5+ collinear)."""
    n = len(pts)
    
    # Collect old lines through point i
    old_line_keys = {}
    for j in range(n):
        if j == i:
            continue
        k = _line_key(pts[i], pts[j])
        if k not in old_line_keys:
            old_line_keys[k] = set()
        old_line_keys[k].add(j)
    
    # Collect new lines through new position
    new_line_keys = {}
    for j in range(n):
        if j == i:
            continue
        k = _line_key(new_pos, pts[j])
        if k not in new_line_keys:
            new_line_keys[k] = set()
        new_line_keys[k].add(j)
    
    # Check validity: no line through new_pos has 5+ total points
    for k, others in new_line_keys.items():
        if len(others) + 1 >= 5:
            return None
    
    # Compute score change
    affected = set(old_line_keys.keys()) | set(new_line_keys.keys())
    
    delta_n4 = delta_n5 = delta_n3 = 0
    for k in affected:
        # Determine old size
        if k in old_line_keys:
            old_size = len(old_line_keys[k]) + 1
        else:
            old_size = 0
        
        # Determine new size
        if k in new_line_keys:
            new_size = len(new_line_keys[k]) + 1
        else:
            new_size = 0
        
        # Old contribution
        if old_size == 4:
            delta_n4 -= 1
        elif old_size >= 5:
            delta_n5 -= 1
        elif old_size == 3:
            delta_n3 -= 1
        
        # New contribution
        if new_size == 4:
            delta_n4 += 1
        elif new_size >= 5:
            delta_n5 += 1
        elif new_size == 3:
            delta_n3 += 1
    
    return cur_n4 + delta_n4, cur_n5 + delta_n5, cur_n3 + delta_n3


def _fast_local_search(points, rng, time_budget):
    """Local search with precomputed line structure for fast incremental updates."""
    n = len(points)
    cur = [list(p) for p in points]
    
    line_pts = _compute_line_pts(cur)
    cur_n4, cur_n5, cur_n3 = _score_from_line_pts(line_pts)
    best_key = (cur_n4, -cur_n5, cur_n3)
    best_pts = [list(p) for p in cur]
    step = 0
    
    while time.time() < time_budget:
        improved = False
        
        for i in range(n):
            if time.time() >= time_budget:
                break
            x0, y0 = cur[i]
            
            # Generate candidates: local grid
            candidates = set()
            w = 3
            for dx in range(-w, w + 1):
                for dy in range(-w, w + 1):
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) <= _BOUND and abs(ny) <= _BOUND:
                        candidates.add((nx, ny))
            
            # Add some random candidates at larger distances
            for _ in range(15):
                angle = rng.uniform(0, 2 * math.pi)
                dist = rng.randint(1, 10)
                nx = x0 + int(dist * math.cos(angle))
                ny = y0 + int(dist * math.sin(angle))
                if abs(nx) <= _BOUND and abs(ny) <= _BOUND:
                    candidates.add((nx, ny))
            
            best_local = None
            best_local_key = None
            best_local_score = None
            
            for (nx, ny) in candidates:
                if time.time() >= time_budget:
                    break
                np_ = (nx, ny)
                if any(np_ == tuple(p) for p in cur):
                    continue
                
                # Quick validity check: no line through new_pos can have 5+ points
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    k = _line_key(np_, cur[j])
                    if k in line_pts:
                        existing = line_pts[k]
                        others = len(existing - {i, j})
                        if others + 2 >= 5:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Incremental score using running totals
                new_score = _incremental_score(cur, line_pts, i, np_, cur_n4, cur_n5, cur_n3)
                if new_score is None:
                    continue
                
                n4, n5, n3 = new_score
                key = (n4, -n5, n3)
                
                if best_local_key is None or key > best_local_key:
                    best_local_key = key
                    best_local = np_
                    best_local_score = new_score
            
            if best_local is not None:
                cur_key = (cur_n4, -cur_n5, cur_n3)
                if best_local_key > cur_key:
                    cur[i] = list(best_local)
                    cur_n4, cur_n5, cur_n3 = best_local_score
                    # Update line_pts
                    line_pts = _compute_line_pts(cur)
                    improved = True
                    if (cur_n4, -cur_n5, cur_n3) > best_key:
                        best_key = (cur_n4, -cur_n5, cur_n3)
                        best_pts = [list(p) for p in cur]
        
        step += 1
        
        if not improved and step > 2:
            if time.time() >= time_budget:
                break
            # Random kick: try several random moves, accept if not worse
            i = rng.randrange(n)
            x0, y0 = cur[i]
            for _ in range(300):
                nx = x0 + rng.randint(-8, 8)
                ny = y0 + rng.randint(-8, 8)
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = (nx, ny)
                if any(np_ == tuple(p) for p in cur):
                    continue
                
                # Quick validity
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    k = _line_key(np_, cur[j])
                    if k in line_pts:
                        existing = line_pts[k]
                        others = len(existing - {i, j})
                        if others + 2 >= 5:
                            valid = False
                            break
                if not valid:
                    continue
                
                new_score = _incremental_score(cur, line_pts, i, np_, cur_n4, cur_n5, cur_n3)
                if new_score is None:
                    continue
                
                n4, n5, n3 = new_score
                key = (n4, -n5, n3)
                cur_key = (cur_n4, -cur_n5, cur_n3)
                if key >= cur_key:
                    cur[i] = list(np_)
                    cur_n4, cur_n5, cur_n3 = n4, n5, n3
                    line_pts = _compute_line_pts(cur)
                    if (cur_n4, -cur_n5, cur_n3) > best_key:
                        best_key = (cur_n4, -cur_n5, cur_n3)
                        best_pts = [list(p) for p in cur]
                    break
    
    return (best_key[0], -best_key[1], best_key[2], best_pts)


def construct():
    global _START, _DEADLINE
    rng = random.Random(42)
    
    # Primary seed: AG(2,4) structure
    ag24 = _build_ag24()
    
    # Additional diverse seeds
    seeds = [
        ag24,
        _seed_staggered(),
        _seed_hexagonal(),
        _seed_3x3_plus(),
        _seed_lattice(),
    ]
    
    # Add some perturbed versions of the grid
    for trial in range(5):
        pts = [list(p) for p in ag24]
        for _ in range(3):
            i = rng.randrange(16)
            pts[i] = [pts[i][0] + rng.randint(-2, 2), pts[i][1] + rng.randint(-2, 2)]
        if len(set(map(tuple, pts))) == 16 and _valid(pts):
            seeds.append(pts)
    
    # Dedupe
    seen = set()
    uniq = []
    for s in seeds:
        key = tuple(sorted(map(tuple, s)))
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    
    best_overall = None
    elapsed = time.time() - _START
    remaining = _DEADLINE - _START - elapsed
    per_budget = max(3.0, remaining / max(1, len(uniq)))
    
    for s in uniq:
        if time.time() >= _DEADLINE - 3.0:
            break
        if not _valid(s):
            continue
        t_end = min(_DEADLINE - 2.0, time.time() + per_budget)
        res = _fast_local_search(s, rng, t_end)
        if best_overall is None or (res[0], -res[1], res[2]) > (best_overall[0], -best_overall[1], best_overall[2]):
            best_overall = res
    
    if best_overall is None:
        return ag24
    
    pts = best_overall[3]
    
    # Final safety checks
    if not _valid(pts):
        return ag24
    for p in pts:
        if abs(p[0]) > _BOUND or abs(p[1]) > _BOUND:
            return ag24
    if len(set(map(tuple, pts))) != 16:
        return ag24
    
    return [list(p) for p in pts]

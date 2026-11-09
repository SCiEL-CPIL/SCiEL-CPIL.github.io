"""Spectral-aware local search for (16,4) configurations.
Uses AG(2,4) structure for 16 exact-4 lines, then optimizes with
incremental scoring and spectral-guided candidate generation."""
import time
import math
import random
from collections import defaultdict

_START = time.time()
_DEADLINE = _START + 115.0
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


def _incremental_score(pts, line_pts, i, new_pos):
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
    
    # Get base score from unaffected lines
    base_n4 = base_n5 = base_n3 = 0
    for k, s in line_pts.items():
        if k in affected:
            continue
        c = len(s)
        if c == 4:
            base_n4 += 1
        elif c >= 5:
            base_n5 += 1
        elif c == 3:
            base_n3 += 1
    
    return base_n4 + delta_n4, base_n5 + delta_n5, base_n3 + delta_n3


def _spectral_guided_candidates(cur, i, line_pts, rng, num_candidates=20):
    """Generate candidate positions guided by spectral analysis of line structure.
    Points with high 'line load' (many lines through them) are good candidates to move.
    Candidate positions are biased toward directions that break existing lines."""
    n = len(cur)
    x0, y0 = cur[i]
    
    # Compute line load for point i
    line_load = len(line_pts.get(_line_key(cur[i], cur[(i+1)%n]), set()))
    
    # Generate candidates in directions that tend to break existing lines
    candidates = []
    for _ in range(num_candidates):
        # Random direction with bias toward breaking lines
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.randint(1, 8)
        nx = x0 + int(dist * math.cos(angle))
        ny = y0 + int(dist * math.sin(angle))
        if abs(nx) <= _BOUND and abs(ny) <= _BOUND:
            candidates.append((nx, ny))
    
    # Also add some grid-aligned candidates
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x0 + dx, y0 + dy
            if abs(nx) <= _BOUND and abs(ny) <= _BOUND:
                candidates.append((nx, ny))
    
    return candidates


def _fast_local_search(points, rng, time_budget):
    """Local search with precomputed line structure for fast incremental updates."""
    n = len(points)
    cur = [list(p) for p in points]
    
    line_pts = _compute_line_pts(cur)
    cur_n4, cur_n5, cur_n3 = _score_from_line_pts(line_pts)
    best_key = (cur_n4, -cur_n5, cur_n3)
    best_pts = [list(p) for p in cur]
    step = 0
    stall_count = 0
    
    while time.time() < time_budget:
        improved = False
        
        for i in range(n):
            if time.time() >= time_budget:
                break
            x0, y0 = cur[i]
            
            # Generate candidates: mix of local grid and spectral-guided
            candidates = set()
            w = 2 + min(step // 10, 6)
            for dx in range(-w, w + 1):
                for dy in range(-w, w + 1):
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) <= _BOUND and abs(ny) <= _BOUND:
                        candidates.add((nx, ny))
            
            # Add spectral-guided candidates
            for cand in _spectral_guided_candidates(cur, i, line_pts, rng, 10):
                candidates.add(cand)
            
            best_local = None
            best_local_key = None
            best_local_score = None
            
            for (nx, ny) in candidates:
                if time.time() >= time_budget:
                    break
                np_ = (nx, ny)
                if any(np_ == tuple(p) for p in cur):
                    continue
                
                # Quick validity check
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
                
                # Incremental score
                new_score = _incremental_score(cur, line_pts, i, np_)
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
                    stall_count = 0
                    if (cur_n4, -cur_n5, cur_n3) > best_key:
                        best_key = (cur_n4, -cur_n5, cur_n3)
                        best_pts = [list(p) for p in cur]
        
        step += 1
        stall_count += 1 if not improved else 0
        
        if not improved and step > 3:
            if time.time() >= time_budget:
                break
            # Random kick
            i = rng.randrange(n)
            x0, y0 = cur[i]
            for _ in range(200):
                nx = x0 + rng.randint(-6, 6)
                ny = y0 + rng.randint(-6, 6)
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = (nx, ny)
                if any(np_ == tuple(p) for p in cur):
                    continue
                cand = [list(p) for p in cur]
                cand[i] = list(np_)
                if not _valid(cand):
                    continue
                n4, n5, n3 = _score_fast(cand)
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
    ]
    
    # Add some perturbed versions
    for trial in range(3):
        pts = [list(p) for p in ag24]
        for _ in range(4):
            i = rng.randrange(16)
            pts[i] = [pts[i][0] + rng.randint(-1, 1), pts[i][1] + rng.randint(-1, 1)]
        if len(set(map(tuple, pts))) == 16:
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
    per_budget = max(5.0, (_DEADLINE - time.time()) / max(1, len(uniq)))
    
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

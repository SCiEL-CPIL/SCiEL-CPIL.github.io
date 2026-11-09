"""Improved approach: smarter seeds + faster incremental local search with
two-point swaps, wider exploration, and a final polish pass.
Maximizes lines with exactly four points, no five collinear."""
import time
import math
import random
import itertools

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin before the 150 s kill
_BOUND = 1000


def _gcd(a, b):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a


def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return (dx, dy, x1 * dy - y1 * dx)


def _score_from_counts(counts):
    n4 = n5 = n3 = n2 = 0
    for c in counts.values():
        if c == 6:
            n4 += 1
        elif c >= 10:
            n5 += 1
        elif c == 3:
            n3 += 1
        elif c == 2:
            n2 += 1
    return n4, n5, n3, n2


def _score(points):
    """Returns (num_exact4, num_5plus, num_3, num_2)."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
    return _score_from_counts(counts)


def _valid(points):
    """True iff no five are collinear (no line with >=5 points)."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
            if counts[k] >= 10:
                return False
    return True


def _seed_grid():
    return [[x, y] for x in range(4) for y in range(4)]


def _seed_perturbed_grid():
    pts = [[x, y] for x in range(4) for y in range(4)]
    pts[0] = [0, 0]
    pts[3] = [3, 3]
    pts[4] = [0, 5]
    pts[15] = [4, 3]
    return pts


def _seed_3x3_plus():
    pts = [[x, y] for x in range(3) for y in range(3)]
    extra = [(3, 1), (-1, 1), (1, 3), (1, -1), (3, 3), (-1, -1), (4, 0)]
    for e in extra:
        pts.append(list(e))
    return pts


def _seed_lattice():
    base = [[x, y] for x in range(5) for y in range(5)]
    random.shuffle(base)
    return base[:16]


def _seed_diagonal_grid():
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i, j])
    pts[5] = [1, 5]
    pts[10] = [2, -1]
    pts[15] = [5, 3]
    return pts


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


def _seed_affine_grid():
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x + y, y])
    return pts


def _seed_mixed():
    pts = []
    for i in range(4):
        for j in range(4):
            if i % 2 == 0:
                pts.append([i, j])
            else:
                pts.append([i + 1, j])
    return pts


def _seed_sheared():
    """Shear transformation: (x, y) -> (x + 2*y, y)"""
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x + 2 * y, y])
    return pts


def _seed_rotated_grid():
    """Slightly rotated grid using integer approximations."""
    pts = []
    for x in range(4):
        for y in range(4):
            # Approximate 15-degree rotation
            rx = x - y // 4
            ry = y + x // 4
            pts.append([rx, ry])
    # Ensure distinctness
    seen = set()
    unique = []
    for p in pts:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique.append(p)
    while len(unique) < 16:
        x = len(unique) % 5
        y = len(unique) // 5
        if (x, y) not in seen:
            seen.add((x, y))
            unique.append([x, y])
    return unique[:16]


def _seed_fibonacci():
    """Use Fibonacci-like spacing for more diagonal lines."""
    fib = [1, 1, 2, 3, 5, 8, 13]
    pts = []
    for i in range(16):
        x = fib[i % 7]
        y = fib[(i * 3 + 1) % 7]
        if (x, y) not in [tuple(p) for p in pts]:
            pts.append([x, y])
    while len(pts) < 16:
        x = len(pts)
        y = len(pts) * 2
        if (x, y) not in [tuple(p) for p in pts]:
            pts.append([x, y])
    return pts


def _seed_cross():
    """Cross pattern with extra points."""
    pts = [[0, 0]]
    for r in range(1, 5):
        pts.append([r, 0])
        pts.append([-r, 0])
        pts.append([0, r])
        pts.append([0, -r])
    return pts[:16]


def _seed_parabola():
    """Points on/near a parabola."""
    pts = []
    for x in range(-3, 5):
        pts.append([x, x * x])
    # Add some off-parabola points
    extra = [(0, 1), (1, 2), (-1, 2), (2, 3), (-2, 3)]
    for e in extra:
        if len(pts) >= 16:
            break
        if e not in [tuple(p) for p in pts]:
            pts.append(list(e))
    while len(pts) < 16:
        x = len(pts) - 8
        y = x * x + 1
        if (x, y) not in [tuple(p) for p in pts]:
            pts.append([x, y])
    return pts[:16]


def _local_search(points, rng, time_budget):
    """Simulated Annealing local search with incremental scoring."""
    n = len(points)
    cur = [list(p) for p in points]
    cur_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(cur[i], cur[j])
            cur_counts[k] = cur_counts.get(k, 0) + 1
    cur_n4, cur_n5, cur_n3, cur_n2 = _score_from_counts(cur_counts)
    cur_val = cur_n4 - 100 * cur_n5
    cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
    
    best_val = cur_val
    best_key = cur_key
    best_pts = [list(p) for p in cur]
    
    T = 5.0
    
    while time.time() < time_budget:
        T *= 0.99
        if T < 0.01: T = 0.01
        
        i = rng.randrange(n)
        x0, y0 = cur[i]
        w = 8
        
        best_local_val = None
        best_local_key = None
        best_local = None
        best_local_counts = None
        
        old_lines = {}
        for j in range(n):
            if j == i: continue
            k = _line_key(cur[i], cur[j])
            old_lines[j] = k
            
        for dx in range(-w, w + 1):
            for dy in range(-w, w + 1):
                nx, ny = x0 + dx, y0 + dy
                if abs(nx) > _BOUND or abs(ny) > _BOUND: continue
                np_ = [nx, ny]
                if any(np_ == p for p in cur): continue
                
                new_counts = dict(cur_counts)
                
                for j in range(n):
                    if j == i: continue
                    k = old_lines[j]
                    new_counts[k] -= 1
                    if new_counts[k] == 0: del new_counts[k]
                    
                valid = True
                for j in range(n):
                    if j == i: continue
                    k = _line_key(np_, cur[j])
                    new_counts[k] = new_counts.get(k, 0) + 1
                    if new_counts[k] >= 10:
                        valid = False
                        break
                if not valid: continue
                
                n4, n5, n3, n2 = _score_from_counts(new_counts)
                val = n4 - 100 * n5
                key = (n4, -n5, n3, n2)
                
                if best_local_val is None or val > best_local_val:
                    best_local_val = val
                    best_local_key = key
                    best_local = np_
                    best_local_counts = new_counts
        
        if best_local is not None:
            delta = best_local_val - cur_val
            if delta >= 0 or rng.random() < math.exp(delta / T):
                cur[i] = best_local
                cur_counts = best_local_counts
                cur_val = best_local_val
                cur_key = best_local_key
                
                if cur_val > best_val:
                    best_val = cur_val
                    best_key = cur_key
                    best_pts = [list(p) for p in cur]
                    
        # Kick
        if rng.random() < 0.05:
            i2 = rng.randrange(n)
            if i2 == i: i2 = (i2 + 1) % n
            x1, y1 = cur[i2]
            nx1 = x1 + rng.randint(-10, 10)
            ny1 = y1 + rng.randint(-10, 10)
            if abs(nx1) <= _BOUND and abs(ny1) <= _BOUND:
                if not any([nx1, ny1] == p for p in cur):
                    cand = [list(p) for p in cur]
                    cand[i2] = [nx1, ny1]
                    if _valid(cand):
                        n4, n5, n3, n2 = _score(cand)
                        val = n4 - 100 * n5
                        if val > cur_val - 20:
                            cur[i2] = [nx1, ny1]
                            cur_counts = {}
                            for a in range(n):
                                for b in range(a + 1, n):
                                    k = _line_key(cur[a], cur[b])
                                    cur_counts[k] = cur_counts.get(k, 0) + 1
                            cur_val = val
                            cur_key = (n4, -n5, n3, n2)
                            if cur_val > best_val:
                                best_val = cur_val
                                best_key = cur_key
                                best_pts = [list(p) for p in cur]
                                
    return (best_key[0], -best_key[1], best_key[2], best_key[3], best_pts)


def _polish(points, rng, time_budget):
    """Final high-precision polish pass with SA."""
    n = len(points)
    cur = [list(p) for p in points]
    cur_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(cur[i], cur[j])
            cur_counts[k] = cur_counts.get(k, 0) + 1
    cur_n4, cur_n5, cur_n3, cur_n2 = _score_from_counts(cur_counts)
    cur_val = cur_n4 - 100 * cur_n5
    cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
    
    best_val = cur_val
    best_key = cur_key
    best_pts = [list(p) for p in cur]
    
    T = 1.0
    
    while time.time() < time_budget:
        T *= 0.99
        if T < 0.001: T = 0.001
        
        i = rng.randrange(n)
        x0, y0 = cur[i]
        w = 15
        
        best_local_val = None
        best_local_key = None
        best_local = None
        best_local_counts = None
        
        old_lines = {}
        for j in range(n):
            if j == i: continue
            k = _line_key(cur[i], cur[j])
            old_lines[j] = k
            
        for dx in range(-w, w + 1):
            for dy in range(-w, w + 1):
                nx, ny = x0 + dx, y0 + dy
                if abs(nx) > _BOUND or abs(ny) > _BOUND: continue
                np_ = [nx, ny]
                if any(np_ == p for p in cur): continue
                
                new_counts = dict(cur_counts)
                
                for j in range(n):
                    if j == i: continue
                    k = old_lines[j]
                    new_counts[k] -= 1
                    if new_counts[k] == 0: del new_counts[k]
                    
                valid = True
                for j in range(n):
                    if j == i: continue
                    k = _line_key(np_, cur[j])
                    new_counts[k] = new_counts.get(k, 0) + 1
                    if new_counts[k] >= 10:
                        valid = False
                        break
                if not valid: continue
                
                n4, n5, n3, n2 = _score_from_counts(new_counts)
                val = n4 - 100 * n5
                key = (n4, -n5, n3, n2)
                
                if best_local_val is None or val > best_local_val:
                    best_local_val = val
                    best_local_key = key
                    best_local = np_
                    best_local_counts = new_counts
        
        if best_local is not None:
            delta = best_local_val - cur_val
            if delta >= 0 or rng.random() < math.exp(delta / T):
                cur[i] = best_local
                cur_counts = best_local_counts
                cur_val = best_local_val
                cur_key = best_local_key
                
                if cur_val > best_val:
                    best_val = cur_val
                    best_key = cur_key
                    best_pts = [list(p) for p in cur]
                    
    return (best_key[0], -best_key[1], best_key[2], best_key[3], best_pts)


def construct():
    global _START, _DEADLINE
    rng = random.Random(12345)
    
    seeds = [
        _seed_grid(),
        _seed_perturbed_grid(),
        _seed_3x3_plus(),
        _seed_lattice(),
        _seed_diagonal_grid(),
        _seed_staggered(),
        _seed_hexagonal(),
        _seed_affine_grid(),
        _seed_mixed(),
        _seed_sheared(),
        _seed_rotated_grid(),
        _seed_fibonacci(),
        _seed_cross(),
        _seed_parabola(),
    ]
    
    # Dedupe
    seen = set()
    uniq = []
    for s in seeds:
        key = tuple(sorted(map(tuple, s)))
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    
    # Add random seeds
    for _ in range(10):
        pts = []
        for _ in range(16):
            x = rng.randint(-20, 20)
            y = rng.randint(-20, 20)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        while len(pts) < 16:
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        key = tuple(sorted(map(tuple, pts)))
        if key not in seen:
            seen.add(key)
            uniq.append(pts)
    
    best_overall = None
    # Allocate more time per seed for the first few (structured) seeds
    per_budget_base = max(2.5, (_DEADLINE - time.time()) / max(1, len(uniq)))
    
    for idx, s in enumerate(uniq):
        if time.time() >= _DEADLINE - 5.0:
            break
        if not _valid(s):
            continue
        # Give structured seeds (first 14) slightly more time
        if idx < 14:
            t_end = min(_DEADLINE - 3.0, time.time() + per_budget_base * 1.2)
        else:
            t_end = min(_DEADLINE - 3.0, time.time() + per_budget_base)
        res = _local_search(s, rng, t_end)
        if best_overall is None or (res[0], -res[1], res[2], res[3]) > (best_overall[0], -best_overall[1], best_overall[2], best_overall[3]):
            best_overall = res
    
    # Final polish on the best configuration
    if best_overall is not None and time.time() < _DEADLINE - 5.0:
        pts = best_overall[4]
        polish_budget = min(_DEADLINE - 3.0, time.time() + 15.0)
        polish_res = _polish(pts, rng, polish_budget)
        if (polish_res[0], -polish_res[1], polish_res[2], polish_res[3]) > (best_overall[0], -best_overall[1], best_overall[2], best_overall[3]):
            best_overall = polish_res
    
    if best_overall is None:
        return _seed_grid()
    
    pts = best_overall[4]
    
    # Final safety checks
    if not _valid(pts):
        return _seed_grid()
    for p in pts:
        if abs(p[0]) > _BOUND or abs(p[1]) > _BOUND:
            return _seed_grid()
    if len(set(map(tuple, pts))) != 16:
        return _seed_grid()
    
    return [list(p) for p in pts]

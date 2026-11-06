import time
import random
from itertools import combinations

_DEADLINE = 110.0

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _valid_no_four(points):
    n = len(points)
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
        if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
            return False
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                return False
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes[(dx, dy)] = slopes.get((dx, dy), 0) + 1
            if slopes[(dx, dy)] >= 3:
                return False
    return True

def _build_triples(points):
    """Return list of frozensets of size-3 collinear index sets."""
    n = len(points)
    triples = set()
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                continue
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes.setdefault((dx, dy), []).append(j)
        for k in slopes.values():
            for a, b in combinations(k, 2):
                triples.add(frozenset((i, a, b)))
    return list(triples)

def _max_3collinear_free(points, deadline_start):
    """Branch and bound for largest subset with no three collinear.
    Returns best size found (exact if not cut off by deadline)."""
    n = len(points)
    triples = _build_triples(points)
    allmask = (1 << n) - 1
    
    # Precompute for each vertex the list of triples (as bitmask) containing it
    vtrip = [[] for _ in range(n)]
    triple_masks = []
    for t in triples:
        m = 0
        for v in t:
            m |= (1 << v)
        triple_masks.append(m)
        for v in t:
            vtrip[v].append(m)
    
    # Degree of each vertex (number of triples it participates in)
    degrees = [len(vtrip[v]) for v in range(n)]
    
    best = [0]
    
    # Order vertices by degree descending for pruning
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    # For quick lookup: given a set of included vertices (mask), 
    # which vertices are "blocked" (would complete a triple)?
    # We'll track this incrementally.
    
    def dfs(idx, included_mask, count, blocked_mask):
        if time.time() - deadline_start > 109.0:
            return
        # Upper bound: count + remaining vertices
        remaining = n - idx
        if count + remaining <= best[0]:
            return
        if idx == n:
            if count > best[0]:
                best[0] = count
            return
        
        v = order[idx]
        
        # Option 1: exclude v
        dfs(idx + 1, included_mask, count, blocked_mask)
        
        # Option 2: include v if not blocked
        if not (blocked_mask & (1 << v)):
            # Including v blocks any vertex w that forms a triple with v and one other included vertex
            new_blocked = blocked_mask
            for m in vtrip[v]:
                others = m & ~(1 << v)
                # The other two vertices in this triple: if exactly one is already included, 
                # then including v would complete the triple, so that other is already blocked.
                # If neither is included, then including v means the other two become "partially blocked"
                # but we handle this by: when we include v, we need to ensure no triple containing v is fully selected.
                # A triple {v, a, b} is violated if a and b are both in included_mask OR if one is in included_mask 
                # and the other is later included.
                # Simpler: when including v, we must exclude all pairs {a,b} such that {v,a,b} is a triple.
                # This means a and b cannot both be in the final set.
                # For B&B: when we include v, we add a constraint that for each triple {v,a,b}, 
                # we cannot have both a and b in the remaining selection.
                # This is complex. Let's use a different approach:
                # When including v, we check if any triple containing v has both other vertices already in included_mask.
                # If so, we can't include v.
                # If not, we include v and proceed. But we also need to prevent future selections from completing triples.
                # Standard approach: maintain a set of "forbidden pairs" or use a different state.
                
                # Let's use the approach from inspiration: 
                # A vertex v can be included iff none of its triples are already "completed" by two other included vertices.
                # When we include v, we don't immediately block others, but when we later try to include w,
                # we check if including w would complete any triple with two already-included vertices.
                pass
            
            # Check if including v completes any triple with two already-included vertices
            ok = True
            for m in vtrip[v]:
                others = m & ~(1 << v)
                if (included_mask & others) == others:
                    ok = False
                    break
            
            if ok:
                dfs(idx + 1, included_mask | (1 << v), count + 1, blocked_mask)
    
    dfs(0, 0, 0, 0)
    return best[0]

def _score(points):
    return _max_3collinear_free(points, time.time())

def _triple_count(points):
    """Count number of collinear triples (for heuristic)."""
    n = len(points)
    count = 0
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                continue
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes[(dx, dy)] = slopes.get((dx, dy), 0) + 1
        for k in slopes.values():
            if k >= 3:
                count += combinations(k, 2)
    return count

def _greedy_construct(rng, start_time):
    """Greedy construction: start with a small set and extend lines."""
    pts = []
    for _ in range(4):
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        if (x, y) not in pts:
            pts.append((x, y))
    
    while len(pts) < 20:
        if time.time() - start_time > 105.0:
            break
        best_pt = None
        best_gain = -1
        for _ in range(50):
            x = rng.randint(-10, 10)
            y = rng.randint(-10, 10)
            if (x, y) in pts:
                continue
            new_pts = pts + [(x, y)]
            if not _valid_no_four(new_pts):
                continue
            tc = _triple_count(new_pts)
            gain = tc - _triple_count(pts)
            if gain > best_gain:
                best_gain = gain
                best_pt = (x, y)
        if best_pt is None:
            while True:
                x = rng.randint(-10, 10)
                y = rng.randint(-10, 10)
                if (x, y) not in pts:
                    best_pt = (x, y)
                    break
        pts.append(best_pt)
    
    return pts

def construct():
    start = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = 21
    
    def consider(pts):
        nonlocal best_pts, best_score
        if not _valid_no_four(pts):
            return
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_pts = list(pts)
    
    # Strategy A: Perturbed 5x4 grid
    base = [(x, y) for x in range(0, 5) for y in range(0, 4)]
    for attempt in range(15000):
        if time.time() - start > 100.0:
            break
        pts = list(base)
        for _ in range(rng.randint(3, 8)):
            idx = rng.randrange(20)
            dx = rng.randint(-3, 3)
            dy = rng.randint(-3, 3)
            x, y = pts[idx]
            nx, ny = x + dx, y + dy
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                pts[idx] = (nx, ny)
        consider(pts)
        if best_score <= 12:
            break
    
    # Strategy B: Simulated annealing local search
    if best_pts:
        current_pts = list(best_pts)
        current_score = best_score
        current_tc = _triple_count(current_pts)
        
        temp = 5.0
        for step in range(30000):
            if time.time() - start > 108.0:
                break
            temp *= 0.9995
            
            idx = rng.randrange(20)
            x, y = current_pts[idx]
            
            best_move = None
            best_tc = current_tc
            for _ in range(8):
                nx = x + rng.randint(-3, 3)
                ny = y + rng.randint(-3, 3)
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    new_pt = (nx, ny)
                    if new_pt not in current_pts:
                        old_pt = current_pts[idx]
                        current_pts[idx] = new_pt
                        if _valid_no_four(current_pts):
                            tc = _triple_count(current_pts)
                            delta_tc = tc - current_tc
                            if delta_tc < 0 or rng.random() < pow(2.718281828, -delta_tc / max(temp, 0.1)):
                                if tc < best_tc:
                                    best_tc = tc
                                    best_move = new_pt
                        current_pts[idx] = old_pt
            
            if best_move is not None:
                old_pt = current_pts[idx]
                current_pts[idx] = best_move
                current_tc = best_tc
                new_score = _score(current_pts)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_pts = list(current_pts)
                elif rng.random() < pow(2.718281828, -(new_score - current_score) / max(temp, 0.1)):
                    current_score = new_score
            else:
                pass
    
    # Strategy C: Random small box
    for attempt in range(20000):
        if time.time() - start > 109.0:
            break
        pts = []
        for _ in range(20):
            pts.append((rng.randint(-8, 8), rng.randint(-8, 8)))
        consider(pts)
        if best_score <= 12:
            break
    
    # Strategy D: Structured configurations
    configs = [
        [(x, y) for x in range(5) for y in range(4) if not (x == 4 and y == 0)] + [(4, 10)],
        [(x, y + (x % 2)) for x in range(5) for y in range(4)],
        [(x, y + (x // 2)) for x in range(5) for y in range(4)],
    ]
    for cfg in configs:
        if time.time() - start > 109.5:
            break
        consider(cfg)
        if best_score <= 12:
            break
    
    # Strategy E: Greedy line-extending construction
    for attempt in range(5):
        if time.time() - start > 109.5:
            break
        pts = _greedy_construct(rng, start)
        if len(pts) == 20:
            consider(pts)
            if best_score <= 12:
                break
    
    if best_pts is None:
        return [[x, x * x] for x in range(-10, 10)]
    
    return [list(p) for p in best_pts]

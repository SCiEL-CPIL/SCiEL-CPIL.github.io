import time
import random
from itertools import combinations

_DEADLINE = 110.0

def _elapsed(start):
    return time.time() - start

def _is_collinear(p, q, r):
    (x1, y1), (x2, y2), (x3, y3) = p, q, r
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

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

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _build_triples(points):
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
    return [tuple(t) for t in triples]

def _min_hitting_set(triples_list, n, deadline_start):
    if not triples_list:
        return 0
    
    # Precompute for each vertex, the set of triple indices it belongs to
    v_to_triples = [set() for _ in range(n)]
    for idx, t in enumerate(triples_list):
        for v in t:
            v_to_triples[v].add(idx)
    
    best = [n]
    
    # Initial active triples
    initial_active = set(range(len(triples_list)))
    
    # DFS
    def dfs(excluded_count, active_triples):
        if time.time() - deadline_start > 109.0:
            return
        if not active_triples:
            if excluded_count < best[0]:
                best[0] = excluded_count
            return
        
        if excluded_count >= best[0]:
            return
        
        # Heuristic: Pick a triple with minimum number of "available" vertices?
        # All vertices are available (we haven't excluded them yet in this branch).
        # But some vertices might be "better" to exclude because they hit more triples.
        # Let's pick a triple, then order vertices in it by degree in the *remaining* hypergraph?
        # Simple heuristic: Pick the first active triple.
        # Better: Pick the active triple that has a vertex with highest degree?
        
        # Let's just pick the first one for now, it's fast enough.
        t_idx = next(iter(active_triples))
        t = triples_list[t_idx]
        
        # Sort vertices in t by their degree in the *current* active hypergraph
        # Degree of v = number of active triples containing v
        # We can approximate or compute.
        # To keep it fast, just iterate.
        
        for v in t:
            new_excluded_count = excluded_count + 1
            if new_excluded_count >= best[0]:
                continue
            
            new_active = active_triples - v_to_triples[v]
            dfs(new_excluded_count, new_active)

    dfs(0, initial_active)
    return best[0]

def _score(points):
    triples = _build_triples(points)
    n = len(points)
    hs_size = _min_hitting_set(triples, n, time.time())
    return n - hs_size

def construct():
    start = time.time()
    rng = random.Random(12345)

    best_pts = None
    best_score = 21

    def consider(pts):
        nonlocal best_pts, best_score
        if not _valid_no_four(pts):
            return
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_pts = pts

    # Strategy A: Perturbed 5x4 grid
    base = [(x, y) for x in range(0, 5) for y in range(0, 4)]
    for attempt in range(20000):
        if time.time() - start > 105.0:
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

    # Strategy B: Local Search on best found so far
    if best_pts:
        current_pts = list(best_pts)
        current_score = best_score
        for step in range(5000):
            if time.time() - start > 108.0:
                break
            # Pick a random point
            idx = rng.randrange(20)
            x, y = current_pts[idx]
            # Try moving to a nearby empty spot
            found_move = False
            for _ in range(10):
                nx = x + rng.randint(-2, 2)
                ny = y + rng.randint(-2, 2)
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    new_pt = (nx, ny)
                    if new_pt not in current_pts:
                        old_pt = current_pts[idx]
                        current_pts[idx] = new_pt
                        if _valid_no_four(current_pts):
                            new_score = _score(current_pts)
                            if new_score < current_score:
                                current_score = new_score
                                if current_score < best_score:
                                    best_score = current_score
                                    best_pts = list(current_pts)
                                found_move = True
                                break
                        current_pts[idx] = old_pt
            if not found_move:
                # Maybe just skip
                pass

    # Strategy C: Random small box
    for attempt in range(30000):
        if time.time() - start > 109.5:
            break
        pts = []
        for _ in range(20):
            pts.append((rng.randint(-6, 6), rng.randint(-6, 6)))
        consider(pts)
        if best_score <= 12:
            break

    if best_pts is None:
        return [[x, x * x] for x in range(-10, 10)]

    return [list(p) for p in best_pts]

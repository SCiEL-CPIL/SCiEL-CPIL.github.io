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

def _build_conflict_graph(points):
    """Build conflict graph: edge between i,j if they share a collinear triple.
    Returns list of bitmasks where conflict_mask[i] has bit j set if i and j conflict."""
    n = len(points)
    conflict_mask = [0] * n
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
            if len(k) >= 2:
                for a_idx in range(len(k)):
                    for b_idx in range(a_idx + 1, len(k)):
                        a, b = k[a_idx], k[b_idx]
                        conflict_mask[i] |= (1 << a) | (1 << b)
                        conflict_mask[a] |= (1 << i) | (1 << b)
                        conflict_mask[b] |= (1 << i) | (1 << a)
    return conflict_mask

def _popcount(x):
    return bin(x).count('1')

def _max_independent_set(conflict_mask, n, deadline_start):
    """Find maximum independent set in the conflict graph using branch and bound.
    An independent set in the conflict graph = set with no three collinear."""
    best = [0]
    all_mask = (1 << n) - 1
    
    # Degree of each vertex in conflict graph
    degrees = [_popcount(conflict_mask[i]) for i in range(n)]
    
    # Order vertices by degree descending
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    def dfs(available, count):
        if time.time() - deadline_start > 109.0:
            return
        if count + _popcount(available) <= best[0]:
            return
        if available == 0:
            if count > best[0]:
                best[0] = count
            return
        
        # Pick vertex with max degree in available subgraph
        v = None
        max_deg = -1
        for idx in order:
            if available & (1 << idx):
                deg = _popcount(conflict_mask[idx] & available)
                if deg > max_deg:
                    max_deg = deg
                    v = idx
        if v is None:
            return
        
        # Option 1: exclude v
        dfs(available & ~(1 << v), count)
        
        # Option 2: include v, remove v and its neighbors
        new_available = available & ~(1 << v) & ~conflict_mask[v]
        dfs(new_available, count + 1)
    
    dfs(all_mask, 0)
    return best[0]

def _score(points):
    n = len(points)
    conflict_mask = _build_conflict_graph(points)
    return _max_independent_set(conflict_mask, n, time.time())

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
    # Start with a few points
    pts = []
    # Pick initial points to create some collinear structure
    for _ in range(4):
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        if (x, y) not in pts:
            pts.append((x, y))
    
    while len(pts) < 20:
        if time.time() - start_time > 105.0:
            break
        # Try to add a point that creates a new collinear triple
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
            # Just add a random point
            while True:
                x = rng.randint(-10, 10)
                y = rng.randint(-10, 10)
                if (x, y) not in pts:
                    best_pt = (x, y)
                    break
        pts.append(best_pt)
    
    return pts

def _sa_local_search(pts, rng, start_time, max_steps=5000):
    """Simulated Annealing to maximize triple count."""
    current_pts = [tuple(p) for p in pts]
    current_tc = _triple_count(current_pts)
    best_pts = list(current_pts)
    best_tc = current_tc
    best_score = _score(current_pts)
    
    temp = 2.0
    cooling = 0.999
    
    for step in range(max_steps):
        if time.time() - start_time > 108.0:
            break
        
        # Pick a random point to move
        idx = rng.randrange(20)
        old_x, old_y = current_pts[idx]
        
        # Try a few random new positions
        accepted = False
        for _ in range(5):
            new_x = old_x + rng.randint(-3, 3)
            new_y = old_y + rng.randint(-3, 3)
            if not (-100 <= new_x <= 100 and -100 <= new_y <= 100):
                continue
            new_pt = (new_x, new_y)
            if new_pt in current_pts:
                continue
            
            # Tentatively move
            current_pts[idx] = new_pt
            
            if not _valid_no_four(current_pts):
                current_pts[idx] = (old_x, old_y)
                continue
            
            new_tc = _triple_count(current_pts)
            delta = new_tc - current_tc
            
            # Accept if better, or with probability exp(-delta/temp) if worse
            if delta > 0 or rng.random() < pow(2.718281828, -delta / max(temp, 0.1)):
                current_tc = new_tc
                accepted = True
                
                # Update best
                if new_tc > best_tc:
                    best_tc = new_tc
                    best_pts = list(current_pts)
                    
                # Periodically check score (expensive)
                if step % 50 == 0:
                    s = _score(current_pts)
                    if s < best_score:
                        best_score = s
                        best_pts = list(current_pts)
                break
        
        if not accepted:
            # If no move accepted, maybe try a bigger jump or just cool
            pass
        
        temp *= cooling

    return best_pts, best_score

def _structured_greedy(rng, start_time):
    """
    Structured greedy: build points by extending lines through existing pairs,
    maximizing new collinear triples while maintaining no-4-collinear constraint.
    This creates denser collinear structures than random search.
    """
    pts = []
    
    # Start with a small seed that has some structure
    seed_options = [
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (2, 1), (3, 1), (1, 2)],
        [(0, 0), (2, 0), (1, 1), (3, 1), (0, 2)],
    ]
    seed = list(rng.choice(seed_options))
    for p in seed:
        if p not in pts:
            pts.append(p)
    
    while len(pts) < 20:
        if time.time() - start_time > 100.0:
            break
        
        best_pt = None
        best_gain = -1
        best_tc = _triple_count(pts)
        
        # Generate candidates: points on lines through existing pairs
        candidates = set()
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                p1 = pts[i]
                p2 = pts[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = _gcd(dx, dy) if (dx != 0 or dy != 0) else 1
                dx_n = dx // g if g != 0 else dx
                dy_n = dy // g if g != 0 else dy
                
                # Check nearby points on this line
                for t in range(-3, 4):
                    nx = p1[0] + t * dx_n
                    ny = p1[1] + t * dy_n
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        cand = (nx, ny)
                        if cand not in pts:
                            candidates.add(cand)
        
        # Also add some random candidates
        for _ in range(30):
            x = rng.randint(-15, 15)
            y = rng.randint(-15, 15)
            if (x, y) not in pts:
                candidates.add((x, y))
        
        # Evaluate candidates
        for cand in candidates:
            if cand in pts:
                continue
            new_pts = pts + [cand]
            if not _valid_no_four(new_pts):
                continue
            tc = _triple_count(new_pts)
            gain = tc - best_tc
            if gain > best_gain:
                best_gain = gain
                best_pt = cand
        
        if best_pt is None:
            # Fallback: add random valid point
            found = False
            for _ in range(100):
                x = rng.randint(-15, 15)
                y = rng.randint(-15, 15)
                if (x, y) not in pts:
                    test_pts = pts + [(x, y)]
                    if _valid_no_four(test_pts):
                        best_pt = (x, y)
                        found = True
                        break
            if not found:
                # Last resort
                for x in range(-15, 16):
                    for y in range(-15, 16):
                        if (x, y) not in pts:
                            test_pts = pts + [(x, y)]
                            if _valid_no_four(test_pts):
                                best_pt = (x, y)
                                found = True
                                break
                    if found:
                        break
                if not found:
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

    # Strategy 1: Structured greedy construction (new approach)
    for attempt in range(3):
        if time.time() - start > 30.0:
            break
        pts = _structured_greedy(rng, start)
        if len(pts) == 20:
            consider(pts)
            # Refine with SA
            improved_pts, improved_score = _sa_local_search(pts, rng, start, max_steps=3000)
            consider(improved_pts)
    
    # Strategy 2: Start with a dense grid and perturb
    base = [(x, y) for x in range(0, 5) for y in range(0, 4)]
    
    candidates = []
    for attempt in range(30):
        if time.time() - start > 45.0:
            break
        pts = [tuple(p) for p in base]
        for _ in range(10):
            idx = rng.randrange(20)
            dx = rng.randint(-2, 2)
            dy = rng.randint(-2, 2)
            x, y = pts[idx]
            nx, ny = x + dx, y + dy
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                pts[idx] = (nx, ny)
        
        if _valid_no_four(pts):
            candidates.append(pts)
    
    if candidates:
        best_candidate = None
        best_candidate_tc = -1
        for c in candidates:
            tc = _triple_count(c)
            if tc > best_candidate_tc:
                best_candidate_tc = tc
                best_candidate = c
        if best_candidate:
            consider(best_candidate)
            improved_pts, improved_score = _sa_local_search(best_candidate, rng, start, max_steps=5000)
            consider(improved_pts)
    
    # Strategy 3: More SA restarts if time allows
    if best_score > 12:
        for _ in range(5):
            if time.time() - start > 95.0:
                break
            # Perturb the current best
            if best_pts is None:
                pts = []
                while len(pts) < 20:
                    x, y = rng.randint(-10, 10), rng.randint(-10, 10)
                    if (x, y) not in pts:
                        test_pts = pts + [(x, y)]
                        if _valid_no_four(test_pts):
                            pts.append((x, y))
            else:
                pts = [tuple(p) for p in best_pts]
                for _ in range(3):
                    idx = rng.randrange(20)
                    dx = rng.randint(-2, 2)
                    dy = rng.randint(-2, 2)
                    x, y = pts[idx]
                    nx, ny = x + dx, y + dy
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        pts[idx] = (nx, ny)
            
            if _valid_no_four(pts):
                improved_pts, improved_score = _sa_local_search(pts, rng, start, max_steps=2000)
                consider(improved_pts)
    
    if best_pts is None:
        return [[x, x * x] for x in range(-10, 10)]
    
    return [list(p) for p in best_pts]

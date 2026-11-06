import time
import random
from itertools import combinations

_DEADLINE = 115.0

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
            if g == 0:
                return False
            dx //= g
            dy //= g
            slopes[(dx, dy)] = slopes.get((dx, dy), 0) + 1
            if slopes[(dx, dy)] >= 3:
                return False
    return True

def _build_conflict_graph(points):
    """Build conflict graph: edge between i,j if they share a collinear triple."""
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
            if g == 0:
                continue
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

def _triple_count(points):
    """Count collinear triples for heuristic."""
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
            if g == 0:
                continue
            dx //= g
            dy //= g
            slopes[(dx, dy)] = slopes.get((dx, dy), 0) + 1
        for k in slopes.values():
            if k >= 3:
                count += combinations(k, 3)
    return count

def _max_independent_set(conflict_mask, n, deadline_start, timeout=10.0):
    """Branch and bound for maximum independent set."""
    best = [0]
    all_mask = (1 << n) - 1
    
    # Degree ordering
    degrees = [bin(conflict_mask[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    # Precompute: for each vertex in order, which later vertices it conflicts with
    # We'll use a recursive approach with available mask
    
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(available, count, depth):
        if time.time() - deadline_start > timeout:
            return
        if count + bin(available).count('1') <= best[0]:
            return
        if available == 0:
            if count > best[0]:
                best[0] = count
            return
        
        # Pick vertex with max degree in available subgraph
        # For speed, use first in order that's available
        v = None
        for idx in order:
            if available & (1 << idx):
                v = idx
                break
        if v is None:
            return
        
        # Option 1: exclude v
        dfs(available & ~(1 << v), count, depth + 1)
        
        # Option 2: include v
        new_available = available & ~(1 << v) & ~conflict_mask[v]
        dfs(new_available, count + 1, depth + 1)
    
    dfs(all_mask, 0, 0)
    return best[0]

def _score(points):
    n = len(points)
    conflict_mask = _build_conflict_graph(points)
    return _max_independent_set(conflict_mask, n, time.time(), timeout=8.0)

def _try_configuration(pts, best_score_ref, start_time):
    """Try a configuration, update best if valid and better."""
    if not _valid_no_four(pts):
        return False
    tc = _triple_count(pts)
    s = _score(pts)
    if s < best_score_ref[0]:
        best_score_ref[0] = s
        best_score_ref[1] = list(pts)
        return True
    return False

def construct():
    start = time.time()
    rng = random.Random(42)
    
    best_score = [21, None]  # [score, points]
    
    def consider(pts):
        if time.time() - start > 112.0:
            return
        _try_configuration(pts, best_score, start)
    
    # Strategy 1: Perturbed 5x4 grid - this creates many collinear triples
    base = [(x, y) for x in range(5) for y in range(4)]
    for attempt in range(30000):
        if time.time() - start > 95.0:
            break
        pts = list(base)
        # Perturb multiple points to break 4-collinearities
        n_perturb = rng.randint(4, 10)
        for _ in range(n_perturb):
            idx = rng.randrange(20)
            dx = rng.randint(-2, 2)
            dy = rng.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
            x, y = pts[idx]
            nx, ny = x + dx, y + dy
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                pts[idx] = (nx, ny)
        consider(pts)
        if best_score[0] <= 11:
            break
    
    # Strategy 2: Local search from best found
    if best_score[1]:
        current_pts = list(best_score[1])
        current_score = best_score[0]
        current_tc = _triple_count(current_pts)
        
        temp = 3.0
        for step in range(50000):
            if time.time() - start > 110.0:
                break
            temp *= 0.9998
            
            idx = rng.randrange(20)
            x, y = current_pts[idx]
            
            # Try moving this point to a nearby location
            best_move = None
            best_tc = current_tc
            best_new_score = current_score
            
            for _ in range(12):
                nx = x + rng.randint(-3, 3)
                ny = y + rng.randint(-3, 3)
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    new_pt = (nx, ny)
                    if new_pt not in current_pts:
                        old_pt = current_pts[idx]
                        current_pts[idx] = new_pt
                        if _valid_no_four(current_pts):
                            tc = _triple_count(current_pts)
                            if tc < best_tc:
                                best_tc = tc
                                best_move = new_pt
                                # Only compute score for best TC move to save time
                                if tc < current_tc - 2:
                                    s = _score(current_pts)
                                    if s < best_new_score:
                                        best_new_score = s
                        current_pts[idx] = old_pt
            
            if best_move is not None:
                old_pt = current_pts[idx]
                current_pts[idx] = best_move
                current_tc = best_tc
                # Accept if score improves or with probability
                new_score = best_new_score
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score[0]:
                        best_score[0] = current_score
                        best_score[1] = list(current_pts)
                elif rng.random() < pow(2.718281828, -(new_score - current_score) / max(temp, 0.1)):
                    current_score = new_score
    
    # Strategy 3: Random configurations in small box
    for attempt in range(20000):
        if time.time() - start > 113.0:
            break
        pts = []
        for _ in range(20):
            pts.append((rng.randint(-6, 6), rng.randint(-6, 6)))
        consider(pts)
        if best_score[0] <= 10:
            break
    
    # Strategy 4: Structured - use lines with exactly 3 points
    # Create a set of 3-point lines that cover all 20 points
    # This is like a linear space / Steiner system approach
    # Try: 20 points arranged so that many triples are collinear
    # Use a 4x5 grid but shift rows to avoid 4-collinear
    for shift_pattern in range(20):
        if time.time() - start > 114.0:
            break
        pts = []
        for y in range(4):
            shift = (shift_pattern * (y + 1)) % 3
            for x in range(5):
                px = x + shift
                py = y
                pts.append((px, py))
        # Add 4 more points
        for i in range(4):
            pts.append((rng.randint(-2, 7), 4 + i))
        if len(pts) == 20:
            consider(pts)
    
    # Fallback: parabola (score 20, always valid)
    if best_score[1] is None:
        return [[x, x * x] for x in range(-10, 10)]
    
    return [list(p) for p in best_score[1]]

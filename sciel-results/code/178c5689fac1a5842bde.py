import time
import random
import math
from itertools import combinations

_DEADLINE = 110.0

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _valid_no_four(points):
    n = len(points)
    if n != 20:
        return False
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
        if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
            return False
    
    # Check no 4 collinear
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
            # Normalize direction so (dx, dy) and (-dx, -dy) are same
            # Actually for collinearity from i, we just need the line.
            # Slope (dx, dy) is sufficient to group points on the same line through i.
            # We just need to count how many points share the same normalized (dx, dy).
            # If count >= 3, then i + 3 others are collinear -> 4 points.
            key = (dx, dy)
            slopes[key] = slopes.get(key, 0) + 1
            if slopes[key] >= 3:
                return False
    return True

def _build_triples(points):
    n = len(points)
    triples = []
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
            key = (dx, dy)
            if key not in slopes:
                slopes[key] = []
            slopes[key].append(j)
        
        for key, pts in slopes.items():
            if len(pts) >= 2:
                # Any pair from pts forms a triple with i
                for a, b in combinations(pts, 2):
                    triple = (i, a, b)
                    # Sort to ensure canonical form if needed, but for set membership it's fine
                    triples.append(tuple(sorted(triple)))
    
    # Remove duplicates
    unique_triples = list(set(triples))
    return unique_triples

def _min_hitting_set(triples_list, n, deadline_start):
    if not triples_list:
        return 0
    
    M = len(triples_list)
    
    # Map triple index to bit
    # If M > 60, bitmask might be large but Python handles it.
    # However, iterating bits is slow if M is large.
    # But M is small (< 200).
    
    # Precompute mask for each point
    point_masks = [0] * n
    for idx, t in enumerate(triples_list):
        bit = 1 << idx
        for v in t:
            point_masks[v] |= bit
    
    best = [n]
    
    # Initial active mask
    initial_mask = (1 << M) - 1
    
    # Get list of active triple indices from mask
    # We need a way to pick a triple.
    # We can just iterate through the list and check if bit is set.
    
    def get_first_active(mask):
        # Find the index of the lowest set bit
        if mask == 0:
            return -1
        # (mask & -mask).bit_length() - 1
        return (mask & -mask).bit_length() - 1

    def dfs(excluded_count, active_mask):
        if time.time() - deadline_start > 109.0:
            return
        if active_mask == 0:
            if excluded_count < best[0]:
                best[0] = excluded_count
            return
        
        if excluded_count >= best[0]:
            return
        
        # Pick a triple
        t_idx = get_first_active(active_mask)
        if t_idx == -1:
            return
            
        t = triples_list[t_idx]
        
        # Heuristic: Order vertices in t by their "degree" in the active hypergraph
        # Degree of v = popcount(active_mask & point_masks[v])
        # Higher degree means removing v hits more active triples, which is good.
        # So we try removing high-degree vertices first.
        
        def degree(v):
            m = active_mask & point_masks[v]
            # Count bits
            count = 0
            while m:
                m &= m - 1
                count += 1
            return count
        
        vertices = sorted(t, key=degree, reverse=True)
        
        for v in vertices:
            new_mask = active_mask & ~point_masks[v]
            dfs(excluded_count + 1, new_mask)

    dfs(0, initial_mask)
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
            best_pts = list(pts)

    # Strategy 1: Perturbed 5x4 Grid
    # Base grid 0..4, 0..3
    base = [(x, y) for x in range(5) for y in range(4)]
    
    # We want to break 4-collinearities.
    # In a 5x4 grid, rows have 5 points (4 collinear violations? No, 5 points on a line means any 4 are collinear. 
    # Wait, "no four collinear" means no subset of 4 is collinear.
    # So a line with 5 points is INVALID.
    # A line with 4 points is INVALID.
    # So we must ensure max collinear count is 3.
    
    # The 5x4 grid has rows of 5. We MUST perturb them.
    
    # Let's try a different base.
    # Maybe a "Parabola" y = x^2 mod something?
    # Or just random start + local search.
    
    # Let's use a random start first to ensure we have a valid config quickly.
    # Then improve.
    
    # Random Start
    for _ in range(1000):
        if time.time() - start > 10.0:
            break
        pts = []
        for _ in range(20):
            x = rng.randint(-10, 10)
            y = rng.randint(-10, 10)
            pts.append((x, y))
        if _valid_no_four(pts):
            consider(pts)
            break

    if best_pts is None:
        # Fallback: Parabola
        pts = [(x, x*x) for x in range(-10, 10)]
        # Check validity
        if _valid_no_four(pts):
            best_pts = pts
            best_score = _score(pts)
        else:
            # Force valid
            return [[x, x * x] for x in range(-10, 10)]

    # Local Search / Simulated Annealing
    current_pts = list(best_pts)
    current_score = best_score
    T = 1.0
    T_min = 0.01
    
    # Precompute valid neighbors? No, just generate on the fly.
    
    steps = 0
    while time.time() - start < 105.0:
        steps += 1
        idx = rng.randrange(20)
        x, y = current_pts[idx]
        
        # Generate a neighbor
        # Move to a random point in a small box around current
        # Or swap with another point?
        # Let's try moving one point.
        
        # Try a few random moves
        found_move = False
        for _ in range(5):
            nx = x + rng.randint(-3, 3)
            ny = y + rng.randint(-3, 3)
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                new_pt = (nx, ny)
                if new_pt in current_pts:
                    continue
                
                old_pt = current_pts[idx]
                current_pts[idx] = new_pt
                
                if _valid_no_four(current_pts):
                    new_score = _score(current_pts)
                    delta = new_score - current_score
                    if delta < 0:
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_pts = list(current_pts)
                        found_move = True
                        break
                    else:
                        # Accept with probability
                        if T > T_min:
                            p = math.exp(-delta / T)
                            if rng.random() < p:
                                current_score = new_score
                                found_move = True
                                break
                            else:
                                current_pts[idx] = old_pt
                        else:
                            current_pts[idx] = old_pt
                else:
                    current_pts[idx] = old_pt
        
        # Cooling
        T *= 0.999
        if T < T_min:
            T = T_min

    return [list(p) for p in best_pts]

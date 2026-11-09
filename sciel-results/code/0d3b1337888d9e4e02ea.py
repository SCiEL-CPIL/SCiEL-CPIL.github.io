"""Combined program: search for 20 integer points with no four collinear and a small no-three-collinear subset."""
import time
import random
from math import gcd

_DEADLINE_SECONDS = 130.0  # safety margin before the 150s kill
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE_SECONDS - _elapsed()


def _collinear(a, b, c):
    # integer cross product test
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0


def _has_four_collinear(points):
    # returns True if any 4 of the points are collinear
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        # For each point, count slopes to others; if any slope has 3+ points, we have 4 collinear
        slope_counts = {}
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dxn = dx // g
            dyn = dy // g
            # Canonical sign
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn, dyn = -dxn, -dyn
            key = (dxn, dyn)
            slope_counts[key] = slope_counts.get(key, 0) + 1
        for cnt in slope_counts.values():
            if cnt >= 3:
                return True
    return False


def _max_no_three_collinear(points):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    # Precompute collinearity for triples
    # triple_bad[i][j] is True if i, j, and some k form a collinear triple
    # Actually, we need to know if adding point i to set S creates a bad triple.
    # A bad triple is (a, b, i) where a, b in S and collinear(a, b, i).
    # So for each pair (a, b), we can precompute the set of points collinear with them.
    
    # Let's use a bitmask approach for speed since n=20.
    # Precompute for each pair (i, j), a bitmask of points k such that i, j, k are collinear.
    # Actually, simpler: for each point i, and each pair (a, b) of other points, 
    # it's too expensive to precompute all pairs.
    
    # Alternative: Precompute for each point i, a list of points j such that they are "conflict" partners?
    # No, the conflict depends on the current set.
    
    # Let's stick to the recursive B&B but optimize the check.
    # Precompute a 3D array or a dictionary?
    # collinear_triplets[(i, j, k)] = True? No, we just need to check on the fly.
    
    # Optimized check:
    # Maintain a list of chosen points.
    # When adding point i, check against all pairs in chosen.
    # To speed this up, we can maintain for each point p in chosen, a bitmask of points that are collinear with p and some other point in chosen?
    # This is complex.
    
    # Let's try a different B&B strategy:
    # We want to find the largest independent set in a 3-uniform hypergraph (where edges are collinear triples).
    # This is NP-hard.
    
    # Heuristic B&B:
    # Order points by some heuristic (e.g., degree in the hypergraph - number of collinear triples they are part of).
    # Points with high degree are more likely to be excluded.
    
    # Let's precompute the degree of each point (number of other points collinear with it in any pair? No, number of pairs (a,b) such that (a,b,i) is collinear).
    # Actually, degree of i = number of pairs (a,b) such that a,b,i are collinear.
    # This is O(n^3) to compute.
    
    deg = [0] * n
    for i in range(n):
        for a in range(n):
            if a == i: continue
            for b in range(a + 1, n):
                if b == i: continue
                if _collinear(points[i], points[a], points[b]):
                    deg[i] += 1
    
    # Sort indices by degree descending
    order = sorted(range(n), key=lambda x: -deg[x])
    
    best = [0]
    
    # We will recurse on the sorted order.
    # State: pos in order, current_set (list of indices), current_size
    # To check if we can add order[pos], we need to check if it forms a triple with any two points in current_set.
    
    # To make the check fast, we can precompute for each pair (a, b) in the original indices, 
    # but we are iterating in sorted order.
    
    # Let's just use the simple check but with the sorted order.
    # The sorted order might help prune earlier if high-degree points are considered first and excluded?
    # Or if we include low-degree points first?
    # Usually, including points that are "less conflicting" first is better for finding large independent sets.
    # So sort by degree ascending?
    
    order_asc = sorted(range(n), key=lambda x: deg[x])
    
    chosen = [False] * n
    chosen_list = []
    
    def can_add(i):
        # Check if i forms a collinear triple with any pair in chosen_list
        for idx_a in range(len(chosen_list)):
            a = chosen_list[idx_a]
            for idx_b in range(idx_a + 1, len(chosen_list)):
                b = chosen_list[idx_b]
                if _collinear(points[i], points[a], points[b]):
                    return False
        return True

    def branch(pos, size):
        if size > best[0]:
            best[0] = size
        if pos >= n:
            return
        # Upper bound
        if size + (n - pos) <= best[0]:
            return
        
        i = order_asc[pos]
        
        # Try to include i
        if can_add(i):
            chosen[i] = True
            chosen_list.append(i)
            branch(pos + 1, size + 1)
            chosen_list.pop()
            chosen[i] = False
        
        # Try to exclude i
        branch(pos + 1, size)

    branch(0, 0)
    return best[0]


def _valid(points):
    # distinct, in range, no four collinear
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


def _construct_random(seed):
    rng = random.Random(seed)
    # generate random points and check
    for _ in range(200):
        pts = []
        used = set()
        while len(pts) < 20:
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        if not _has_four_collinear(pts):
            return pts
    return None


def construct():
    global _START
    _START = time.time()
    
    best_points = None
    best_score = 999
    
    # 1. Try deterministic patterns
    # Parabola: score 20
    pts_parab = [[x, x * x] for x in range(-10, 10)]
    if _valid(pts_parab):
        s = _max_no_three_collinear(pts_parab)
        if s < best_score:
            best_score = s
            best_points = pts_parab
    
    # 2. Random search with local improvement
    # Start with a few random valid points
    candidates = []
    for seed in range(20):
        if _remaining() < 10.0:
            break
        pts = _construct_random(seed)
        if pts is not None:
            candidates.append(pts)
    
    # Evaluate candidates
    for pts in candidates:
        if _remaining() < 5.0:
            break
        s = _max_no_three_collinear(pts)
        if s < best_score:
            best_score = s
            best_points = pts
    
    # If we have a good candidate, try local search to improve it
    if best_points is not None and best_score < 20:
        current_pts = best_points
        current_score = best_score
        
        # Local search loop
        while _remaining() > 5.0:
            improved = False
            rng = random.Random(int(_elapsed() * 1000) % 10000)
            
            # Try moving a random point
            for _ in range(100):
                if _remaining() < 5.0:
                    break
                idx = rng.randint(0, 19)
                old_x, old_y = current_pts[idx]
                
                # Try a random new position
                # Bias towards nearby positions to maintain structure?
                # Or fully random?
                # Let's try random in [-100, 100]
                new_x = rng.randint(-100, 100)
                new_y = rng.randint(-100, 100)
                
                if (new_x, new_y) == (old_x, old_y):
                    continue
                
                # Check if new point is already in the set
                if any(p[0] == new_x and p[1] == new_y for p in current_pts):
                    continue
                
                new_pts = current_pts.copy()
                new_pts[idx] = [new_x, new_y]
                
                if not _valid(new_pts):
                    continue
                
                # Calculate score
                # Optimization: if the change is small, maybe we can update the score incrementally?
                # For now, recompute.
                s = _max_no_three_collinear(new_pts)
                
                if s < current_score:
                    current_score = s
                    current_pts = new_pts
                    improved = True
                    if s < best_score:
                        best_score = s
                        best_points = current_pts
                    # If we found a very good score, we can break early?
                    # No, keep searching.
                    break # Restart the inner loop with the new point
                
            if not improved:
                # If no improvement in 100 moves, try a different strategy?
                # Or just break?
                # Let's try a few more rounds of random restarts if stuck.
                pass
    
    return best_points

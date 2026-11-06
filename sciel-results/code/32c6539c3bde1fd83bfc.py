"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

N = 64
TIME_LIMIT = 280.0

# EVOLVE-BLOCK-START

def _build_conflict_graph():
    """
    Build the conflict graph where an edge exists between two points if they
    share a common isosceles partner. Two points a, b conflict if there exists
    a point c such that dist(a,c) == dist(b,c) (c is equidistant) OR
    dist(a,c) == dist(a,b) (a is the apex of an isosceles with base b,c) OR
    dist(b,c) == dist(a,b) (b is the apex).
    
    Actually, the correct conflict condition: three points form an isosceles triangle
    if any two sides are equal. So points a, b, c conflict (cannot all be in set)
    if:
    - dist(a,b) == dist(a,c)  (a is apex)
    - dist(a,b) == dist(b,c)  (b is apex)
    - dist(a,c) == dist(b,c)  (c is apex)
    
    For the independent set formulation: we want a set of points with NO three forming
    an isosceles triangle. This is NOT simply an independent set in a pairwise conflict
    graph because the constraint is on TRIPLES, not pairs.
    
    However, we can still use a greedy + local search approach with better heuristics.
    Let me instead focus on a smarter greedy with conflict-based ordering.
    """
    pass


def _can_add_point_fast(point_set, dist_to_point, new_pt):
    """
    Fast check if adding new_pt creates an isosceles triangle.
    """
    n = len(point_set)
    if n < 2:
        return True
    
    nx, ny = new_pt
    
    # Check 1: No two existing points at same distance from new_pt
    seen_dists = set()
    for (px, py) in point_set:
        dx = nx - px
        dy = ny - py
        d2 = dx*dx + dy*dy
        if d2 in seen_dists:
            return False
        seen_dists.add(d2)
    
    # Check 2: For each existing point i, dist(new_pt, i) must not equal dist(i, j) for any j != i
    for (px, py) in point_set:
        dx = nx - px
        dy = ny - py
        d2 = dx*dx + dy*dy
        if d2 in dist_to_point.get((px, py), set()):
            return False
    
    return True


def _greedy_construct_fast(seed, order_type='random', deadline=None):
    """
    Fast greedy construction using distance tracking.
    """
    rng = random.Random(seed)
    points = []
    point_set = set()
    dist_to_point = defaultdict(set)
    
    if order_type == 'random':
        candidates = [(x, y) for x in range(N) for y in range(N)]
        rng.shuffle(candidates)
    elif order_type == 'spiral':
        cx, cy = N // 2, N // 2
        candidates = []
        seen = set()
        for r in range(N):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        x, y = cx + dx, cy + dy
                        if 0 <= x < N and 0 <= y < N and (x, y) not in seen:
                            seen.add((x, y))
                            candidates.append((x, y))
    elif order_type == 'grid':
        candidates = [(x, y) for x in range(N) for y in range(N)]
    else:
        candidates = [(x, y) for x in range(N) for y in range(N)]
        rng.shuffle(candidates)
    
    for pt in candidates:
        if deadline is not None and time.time() > deadline:
            break
        if _can_add_point_fast(point_set, dist_to_point, pt):
            point_set.add(pt)
            points.append(pt)
            for (px, py) in point_set:
                if pt == (px, py):
                    continue
                dx = pt[0] - px
                dy = pt[1] - py
                d2 = dx*dx + dy*dy
                dist_to_point[pt].add(d2)
                dist_to_point[(px, py)].add(d2)
    
    return points


def _local_search_fast(points, seed, max_iters=20, deadline=None):
    """
    Local search: try removing one point and adding another to maintain or improve size.
    """
    rng = random.Random(seed)
    current = list(points)
    current_set = set(current)
    
    dist_to_point = defaultdict(set)
    for i in range(len(current)):
        for j in range(i+1, len(current)):
            dx = current[i][0] - current[j][0]
            dy = current[i][1] - current[j][1]
            d2 = dx*dx + dy*dy
            dist_to_point[current[i]].add(d2)
            dist_to_point[current[j]].add(d2)
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if len(current) < 2:
            break
        
        idx = rng.randrange(len(current))
        removed = current.pop(idx)
        current_set.remove(removed)
        
        for (px, py) in current_set:
            dx = removed[0] - px
            dy = removed[1] - py
            d2 = dx*dx + dy*dy
            if d2 in dist_to_point.get((px, py), set()):
                dist_to_point[(px, py)].discard(d2)
        if removed in dist_to_point:
            del dist_to_point[removed]
        
        added_any = False
        test_candidates = [(x, y) for x in range(N) for y in range(N)]
        rng.shuffle(test_candidates)
        test_candidates = test_candidates[:500]
        
        for pt in test_candidates:
            if pt in current_set:
                continue
            if _can_add_point_fast(current_set, dist_to_point, pt):
                current.append(pt)
                current_set.add(pt)
                for (px, py) in current_set:
                    if pt == (px, py):
                        continue
                    dx = pt[0] - px
                    dy = pt[1] - py
                    d2 = dx*dx + dy*dy
                    dist_to_point[pt].add(d2)
                    dist_to_point[(px, py)].add(d2)
                added_any = True
                break
        
        if not added_any:
            current.append(removed)
            current_set.add(removed)
            for (px, py) in current_set:
                if removed == (px, py):
                    continue
                dx = removed[0] - px
                dy = removed[1] - py
                d2 = dx*dx + dy*dy
                dist_to_point[removed].add(d2)
                dist_to_point[(px, py)].add(d2)
    
    return current


def _compute_conflict_score(pt, point_set, dist_to_point):
    """
    Compute a conflict score for a point: how many existing points would
    conflict with it if added. Lower is better.
    """
    nx, ny = pt
    conflict_count = 0
    seen_dists = set()
    
    for (px, py) in point_set:
        dx = nx - px
        dy = ny - py
        d2 = dx*dx + dy*dy
        if d2 in seen_dists:
            conflict_count += 1
        seen_dists.add(d2)
        
        if d2 in dist_to_point.get((px, py), set()):
            conflict_count += 1
    
    return conflict_count


def _greedy_with_conflict_ordering(seed, deadline=None):
    """
    Greedy construction that dynamically selects the next point based on
    minimum conflict score (most compatible point first).
    """
    rng = random.Random(seed)
    points = []
    point_set = set()
    dist_to_point = defaultdict(set)
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    remaining = set(all_points)
    
    while remaining:
        if deadline is not None and time.time() > deadline:
            break
        
        if len(point_set) < 2:
            # Pick randomly when set is small
            pt = rng.choice(list(remaining))
        else:
            # Score all remaining points and pick the one with minimum conflict
            best_pt = None
            best_score = float('inf')
            # For efficiency, sample a subset if remaining is large
            if len(remaining) > 200:
                sample = rng.sample(list(remaining), 200)
            else:
                sample = list(remaining)
            
            for pt in sample:
                score = _compute_conflict_score(pt, point_set, dist_to_point)
                if score < best_score:
                    best_score = score
                    best_pt = pt
            
            if best_pt is None:
                break
            pt = best_pt
        
        if _can_add_point_fast(point_set, dist_to_point, pt):
            point_set.add(pt)
            points.append(pt)
            remaining.discard(pt)
            
            for (px, py) in point_set:
                if pt == (px, py):
                    continue
                dx = pt[0] - px
                dy = pt[1] - py
                d2 = dx*dx + dy*dy
                dist_to_point[pt].add(d2)
                dist_to_point[(px, py)].add(d2)
    
    return points


def _swap_local_search(points, seed, max_iters=50, deadline=None):
    """
    More aggressive local search: try swapping pairs of points.
    """
    rng = random.Random(seed)
    current = list(points)
    current_set = set(current)
    
    dist_to_point = defaultdict(set)
    for i in range(len(current)):
        for j in range(i+1, len(current)):
            dx = current[i][0] - current[j][0]
            dy = current[i][1] - current[j][1]
            d2 = dx*dx + dy*dy
            dist_to_point[current[i]].add(d2)
            dist_to_point[current[j]].add(d2)
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if len(current) < 2:
            break
        
        # Try removing a random point and adding a better one
        idx = rng.randrange(len(current))
        removed = current.pop(idx)
        current_set.remove(removed)
        
        for (px, py) in current_set:
            dx = removed[0] - px
            dy = removed[1] - py
            d2 = dx*dx + dy*dy
            if d2 in dist_to_point.get((px, py), set()):
                dist_to_point[(px, py)].discard(d2)
        if removed in dist_to_point:
            del dist_to_point[removed]
        
        # Find best replacement
        all_candidates = [(x, y) for x in range(N) for y in range(N) if (x, y) not in current_set]
        if not all_candidates:
            current.append(removed)
            current_set.add(removed)
            for (px, py) in current_set:
                if removed == (px, py):
                    continue
                dx = removed[0] - px
                dy = removed[1] - py
                d2 = dx*dx + dy*dy
                dist_to_point[removed].add(d2)
                dist_to_point[(px, py)].add(d2)
            continue
        
        # Score candidates
        best_pt = None
        best_score = float('inf')
        sample_size = min(300, len(all_candidates))
        sample = rng.sample(all_candidates, sample_size)
        
        for pt in sample:
            if _can_add_point_fast(current_set, dist_to_point, pt):
                # Check if this point has lower conflict score than the removed one
                # (proxy: just take the first valid one for speed)
                best_pt = pt
                break
        
        if best_pt is not None:
            current.append(best_pt)
            current_set.add(best_pt)
            for (px, py) in current_set:
                if best_pt == (px, py):
                    continue
                dx = best_pt[0] - px
                dy = best_pt[1] - py
                d2 = dx*dx + dy*dy
                dist_to_point[best_pt].add(d2)
                dist_to_point[(px, py)].add(d2)
        else:
            current.append(removed)
            current_set.add(removed)
            for (px, py) in current_set:
                if removed == (px, py):
                    continue
                dx = removed[0] - px
                dy = removed[1] - py
                d2 = dx*dx + dy*dy
                dist_to_point[removed].add(d2)
                dist_to_point[(px, py)].add(d2)
    
    return current


def construct():
    start_time = time.time()
    best_points = []
    best_size = 0
    deadline = start_time + TIME_LIMIT - 20
    
    # Phase 1: Quick greedy constructions with different strategies
    for order_type in ['random', 'spiral', 'grid']:
        if time.time() > deadline:
            break
        pts = _greedy_construct_fast(42, order_type, deadline)
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = list(pts)
    
    # Phase 2: Greedy with conflict-based ordering
    for seed in range(5):
        if time.time() > deadline:
            break
        pts = _greedy_with_conflict_ordering(seed, deadline)
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = list(pts)
    
    # Phase 3: Local search improvements
    seed = 0
    while time.time() < deadline:
        # Start from best known or a fresh greedy
        if seed % 3 == 0:
            pts = list(best_points)
        else:
            pts = _greedy_construct_fast(seed, 'random', deadline)
        
        if time.time() < deadline:
            pts = _local_search_fast(pts, seed, max_iters=15, deadline=deadline)
        
        if time.time() < deadline and len(pts) >= best_size:
            pts = _swap_local_search(pts, seed, max_iters=20, deadline=deadline)
        
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = list(pts)
        
        seed += 1
        if seed % 5 == 0:
            pts = _greedy_with_conflict_ordering(seed, deadline)
            if len(pts) > best_size:
                best_size = len(pts)
                best_points = list(pts)
    
    return best_points

# EVOLVE-BLOCK-END

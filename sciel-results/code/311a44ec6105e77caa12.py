"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

N = 64
TIME_LIMIT = 280.0

# EVOLVE-BLOCK-START

def _can_add_point_fast(point_set, dist_to_point, new_pt):
    """
    Fast check if adding new_pt creates an isosceles triangle.
    point_set: set of (x,y) tuples
    dist_to_point: dict mapping (x,y) -> set of squared distances from that point to others
    new_pt: (x, y) tuple
    Returns True if safe to add, False otherwise.
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
    # This means: for each existing point p, the distance from new_pt to p must not be in dist_to_point[p]
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
    dist_to_point = defaultdict(set)  # (x,y) -> set of squared distances to other points
    
    # Generate candidate positions
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
            # Update distance sets
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
    
    # Build dist_to_point for current
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
        
        # Remove distances involving removed point
        for (px, py) in current_set:
            dx = removed[0] - px
            dy = removed[1] - py
            d2 = dx*dx + dy*dy
            if d2 in dist_to_point.get((px, py), set()):
                dist_to_point[(px, py)].discard(d2)
        if removed in dist_to_point:
            del dist_to_point[removed]
        
        # Try to find a replacement point
        added_any = False
        # Try random subset of candidates first for speed
        test_candidates = [(x, y) for x in range(N) for y in range(N)]
        rng.shuffle(test_candidates)
        test_candidates = test_candidates[:500]  # Limit for speed
        
        for pt in test_candidates:
            if pt in current_set:
                continue
            if _can_add_point_fast(current_set, dist_to_point, pt):
                current.append(pt)
                current_set.add(pt)
                # Add distances
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
            # Put back the removed point
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
    deadline = start_time + TIME_LIMIT - 20  # Safety margin
    
    # Quick greedy constructions with different orders
    for order_type in ['random', 'spiral', 'grid']:
        if time.time() > deadline:
            break
        pts = _greedy_construct_fast(42, order_type, deadline)
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = list(pts)
    
    # Multiple random restarts with local search
    seed = 0
    while time.time() < deadline:
        pts = _greedy_construct_fast(seed, 'random', deadline)
        if time.time() < deadline:
            pts = _local_search_fast(pts, seed, max_iters=10, deadline=deadline)
        
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = list(pts)
        
        seed += 1
        # If we've done enough restarts, try different orderings
        if seed % 10 == 0:
            for order_type in ['spiral', 'grid']:
                if time.time() > deadline:
                    break
                pts = _greedy_construct_fast(seed, order_type, deadline)
                if len(pts) > best_size:
                    best_size = len(pts)
                    best_points = list(pts)
    
    return best_points

# EVOLVE-BLOCK-END

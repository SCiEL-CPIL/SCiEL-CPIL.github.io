"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 64
TIME_LIMIT = 280.0

# EVOLVE-BLOCK-START

def _is_isosceles_fast(points, new_pt):
    """
    Check if adding new_pt to points creates any isosceles triangle.
    points is a list of (x, y) tuples.
    Returns True if an isosceles triangle is formed, False otherwise.
    """
    n = len(points)
    if n < 2:
        return False
    
    # Compute squared distances from new_pt to all existing points
    # and check for duplicates among those distances (isosceles with apex at new_pt)
    dists_to_new = []
    for i in range(n):
        dx = new_pt[0] - points[i][0]
        dy = new_pt[1] - points[i][1]
        dists_to_new.append(dx*dx + dy*dy)
    
    # If any two distances from new_pt to existing points are equal,
    # then we have an isosceles triangle with apex at new_pt
    if len(set(dists_to_new)) != len(dists_to_new):
        return True
    
    # Now check for isosceles triangles where new_pt is one of the base vertices.
    # For each pair of existing points (i, j), check if dist(new_pt, i) == dist(i, j)
    # or dist(new_pt, j) == dist(i, j)
    # But we already know dist(new_pt, i) != dist(new_pt, j) for all i != j
    # So we need: dist(new_pt, i) == dist(i, j) or dist(new_pt, j) == dist(i, j)
    
    # Precompute all pairwise distances among existing points
    # This is O(n^2) which might be slow, but let's try a smarter approach
    
    # For each existing point i, compute dist(i, j) for all j > i
    # and check if any equals dist(new_pt, i) or dist(new_pt, j)
    
    # Actually, let's just do the O(n^2) check since n is small in greedy phase
    for i in range(n):
        d_new_i = dists_to_new[i]
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d_ij = dx*dx + dy*dy
            d_new_j = dists_to_new[j]
            if d_new_i == d_ij or d_new_j == d_ij:
                return True
    
    return False


def _greedy_construct(seed, order_type='random'):
    """
    Greedy construction: try to add points in a given order.
    order_type: 'random', 'spiral', 'grid'
    """
    rng = random.Random(seed)
    points = []
    
    # Generate candidate positions
    if order_type == 'random':
        candidates = [(x, y) for x in range(N) for y in range(N)]
        rng.shuffle(candidates)
    elif order_type == 'spiral':
        # Spiral from center
        cx, cy = N // 2, N // 2
        candidates = []
        for r in range(N):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        x, y = cx + dx, cy + dy
                        if 0 <= x < N and 0 <= y < N and (x, y) not in candidates:
                            candidates.append((x, y))
    elif order_type == 'grid':
        candidates = [(x, y) for x in range(N) for y in range(N)]
    else:
        candidates = [(x, y) for x in range(N) for y in range(N)]
        rng.shuffle(candidates)
    
    for pt in candidates:
        if not _is_isosceles_fast(points, pt):
            points.append(pt)
    
    return points


def _local_search(points, seed, max_iters=100):
    """
    Try to improve the set by removing and re-adding points.
    """
    rng = random.Random(seed)
    current = list(points)
    current_set = set(current)
    
    for _ in range(max_iters):
        # Try to remove a random point and see if we can add two new ones
        if len(current) < 2:
            break
        
        idx = rng.randrange(len(current))
        removed = current.pop(idx)
        current_set.remove(removed)
        
        # Try to add back better points
        added_any = False
        # Try all positions
        for x in range(N):
            for y in range(N):
                pt = (x, y)
                if pt in current_set:
                    continue
                if not _is_isosceles_fast(current, pt):
                    current.append(pt)
                    current_set.add(pt)
                    added_any = True
                    break
            if added_any:
                break
        
        if not added_any:
            # Put back the removed point
            current.append(removed)
            current_set.add(removed)
    
    return current


def construct():
    start_time = time.time()
    best_points = []
    best_size = 0
    
    # First, do a quick greedy construction to get a baseline
    for order_type in ['random', 'spiral', 'grid']:
        pts = _greedy_construct(42, order_type)
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = list(pts)
    
    # Now do multiple random restarts with local search
    for seed in range(1000):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 30:
            break
        
        # Greedy construction with random order
        pts = _greedy_construct(seed, 'random')
        
        # Local search
        pts = _local_search(pts, seed, max_iters=50)
        
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = list(pts)
        
        # Early termination if we've found a very good solution
        if best_size >= 20:
            break
    
    return best_points

# EVOLVE-BLOCK-END

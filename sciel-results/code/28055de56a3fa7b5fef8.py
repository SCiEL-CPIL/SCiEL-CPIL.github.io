"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

# Precompute the six neighbor directions
NEIGHBORS = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]

def _dist_sq(p1, p2):
    """Squared distance in triangular lattice."""
    da = p1[0] - p2[0]
    db = p1[1] - p2[1]
    return da*da + da*db + db*db

def _count_edges_fast(point_set):
    """Fast edge count: for each point, check 6 neighbors, count each edge once."""
    count = 0
    for p in point_set:
        for da, db in NEIGHBORS:
            if (p[0]+da, p[1]+db) in point_set:
                count += 1
    return count // 2

def _is_valid_placement(new_pt, point_set, exclude_pt=None):
    """Check if new_pt is at distance >= 1 from all points in point_set (except exclude_pt)."""
    for p in point_set:
        if p == exclude_pt:
            continue
        if _dist_sq(new_pt, p) < 1:
            return False
    return True

def _generate_hex_cluster(n):
    """Generate a dense hexagonal cluster of n points."""
    # Start with center and expand ring by ring
    points = [(0, 0)]
    if n == 1:
        return points
    
    # Ring r has 6*r points
    r = 1
    while len(points) < n:
        # Generate points on ring r in a spiral-like order
        ring_points = []
        # Start at (r, 0) and go clockwise
        for i in range(6*r):
            # Parametrize the hexagon
            if i < r:
                a, b = r, i
            elif i < 2*r:
                a, b = r - (i - r), i
            elif i < 3*r:
                a, b = 0, r + (i - 2*r)
            elif i < 4*r:
                a, b = -(i - 3*r), r
            elif i < 5*r:
                a, b = -r, r - (i - 4*r)
            else:
                a, b = -r + (i - 5*r), 0
            ring_points.append((a, b))
        
        # Add points from this ring until we have n total
        for pt in ring_points:
            if len(points) >= n:
                break
            if pt not in points:
                points.append(pt)
        
        r += 1
    
    return points[:n]

def _greedy_construct(n=32):
    """Greedy construction: add points one at a time, choosing the one that maximizes edges."""
    points = []
    point_set = set()
    
    # Generate candidate points in order of increasing distance from origin
    candidates = []
    for a in range(-20, 21):
        for b in range(-20, 21):
            dist_sq = a*a + a*b + b*b
            candidates.append((dist_sq, a, b))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Start with center
    points.append((0, 0))
    point_set.add((0, 0))
    
    while len(points) < n:
        best_pt = None
        best_edges = -1
        for _, a, b in candidates:
            pt = (a, b)
            if pt in point_set:
                continue
            # Check validity
            if not _is_valid_placement(pt, point_set):
                continue
            # Count edges this point would form
            edges = 0
            for p in point_set:
                if _dist_sq(pt, p) == 1:
                    edges += 1
            if edges > best_edges:
                best_edges = edges
                best_pt = pt
            if best_edges == len(points):  # Can't do better than connecting to all
                break
        
        if best_pt is None:
            break
        
        points.append(best_pt)
        point_set.add(best_pt)
    
    return points

def _local_search(points, deadline, max_iter_per_point=50):
    """Local search: try swapping each point with a nearby unoccupied point."""
    best_points = list(points)
    best_set = set(best_points)
    best_count = _count_edges_fast(best_set)
    
    # Generate candidate replacement points
    candidate_replacements = []
    for a in range(-15, 16):
        for b in range(-15, 16):
            if (a, b) not in best_set:
                candidate_replacements.append((a, b))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_points)):
            if time.time() > deadline:
                break
            old_pt = best_points[i]
            # Try random subset of candidates for speed
            random.shuffle(candidate_replacements)
            tried = 0
            for new_pt in candidate_replacements:
                if tried >= max_iter_per_point:
                    break
                tried += 1
                if new_pt == old_pt or new_pt in best_set:
                    continue
                if not _is_valid_placement(new_pt, best_set, exclude_pt=old_pt):
                    continue
                
                # Compute new edge count efficiently
                # Remove edges from old_pt, add edges to new_pt
                old_edges = sum(1 for p in best_set if p != old_pt and _dist_sq(old_pt, p) == 1)
                new_edges = sum(1 for p in best_set if p != old_pt and _dist_sq(new_pt, p) == 1)
                delta = new_edges - old_edges
                if delta > 0:
                    best_points[i] = new_pt
                    best_set.remove(old_pt)
                    best_set.add(new_pt)
                    best_count += delta
                    improved = True
                    break
    
    return best_points, best_count

def _simulated_annealing(points, deadline, start_temp=1.0, end_temp=0.1, cooling=0.999):
    """Simulated annealing to escape local optima."""
    current_points = list(points)
    current_set = set(current_points)
    current_count = _count_edges_fast(current_set)
    best_points = list(current_points)
    best_set = set(current_points)
    best_count = current_count
    
    temp = start_temp
    candidates = []
    for a in range(-15, 16):
        for b in range(-15, 16):
            if (a, b) not in current_set:
                candidates.append((a, b))
    
    while time.time() < deadline and temp > end_temp:
        # Pick a random point to move
        i = random.randint(0, len(current_points) - 1)
        old_pt = current_points[i]
        
        # Pick a random candidate
        if not candidates:
            break
        new_pt = random.choice(candidates)
        
        if new_pt == old_pt or new_pt in current_set:
            continue
        
        if not _is_valid_placement(new_pt, current_set, exclude_pt=old_pt):
            continue
        
        # Compute edge count change
        old_edges = sum(1 for p in current_set if p != old_pt and _dist_sq(old_pt, p) == 1)
        new_edges = sum(1 for p in current_set if p != old_pt and _dist_sq(new_pt, p) == 1)
        delta = new_edges - old_edges
        
        if delta > 0 or random.random() < math.exp(delta / temp):
            # Accept the move
            current_points[i] = new_pt
            current_set.remove(old_pt)
            current_set.add(new_pt)
            current_count += delta
            
            # Update candidates
            candidates.remove(new_pt)
            candidates.append(old_pt)
            
            if current_count > best_count:
                best_count = current_count
                best_points = list(current_points)
                best_set = set(current_points)
        
        temp *= cooling
    
    return best_points, best_count

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Generate initial configurations using different methods
    configs = []
    
    # 1. Hexagonal cluster
    hex_cluster = _generate_hex_cluster(32)
    configs.append(hex_cluster)
    
    # 2. Greedy construction
    greedy = _greedy_construct(32)
    if len(greedy) == 32:
        configs.append(greedy)
    
    # 3. Spiral order (original method)
    max_coord = 20
    all_points = []
    for a in range(-max_coord, max_coord+1):
        for b in range(-max_coord, max_coord+1):
            dist_sq = a*a + a*b + b*b
            all_points.append((dist_sq, a, b))
    all_points.sort(key=lambda x: (x[0], math.atan2(math.sqrt(3)/2 * x[2], x[1] + x[2]/2)))
    spiral = [(a, b) for _, a, b in all_points[:32]]
    configs.append(spiral)
    
    # Local search on each config
    best_overall = None
    best_count_overall = -1
    
    for config in configs:
        if len(config) != 32:
            continue
        pts, cnt = _local_search(config, deadline)
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_overall = pts
    
    # Simulated annealing on the best config
    if best_overall is not None and time.time() < deadline:
        pts, cnt = _simulated_annealing(best_overall, deadline)
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_overall = pts
    
    # Final local search polish
    if best_overall is not None and time.time() < deadline:
        pts, cnt = _local_search(best_overall, deadline, max_iter_per_point=100)
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_overall = pts
    
    return [[a, b] for a, b in best_overall]
# EVOLVE-BLOCK-END

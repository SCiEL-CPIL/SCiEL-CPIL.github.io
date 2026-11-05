"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 64
TIME_LIMIT = 320  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _is_isosceles(p, q, r):
    """Check if three points form an isosceles triangle (including degenerate)."""
    dx1 = p[0] - q[0]
    dy1 = p[1] - q[1]
    d1_sq = dx1 * dx1 + dy1 * dy1
    
    dx2 = q[0] - r[0]
    dy2 = q[1] - r[1]
    d2_sq = dx2 * dx2 + dy2 * dy2
    
    dx3 = p[0] - r[0]
    dy3 = p[1] - r[1]
    d3_sq = dx3 * dx3 + dy3 * dy3
    
    if d1_sq == d2_sq or d2_sq == d3_sq or d1_sq == d3_sq:
        return True
    return False

def _check_point_against_set(candidate, points):
    """Check if adding candidate to points creates any isosceles triangle."""
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if _is_isosceles(candidate, points[i], points[j]):
                return False
    return True

def _fast_greedy_dist_set(start_seed, deadline):
    """
    Fast greedy using the distance-set trick.
    Maintains a set of all squared distances between existing points.
    When adding a new point P, check:
    1. All distances from P to existing points are distinct.
    2. No distance from P to an existing point is already in the global dist_set.
    If both hold, P is valid.
    """
    random.seed(start_seed)
    all_points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_points)
    
    points = []
    dist_set = set()  # set of all squared distances between pairs in points
    points_array = np.empty((0, 2), dtype=np.int32)
    
    for p in all_points:
        if time.time() > deadline:
            break
        
        px, py = p
        if len(points) == 0:
            points.append(p)
            points_array = np.array([[px, py]], dtype=np.int32)
            continue
        
        # Vectorized distance computation
        dx = px - points_array[:, 0]
        dy = py - points_array[:, 1]
        d2_arr = dx * dx + dy * dy
        
        # Condition 2: No distance from P to existing points is in dist_set
        d2_list = d2_arr.tolist()
        valid = True
        for d2 in d2_list:
            if d2 in dist_set:
                valid = False
                break
        
        if not valid:
            continue
        
        # Condition 1: All distances from P to existing points must be distinct
        if len(set(d2_list)) != len(d2_list):
            continue
        
        # Add p to points
        points.append(p)
        points_array = np.vstack([points_array, [[px, py]]])
        
        # Update dist_set with new distances
        for d2 in d2_list:
            dist_set.add(d2)
    
    return points

def _greedy_random(start_seed, deadline):
    """Greedy algorithm with random ordering."""
    random.seed(start_seed)
    points = []
    all_points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_points)
    
    for pt in all_points:
        if time.time() > deadline:
            break
        if _check_point_against_set(pt, points):
            points.append(pt)
    return points

def _greedy_with_priority(priority_func, deadline):
    """Greedy with a priority function to guide selection."""
    points = []
    center = N // 2
    points = []
    
    if _check_point_against_set((center, center), points):
        points.append((center, center))
    
    candidates = []
    for x in range(N):
        for y in range(N):
            dist = (x - center) ** 2 + (y - center) ** 2
            candidates.append((dist, x, y))
    candidates.sort()
    
    for dist, x, y in candidates:
        if time.time() > deadline:
            break
        pt = (x, y)
        if pt not in points:
            if _check_point_against_set(pt, points):
                points.append(pt)
    return points

def _bitwise_conflict_greedy(start_seed, deadline):
    """
    Bitset-based greedy on the conflict graph.
    We precompute a bitset for each point indicating which other points
    form an isosceles triangle with it as the vertex.
    
    Actually, the conflict is: two points A and B are in conflict if there
    exists a third point C such that ABC is isosceles. But that's not right.
    
    The condition is: no three points form an isosceles triangle.
    This is equivalent to: for any three points A, B, C, the distances AB, BC, CA
    must all be distinct.
    
    A simpler view: We build a set S. When adding a new point P, we need to ensure
    that for any two existing points Q, R, PQR is not isosceles.
    This is what _check_point_against_set does.
    
    The "conflict graph" idea from the prompt is slightly different. Let's stick to
    the efficient distance-set method but use bitwise operations for the "distinctness"
    and "membership" checks if possible, or just optimize the existing loop.
    
    However, the prompt suggests: "Formulate the problem as a maximum independent set on a conflict graph...
    and solve it using a fast, vectorized bitset-based greedy algorithm".
    
    Let's try a different structural approach:
    Instead of checking against all pairs (O(M^2) per point), we can maintain
    a dictionary mapping each point to the set of distances it has to other points.
    But the global dist_set is O(1) lookup. The bottleneck is the O(M) loop to check
    if any of the M distances is in the global set, plus the O(M) check for duplicates.
    
    We can vectorize the "in set" check? Not easily with Python sets.
    But we can use a boolean array for the distance set if the max distance is small enough.
    Max squared distance in 64x64 grid: (63-0)^2 + (63-0)^2 = 2 * 3969 = 7938.
    So we can use a boolean array of size ~8000.
    """
    random.seed(start_seed)
    all_points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_points)
    
    points = []
    max_dist_sq = 2 * (N - 1) ** 2
    dist_used = np.zeros(max_dist_sq + 1, dtype=bool)
    
    points_array = np.empty((0, 2), dtype=np.int32)
    
    for p in all_points:
        if time.time() > deadline:
            break
        
        px, py = p
        if len(points) == 0:
            points.append(p)
            points_array = np.array([[px, py]], dtype=np.int32)
            continue
        
        # Vectorized distance computation
        dx = px - points_array[:, 0]
        dy = py - points_array[:, 1]
        d2_arr = dx * dx + dy * dy
        
        # Check if any distance is already used
        # d2_arr is an array of ints. We can index dist_used with it.
        if np.any(dist_used[d2_arr]):
            continue
        
        # Check if any distances are duplicated within d2_arr
        # We can check this by seeing if the number of unique values is less than the length
        # Or, we can just add them to dist_used and see if any was already true? No, we already checked.
        # We need to ensure d2_arr itself has no duplicates.
        # np.unique(d2_arr).size != len(d2_arr)
        if len(np.unique(d2_arr)) != len(d2_arr):
            continue
        
        # Add p to points
        points.append(p)
        points_array = np.vstack([points_array, [[px, py]]])
        
        # Mark these distances as used
        dist_used[d2_arr] = True
    
    return points

def _hybrid_greedy(start_seed, deadline):
    """
    Combines the fast bitwise check with a smarter ordering.
    We try to order points by some heuristic, e.g., random but with a bias
    towards corners or edges? Or just random restarts.
    """
    return _bitwise_conflict_greedy(start_seed, deadline)

def construct():
    """Construct a set of points with no three forming an isosceles triangle."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = []
    
    # Try the fast bitwise greedy with multiple random restarts
    # This should be significantly faster than the Python-set version
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        
        points = _bitwise_conflict_greedy(seed, deadline)
        if len(points) > len(best_points):
            best_points = points
    
    # Also try the original slower greedy with random restarts if time permits
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        
        points = _greedy_random(seed, deadline)
        if len(points) > len(best_points):
            best_points = points
    
    # Also try the center-outward strategy
    if time.time() < deadline - 5:
        points = _greedy_with_priority(None, deadline)
        if len(points) > len(best_points):
            best_points = points
    
    return best_points
# EVOLVE-BLOCK-END

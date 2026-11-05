"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 64
TIME_LIMIT = 320  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _vectorized_check(candidate_xy, points_array, dist_set):
    """
    Check if adding candidate to points creates any isosceles triangle.
    Uses vectorized NumPy operations for speed.
    
    Args:
        candidate_xy: (x, y) tuple
        points_array: Nx2 numpy array of existing points
        dist_set: set of all squared distances between pairs in points
    
    Returns:
        (bool, list) - (is_valid, list of new squared distances)
    """
    if len(points_array) == 0:
        return True, []
    
    px, py = candidate_xy
    dx = points_array[:, 0] - px
    dy = points_array[:, 1] - py
    d2 = dx * dx + dy * dy
    
    # Check condition 1: all distances from P to existing points must be distinct
    # Use sorting to check for duplicates
    sorted_d2 = np.sort(d2)
    if np.any(sorted_d2[:-1] == sorted_d2[1:]):
        return False, []
    
    # Check condition 2: no distance from P to an existing point is already in dist_set
    for d in d2:
        if int(d) in dist_set:
            return False, []
    
    return True, [int(d) for d in d2]

def _fast_greedy_vectorized(start_seed, deadline, strategy="random"):
    """
    Fast greedy using vectorized distance checks.
    
    Args:
        start_seed: random seed
        deadline: time deadline
        strategy: "random", "center_out", or "spiral"
    
    Returns:
        list of (x, y) tuples
    """
    random.seed(start_seed)
    np.random.seed(start_seed)
    
    if strategy == "random":
        all_points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(all_points)
    elif strategy == "center_out":
        center = N // 2
        candidates = []
        for x in range(N):
            for y in range(N):
                dist = (x - center) ** 2 + (y - center) ** 2
                candidates.append((dist, x, y))
        candidates.sort()
        all_points = [(x, y) for _, x, y in candidates]
    elif strategy == "spiral":
        # Generate points in spiral order from center
        center = N // 2
        all_points = []
        for r in range(N):
            for angle in range(0, 360, 30):
                rad = math.radians(angle)
                x = int(round(center + r * math.cos(rad)))
                y = int(round(center + r * math.sin(rad)))
                if 0 <= x < N and 0 <= y < N:
                    all_points.append((x, y))
        # Remove duplicates while preserving order
        seen = set()
        unique_points = []
        for p in all_points:
            if p not in seen:
                seen.add(p)
                unique_points.append(p)
        all_points = unique_points
    else:
        all_points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(all_points)
    
    points = []
    points_array = np.array([]).reshape(0, 2)
    dist_set = set()
    
    for p in all_points:
        if time.time() > deadline:
            break
        
        valid, new_dists = _vectorized_check(p, points_array, dist_set)
        
        if valid:
            points.append(p)
            points_array = np.vstack([points_array, np.array(p)]) if len(points_array) > 0 else np.array([p])
            for d in new_dists:
                dist_set.add(d)
    
    return points

def _local_improvement(points, deadline):
    """
    Try to improve the set by swapping or adding points.
    This is a simple local search heuristic.
    """
    if len(points) == 0:
        return points
    
    points_array = np.array(points)
    dist_set = set()
    
    # Build dist_set
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx * dx + dy * dy
            dist_set.add(d2)
    
    # Try adding points that weren't in the original greedy pass
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_set = set(points)
    
    # Shuffle for random order
    random.shuffle(all_points)
    
    improved = False
    for p in all_points:
        if time.time() > deadline:
            break
        
        if p in point_set:
            continue
        
        valid, new_dists = _vectorized_check(p, points_array, dist_set)
        
        if valid:
            points.append(p)
            points_array = np.vstack([points_array, np.array(p)]) if len(points_array) > 0 else np.array([p])
            for d in new_dists:
                dist_set.add(d)
            point_set.add(p)
            improved = True
    
    return points

def construct():
    """Construct a set of points with no three forming an isosceles triangle."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = []
    
    # Try multiple strategies with vectorized greedy
    strategies = ["random", "center_out", "spiral"]
    
    for strategy in strategies:
        for seed in range(30):
            if time.time() > deadline - 10:  # Leave 10s safety margin
                break
            
            points = _fast_greedy_vectorized(seed, deadline, strategy)
            if len(points) > len(best_points):
                best_points = points
    
    # Local improvement on the best solution
    if time.time() < deadline - 5 and len(best_points) > 0:
        improved = _local_improvement(best_points, deadline)
        if len(improved) > len(best_points):
            best_points = improved
    
    return best_points
# EVOLVE-BLOCK-END

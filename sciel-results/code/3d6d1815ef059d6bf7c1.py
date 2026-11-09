"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import isqrt

N = 64
DEADLINE = 320.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    def check_isosceles(points):
        """Check if any three points form an isosceles triangle."""
        n = len(points)
        if n < 3:
            return False
        # For each point, compute squared distances to all other points
        # If any point has two equal distances to two other points, it's isosceles
        for i in range(n):
            dists = []
            for j in range(n):
                if i == j:
                    continue
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dists.append(dx*dx + dy*dy)
            # Check for duplicates
            if len(dists) != len(set(dists)):
                return True
        return False
    
    def is_valid_addition(points, new_point):
        """Check if adding new_point to points creates an isosceles triangle."""
        n = len(points)
        # Compute squared distance from new_point to each existing point
        new_dists = {}
        for i, p in enumerate(points):
            dx = new_point[0] - p[0]
            dy = new_point[1] - p[1]
            d2 = dx*dx + dy*dy
            if d2 in new_dists:
                # Two existing points at same distance from new_point -> isosceles with new_point as apex
                return False
            new_dists[d2] = i
        
        # For each pair of existing points, check if new_point forms isosceles with them
        # i.e., dist(new, p_i) == dist(p_i, p_j) or dist(new, p_j) == dist(p_i, p_j)
        # or dist(p_i, p_j) == dist(new, p_i) which is same as above
        # Actually we need: for any i, j in points, the triangle (new, p_i, p_j) is not isosceles
        # This means: dist(new,p_i) != dist(new,p_j) [already checked above]
        #             dist(new,p_i) != dist(p_i,p_j)
        #             dist(new,p_j) != dist(p_i,p_j)
        
        # Precompute pairwise distances among existing points
        # For efficiency, let's check each pair
        for i in range(n):
            di = new_dists[i]  # dist(new, points[i])
            for j in range(i+1, n):
                dj = new_dists[j]  # dist(new, points[j])
                # dist between points[i] and points[j]
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dij = dx*dx + dy*dy
                if di == dij or dj == dij:
                    return False
        return True
    
    def greedy_construct(random_order):
        """Greedy construction using given random order of candidate points."""
        selected = []
        for p in random_order:
            if is_valid_addition(selected, p):
                selected.append(p)
        return selected
    
    def local_search(points, max_iters=500):
        """Local search with swap operations."""
        points = list(points)
        n = len(points)
        # Generate all candidate points not in current set
        point_set = set(points)
        candidates = [(x, y) for x in range(N) for y in range(N) if (x, y) not in point_set]
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > DEADLINE - 30:
                break
            # Try swapping one point out and one in
            random.shuffle(candidates)
            for i in range(n):
                if time.time() > DEADLINE - 30:
                    break
                removed = points[i]
                remaining = [points[j] for j in range(n) if j != i]
                # Try to add a candidate
                for cand in candidates:
                    if is_valid_addition(remaining, cand):
                        new_points = remaining + [cand]
                        # We gained one point (removed one, added one, but we want to grow)
                        # Actually swap keeps same size. We want to grow.
                        # Let's try adding without removing first
                        break
                else:
                    continue
                # If we can add a point without removing, do so
                if is_valid_addition(points, cand):
                    points.append(cand)
                    n += 1
                    point_set.add(cand)
                    candidates.remove(cand)
                    improved = True
                    break
            # If no growth, try swap to enable future growth
            if not improved:
                for i in range(n):
                    if time.time() > DEADLINE - 30:
                        break
                    removed = points[i]
                    remaining = [points[j] for j in range(n) if j != i]
                    for cand in candidates:
                        if is_valid_addition(remaining, cand):
                            new_points = remaining + [cand]
                            # Check if this swap enables more growth later
                            # For now, just accept if it doesn't reduce size
                            points = new_points
                            point_set.discard(removed)
                            point_set.add(cand)
                            candidates.remove(cand)
                            improved = True
                            break
                    if improved:
                        break
        return points
    
    best_size = 0
    best_points = []
    
    # Strategy 1: Greedy with different random orders
    for seed in range(200):
        if time.time() > DEADLINE - 60:
            break
        random.seed(seed)
        all_points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(all_points)
        points = greedy_construct(all_points)
        if len(points) > best_size:
            best_size = len(points)
            best_points = list(points)
    
    # Strategy 2: Local search on best greedy result
    if time.time() < DEADLINE - 60 and best_points:
        improved = local_search(best_points, max_iters=200)
        if len(improved) > best_size:
            best_size = len(improved)
            best_points = improved
    
    # Strategy 3: More greedy restarts if time permits
    for seed in range(200, 500):
        if time.time() > DEADLINE - 30:
            break
        random.seed(seed)
        all_points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(all_points)
        points = greedy_construct(all_points)
        if len(points) > best_size:
            best_size = len(points)
            best_points = list(points)
    
    # Final local search if time permits
    if time.time() < DEADLINE - 15 and best_points:
        improved = local_search(best_points, max_iters=100)
        if len(improved) > best_size:
            best_size = len(improved)
            best_points = improved
    
    return best_points
# EVOLVE-BLOCK-END

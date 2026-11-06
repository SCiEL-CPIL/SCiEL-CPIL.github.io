import random
import math
import time

N = 64

def construct():
    """Construct a set of grid points in [0, N)^2 with no three forming an isosceles triangle."""
    start_time = time.time()
    deadline = start_time + 320  # Safety margin before 330s kill
    
    best_set = []
    best_size = 0
    
    def is_valid_addition(point, current_set, dist_cache):
        """Check if adding 'point' to current_set maintains no-isosceles property."""
        px, py = point
        M = len(current_set)
        
        # Check: for each existing point, compute distance to new point
        # and verify no isosceles triangle is formed
        for i in range(M):
            ax, ay = current_set[i]
            d_pa_sq = (px - ax)**2 + (py - ay)**2
            
            for j in range(i + 1, M):
                bx, by = current_set[j]
                d_pb_sq = (px - bx)**2 + (py - by)**2
                d_ab_sq = (ax - bx)**2 + (ay - by)**2
                
                # Isosceles if any two of these three distances are equal
                if d_pa_sq == d_pb_sq or d_pa_sq == d_ab_sq or d_pb_sq == d_ab_sq:
                    return False
        
        return True
    
    def greedy_build(seed):
        """Build a set greedily using random ordering."""
        rng = random.Random(seed)
        points = [(x, y) for x in range(N) for y in range(N)]
        rng.shuffle(points)
        
        current = []
        for p in points:
            if is_valid_addition(p, current, None):
                current.append(p)
        
        return current
    
    def local_search(initial_set, seed):
        """Improve the set via local search."""
        rng = random.Random(seed)
        current = list(initial_set)
        current_set = set(current)
        
        def try_add(point):
            nonlocal current, current_set
            if point not in current_set and is_valid_addition(point, current, None):
                current.append(point)
                current_set.add(point)
                return True
            return False
        
        def try_remove(point):
            nonlocal current, current_set
            if point in current_set:
                current.remove(point)
                current_set.discard(point)
                return True
            return False
        
        # Try to add random points
        all_points = [(x, y) for x in range(N) for y in range(N)]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            rng.shuffle(all_points)
            for p in all_points:
                if time.time() >= deadline - 5:
                    break
                if try_add(p):
                    improved = True
        
        return current
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Greedy with different seeds
    for seed in range(10):
        if time.time() >= deadline - 10:
            break
        s = greedy_build(seed)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Strategy 2: Local search from greedy results
    for seed in range(20):
        if time.time() >= deadline - 10:
            break
        s = greedy_build(seed)
        improved = local_search(s, seed)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Strategy 3: Start from diagonal or structured initial sets
    # Points on a line with carefully chosen x-coordinates
    for seed in range(5):
        if time.time() >= deadline - 10:
            break
        rng = random.Random(seed + 1000)
        # Random subset of a line
        line_points = [(x, 0) for x in range(N)]
        rng.shuffle(line_points)
        s = []
        for p in line_points:
            if is_valid_addition(p, s, None):
                s.append(p)
        improved = local_search(s, seed + 1000)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Strategy 4: Random initial sets with local search
    for seed in range(30):
        if time.time() >= deadline - 10:
            break
        rng = random.Random(seed + 2000)
        # Start with a small random set
        s = []
        all_points = [(x, y) for x in range(N) for y in range(N)]
        for _ in range(5):
            p = rng.choice(all_points)
            if p not in s and is_valid_addition(p, s, None):
                s.append(p)
        improved = local_search(s, seed + 2000)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Final: return best found
    return best_set

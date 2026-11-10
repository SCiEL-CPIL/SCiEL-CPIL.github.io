"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 320  # Safety margin before 330s kill
    random.seed(42)
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def sq_dist(a, b):
        return (a[0] - b[0]) ** 2 + (a[0] - b[0]) ** 2 if False else (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    
    def is_isosceles(a, b, c):
        """Check if three points form an isosceles triangle (including degenerate)."""
        d1 = sq_dist(a, b)
        d2 = sq_dist(b, c)
        d3 = sq_dist(a, c)
        # Isosceles if any two distances are equal
        # Note: this includes equilateral and degenerate (collinear with midpoint)
        if d1 == d2 or d2 == d3 or d1 == d3:
            # Also check they're not all the same point
            if a != b and b != c and a != c:
                return True
            # If two points are the same, it's degenerate - still "isosceles" in a sense
            # But typically we consider distinct points
            if d1 == 0 or d2 == 0 or d3 == 0:
                return True
        return False
    
    def can_add(point_set, new_point):
        """Check if new_point can be added without creating an isosceles triangle."""
        n = len(point_set)
        for i in range(n):
            for j in range(i + 1, n):
                if is_isosceles(point_set[i], point_set[j], new_point):
                    return False
        return True
    
    def greedy_add(points, candidates, random_order=True):
        """Greedily add points from candidates to points."""
        if random_order:
            random.shuffle(candidates)
        for p in candidates:
            if can_add(points, p):
                points.append(p)
        return points
    
    def local_search(points, time_limit=1.0):
        """Try to improve the set by removing one point and adding another."""
        start_time = time.time()
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            n = len(points)
            # Try removing each point and see if we can add a better one
            for i in range(n):
                if time.time() > deadline:
                    return points
                # Try removing point i
                remaining = points[:i] + points[i+1:]
                # Find a point not in remaining that can be added
                in_set = set(remaining)
                candidates = [p for p in all_points if p not in in_set]
                random.shuffle(candidates)
                best_add = None
                for p in candidates:
                    if can_add(remaining, p):
                        best_add = p
                        break
                if best_add is not None:
                    # Same size, but different configuration - might enable future additions
                    # For now, let's just try to find a configuration that allows more points
                    pass
            
            # Try adding points from a random order
            in_set = set(points)
            candidates = [p for p in all_points if p not in in_set]
            random.shuffle(candidates)
            added = False
            for p in candidates:
                if can_add(points, p):
                    points.append(p)
                    added = True
                    improved = True
                    break
        return points
    
    best_set = []
    
    # Multiple restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        current = []
        
        # Random initialization
        init_size = random.randint(5, 15)
        candidates = all_points[:]
        random.shuffle(candidates)
        for p in candidates:
            if len(current) >= init_size:
                break
            if can_add(current, p):
                current.append(p)
        
        # Greedy add
        current = greedy_add(current, all_points[:], random_order=True)
        
        # Local search
        current = local_search(current, time_limit=min(2.0, max(0.1, deadline - time.time() - 10)))
        
        if len(current) > len(best_set):
            best_set = current[:]
        
        # If we've used significant time, reduce restarts
        elapsed = time.time() - (deadline - 320)
        if elapsed > 200:
            break
    
    # Final greedy pass on best_set
    in_set = set(best_set)
    candidates = [p for p in all_points if p not in in_set]
    random.shuffle(candidates)
    for p in candidates:
        if can_add(best_set, p):
            best_set.append(p)
    
    return best_set
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 20 distinct integer points with |x|,|y| <= 100, no four collinear,
    minimizing the largest subset with no three collinear.
    
    Strategy: Use a known construction based on the parabola but perturb it to
    create more collinear triples, thereby reducing the maximum cap-free set.
    
    A simpler approach: use a grid-like structure with careful placement.
    
    Actually, the known best for n=20 is around 12-14. Let's use a heuristic
    approach: start with random points, check constraints, and use local search
    to minimize the score.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    def count_collinear_triples(points):
        """Count number of collinear triples in the point set."""
        triples = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    # Check if three points are collinear
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        triples += 1
        return triples
    
    def max_cap_free_subset(points):
        """
        Find the largest subset of points with no three collinear.
        Uses branch and bound for exact computation.
        """
        n = len(points)
        if n <= 2:
            return n
        
        # Precompute which triples are collinear
        collinear = {}
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        collinear[(i, j, k)] = True
        
        # Find all collinear triples
        collinear_triples = [t for t in collinear if collinear[t]]
        
        # Branch and bound
        best = 0
        
        def is_valid_subset(subset):
            """Check if subset has no three collinear."""
            subset_set = set(subset)
            for (i, j, k) in collinear_triples:
                if i in subset_set and j in subset_set and k in subset_set:
                    return False
            return True
        
        def branch_and_bound(remaining, current, depth):
            nonlocal best
            if time.time() > deadline - 5:
                return
            if not remaining:
                if len(current) > best:
                    best = len(current)
                return
            
            # Upper bound: current + remaining
            if len(current) + len(remaining) <= best:
                return
            
            # Try adding each remaining point
            for idx, point in enumerate(remaining):
                new_current = current + [point]
                if is_valid_subset(new_current):
                    branch_and_bound(remaining[idx+1:], new_current, depth+1)
            
            # Try not adding the first remaining point
            branch_and_bound(remaining[1:], current, depth+1)
        
        branch_and_bound(list(range(n)), [], 0)
        return best
    
    def has_four_collinear(points):
        """Check if any four points are collinear."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    # Check slope
                    dx1 = x2 - x1
                    dy1 = y2 - y1
                    dx2 = x3 - x1
                    dy2 = y3 - y1
                    # The line through i and j
                    for l in range(k+1, n):
                        x4, y4 = points[l]
                        dx3 = x4 - x1
                        dy3 = y4 - y1
                        # Check if l is on the line through i,j
                        if dx1 * dy3 == dy1 * dx3:
                            return True
        return False
    
    # Start with a known good configuration
    # Use points on a circle or parabola with perturbations
    best_points = None
    best_score = 21  # Worst case
    
    # Try several random configurations
    for attempt in range(1000):
        if time.time() > deadline - 30:
            break
        
        # Generate random points
        points = []
        used = set()
        for _ in range(20):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in used:
                    used.add((x, y))
                    points.append([x, y])
                    break
        
        # Check no four collinear
        if has_four_collinear(points):
            continue
        
        # Compute score
        score = max_cap_free_subset(points)
        
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
            print(f"Found better: score={score}, attempt={attempt}")
        
        if best_score <= 10:
            break
    
    if best_points is None:
        # Fallback: parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_points
# EVOLVE-BLOCK-END

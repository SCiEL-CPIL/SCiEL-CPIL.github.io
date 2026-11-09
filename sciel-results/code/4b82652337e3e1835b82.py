"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64
TIME_LIMIT = 290  # Safety margin below 330s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_pts = []
    best_len = 0
    
    def elapsed():
        return time.time() - start_time
    
    def dist2(a, b):
        return (a[0] - b[0])**2 + (a[1] - b[1])**2
    
    def is_valid_addition(p, pts):
        """Check if adding p to pts creates any isosceles triangle."""
        k = len(pts)
        if k == 0:
            return True
        if k == 1:
            return True  # Two points can't form a triangle
        if k >= 2:
            for i in range(k):
                d1 = dist2(p, pts[i])
                for j in range(i + 1, k):
                    d2 = dist2(p, pts[j])
                    d3 = dist2(pts[i], pts[j])
                    # Isosceles if any two distances are equal
                    if d1 == d2 or d1 == d3 or d2 == d3:
                        return False
        return True
    
    def greedy_construct(seed=None):
        """Randomized greedy construction with local improvement."""
        if seed is not None:
            random.seed(seed)
        
        # Start with a few random points
        pts = []
        all_points = [(x, y) for x in range(N) for y in range(N)]
        
        # Try to add points greedily
        # Use a randomized order for candidates
        candidates = all_points[:]
        random.shuffle(candidates)
        
        for p in candidates:
            if elapsed() > TIME_LIMIT - 5:
                break
            if is_valid_addition(p, pts):
                pts.append(p)
        
        # Local improvement: try to add more points by reordering
        # Do a few passes of local search
        for _ in range(3):
            if elapsed() > TIME_LIMIT - 5:
                break
            improved = True
            while improved:
                improved = False
                remaining = [p for p in all_points if p not in pts]
                random.shuffle(remaining)
                for p in remaining:
                    if elapsed() > TIME_LIMIT - 5:
                        break
                    if is_valid_addition(p, pts):
                        pts.append(p)
                        improved = True
        
        return pts
    
    # Try multiple restarts with different seeds
    num_restarts = 50
    for i in range(num_restarts):
        if elapsed() > TIME_LIMIT - 10:
            break
        pts = greedy_construct(seed=i)
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts[:]
    
    return best_pts
# EVOLVE-BLOCK-END

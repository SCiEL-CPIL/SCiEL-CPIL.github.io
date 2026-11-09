import numpy as np
import itertools
from collections import Counter
import random
import math
import time

def construct():
    # Pool of candidate points
    pool = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    pool_set = set(pool)
    
    # Initial configuration: Random or Grid? 
    # A grid often has high multiplicity for small areas.
    # Let's start with a 5x6 grid centered at 0, scaled by 2 to spread out?
    # Or just random init and let SA do the work. 
    # Given the speed of the new update, we can afford more iterations.
    
    # Let's use a grid init to have a good start.
    # 30 points. 5x6 grid.
    # x in [-4, 0], y in [-3, 2]? 
    # Let's try to place them in [-30, 30].
    # x = -12, -8, -4, 0, 4, 8 (6 cols)
    # y = -10, -5, 0, 5, 10 (5 rows) -> 30 points.
    init_pts = []
    for x in [-12, -8, -4, 0, 4, 8]:
        for y in [-10, -5, 0, 5, 10]:
            init_pts.append((x, y))
            
    current_pts = init_pts[:]
    current_set = set(current_pts)
    
    def compute_initial_counts(pts):
        counts = Counter()
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i+1, n):
                x2, y2 = pts[j]
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    det = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                    if det > 0:
                        counts[det] += 1
        return counts

    area_counts = compute_initial_counts(current_pts)
    
    def get_best(counts):
        if not counts:
            return 0, 0
        return max(counts.items(), key=lambda x: x[1])
    
    best_area, best_count = get_best(area_counts)
    
    global_best_count = best_count
    global_best_pts = current_pts[:]
    global_best_area = best_area
    
    # Simulated Annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    max_iters = 5000 # Can be higher now
    
    # Precompute pairs for faster update? 
    # The update requires iterating over all pairs of the *other* 29 points.
    # We can just loop.
    
    for it in range(max_iters):
        if T < T_min:
            break
            
        # Pick random index to move
        idx = random.randint(0, 29)
        old_pt = current_pts[idx]
        
        # Pick random new point from pool, not in current_set
        # To avoid generating list of candidates every time (expensive), 
        # we can pick random x,y and check if in set. 
        # Pool size 61x61 = 3721. Current size 30.
        # Probability of collision is low.
        new_pt = None
        for _ in range(10): # try a few times
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            p = (x, y)
            if p not in current_set:
                new_pt = p
                break
        if new_pt is None:
            continue
            
        # Perform swap
        # 1. Remove old_pt contributions
        # 2. Add new_pt contributions
        
        # We need to iterate over all pairs of points in current_pts excluding idx
        # This is O(29^2) ~ 800 ops.
        
        # To speed up, we can store the indices of the other points
        # But building the list of other points is O(30).
        
        other_pts = [current_pts[i] for i in range(30) if i != idx]
        
        # Decrement old areas
        for i in range(29):
            x1, y1 = old_pt
            x2, y2 = other_pts[i]
            for j in range(i+1, 29):
                x3, y3 = other_pts[j]
                det = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                if det > 0:
                    area_counts[det] -= 1
                    if area_counts[det] == 0:
                        del area_counts[det]
                        
        # Increment new areas
        for i in range(29):
            x1, y1 = new_pt
            x2, y2 = other_pts[i]
            for j in range(i+1, 29):
                x3, y3 = other_pts[j]
                det = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                if det > 0:
                    area_counts[det] += 1
                    
        # Check new best
        new_best_area, new_best_count = get_best(area_counts)
        
        # Acceptance
        delta = new_best_count - best_count
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept
            current_pts[idx] = new_pt
            current_set.remove(old_pt)
            current_set.add(new_pt)
            best_area, best_count = new_best_area, new_best_count
            
            if best_count > global_best_count:
                global_best_count = best_count
                global_best_pts = current_pts[:]
                global_best_area = best_area
        else:
            # Reject: Revert changes
            # Decrement new areas
            for i in range(29):
                x1, y1 = new_pt
                x2, y2 = other_pts[i]
                for j in range(i+1, 29):
                    x3, y3 = other_pts[j]
                    det = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                    if det > 0:
                        area_counts[det] -= 1
                        if area_counts[det] == 0:
                            del area_counts[det]
                        
            # Increment old areas
            for i in range(29):
                x1, y1 = old_pt
                x2, y2 = other_pts[i]
                for j in range(i+1, 29):
                    x3, y3 = other_pts[j]
                    det = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                    if det > 0:
                        area_counts[det] += 1
                        
        T *= cooling
        
    return global_best_pts

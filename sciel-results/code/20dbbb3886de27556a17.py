import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Generate all points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_set = []
    best_size = 0
    
    # We will try several random permutations
    # A greedy algorithm with random restarts
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        random.shuffle(all_points)
        current_set = []
        # dist_sets[i] is a set of squared distances from current_set[i] to other points in current_set
        dist_sets = []
        
        for p in all_points:
            # Check if p can be added
            valid = True
            if not current_set:
                valid = True
            else:
                # 1. Distances from p to all existing points must be distinct
                dists_from_p = []
                for q in current_set:
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    dists_from_p.append(dx*dx + dy*dy)
                
                if len(set(dists_from_p)) != len(dists_from_p):
                    valid = False
            
            if valid:
                # 2. For each existing point q, distance from q to p must not be in dist_sets[q]
                for i, q in enumerate(current_set):
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    d_sq = dx*dx + dy*dy
                    if d_sq in dist_sets[i]:
                        valid = False
                        break
            
            if valid:
                # Add p to current_set
                current_set.append(p)
                new_dist_set = set()
                for q in current_set[:-1]:
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    new_dist_set.add(dx*dx + dy*dy)
                dist_sets.append(new_dist_set)
                
                # Update dist_sets for existing points
                for i, q in enumerate(current_set[:-1]):
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    dist_sets[i].add(dx*dx + dy*dy)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # If we found a very large set, we might stop early, but let's try a few more
        if best_size > 50: # Heuristic upper bound guess
             # Could break, but let's let the time guard handle it or add more attempts if fast
             pass

    return best_set
# EVOLVE-BLOCK-END

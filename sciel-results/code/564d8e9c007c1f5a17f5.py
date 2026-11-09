import random
import time

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 320
    
    all_points = [(x, y) for y in range(N) for x in range(N)]
    
    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    best_set = []
    best_size = 0
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        random.seed(restarts)
        shuffled = all_points[:]
        random.shuffle(shuffled)
        
        current_set = []
        # dist_sets[i] is the set of squared distances from current_set[i] to other points in current_set
        dist_sets = []
        
        for p in shuffled:
            # 1. Check distances from p to all in current_set
            new_dists = []
            for q in current_set:
                d = dist_sq(p, q)
                new_dists.append(d)
                
            # Check for duplicates in new_dists
            if len(new_dists) != len(set(new_dists)):
                continue
                
            # 2. Check if any new_dists[i] is already in dist_sets[i]
            valid = True
            for i, d in enumerate(new_dists):
                if d in dist_sets[i]:
                    valid = False
                    break
            
            if valid:
                # Add p
                current_set.append(p)
                dist_sets.append(set(new_dists))
                # Update existing dist_sets
                for i in range(len(current_set) - 1):
                    dist_sets[i].add(new_dists[i])
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return best_set

import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 300  # seconds
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_set = []
    best_size = 0
    
    # Try multiple random restarts
    for seed in range(1000):
        if time.time() - start_time > deadline - 5:
            break
            
        random.seed(seed)
        candidates = all_points[:]
        random.shuffle(candidates)
        
        selected = []
        # dist_from_selected[i] is a set of squared distances from selected[i] to other selected points
        dist_from_selected = []
        
        for p in candidates:
            # Check if adding p is valid
            valid = True
            new_dists = []
            
            for i, q in enumerate(selected):
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                d = dx*dx + dy*dy
                new_dists.append(d)
                
                # Check if d is already in dist_from_selected[i]
                # This covers the case where q is the apex
                if d in dist_from_selected[i]:
                    valid = False
                    break
            
            if valid:
                # Check if all new_dists are distinct
                # This covers the case where p is the apex
                if len(set(new_dists)) != len(new_dists):
                    valid = False
            
            if valid:
                selected.append(p)
                dist_from_selected.append(set())
                
                # Update dist_from_selected for all existing points
                for i, q in enumerate(selected[:-1]):
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    d = dx*dx + dy*dy
                    dist_from_selected[i].add(d)
                    dist_from_selected[-1].add(d)
                
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = selected[:]
                    
        # Optional: Local search / improvement?
        # For now, just keep the best greedy result.
        
    return best_set
# EVOLVE-BLOCK-END

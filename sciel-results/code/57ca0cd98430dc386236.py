import time
import random

N = 64
TIME_LIMIT = 280  # Safety margin before 300s kill

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    all_points = [(x, y) for y in range(N) for x in range(N)]
    
    # Try multiple random restarts with different orderings
    for seed in range(10000):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        random.seed(seed)
        
        # Shuffle points for greedy selection
        candidates = all_points[:]
        random.shuffle(candidates)
        
        selected = []
        # For each selected point, track set of squared distances to other selected points
        dist_sets = []  # dist_sets[i] = set of squared distances from selected[i] to other selected points
        # Also track: for each pair of selected points, we don't need to store separately
        # The key check: when adding point Q:
        #   1. For each selected point P_i: d(P_i, Q)^2 not in dist_sets[i]
        #   2. All d(Q, P_i)^2 for i are distinct (i.e., len of set == len(selected))
        
        for Q in candidates:
            qx, qy = Q
            valid = True
            
            # Check condition 1: for each existing point, distance to Q not already used
            new_dists = []
            for i, P in enumerate(selected):
                px, py = P
                d2 = (qx - px) ** 2 + (qy - py) ** 2
                if d2 in dist_sets[i]:
                    valid = False
                    break
                new_dists.append(d2)
            
            if not valid:
                continue
            
            # Check condition 2: all new_dists are distinct
            if len(set(new_dists)) != len(new_dists):
                continue
            
            # Add Q to selected
            selected.append(Q)
            # Update dist_sets: for each existing point, add d(P_i, Q)^2
            for i in range(len(selected) - 1):
                dist_sets[i].add(new_dists[i])
            # New point Q has dist set = new_dists
            dist_sets.append(set(new_dists))
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
        
        # Early termination if we can't beat best
        if best_size >= 40:  # Heuristic threshold
            break
    
    return best_set

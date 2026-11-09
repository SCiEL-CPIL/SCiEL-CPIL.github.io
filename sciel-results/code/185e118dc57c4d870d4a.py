import time
import random
import numpy as np

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # Precompute all points as a numpy array for fast distance calculations
    # Shape: (M, 2)
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    best_size = 0
    best_S_list = []
    best_D = {}
    
    def try_add(S, D, p_idx):
        """
        Attempts to add point p_idx to set S.
        Returns True if successful, updating S and D.
        """
        if p_idx in S:
            return False
            
        p = points[p_idx]
        
        if len(S) == 0:
            S.add(p_idx)
            D[p_idx] = set()
            return True
            
        S_list = list(S)
        S_pts = points[S_list]
        
        # Vectorized distance calculation
        # diff has shape (|S|, 2)
        diff = p - S_pts
        dists = np.sum(diff * diff, axis=1)
        
        # Check for duplicate distances from p to any two points in S
        # This handles the case where p is the apex of an isosceles triangle
        if len(set(dists)) != len(dists):
            return False
            
        # Check if any distance is already present in the distance set of the target point
        # This handles the case where the target point is the apex of an isosceles triangle
        # involving p and another point already in S
        for i, s_idx in enumerate(S_list):
            d_val = int(dists[i])
            if d_val in D[s_idx]:
                return False
                
        # Validity confirmed. Update data structures.
        S.add(p_idx)
        D[p_idx] = set(dists.tolist()) # Store as list of ints
        
        # Update distance sets for all existing points
        for i, s_idx in enumerate(S_list):
            D[s_idx].add(int(dists[i]))
            
        return True

    def greedy_pass(order, time_limit):
        """
        Runs a greedy pass with the given order of points.
        """
        S = set()
        D = {}
        
        for p_idx in order:
            if time.time() > time_limit:
                break
            try_add(S, D, p_idx)
                
        return S, D

    # Phase 1: Random Greedy Restarts
    # The goal is to find a good initial solution quickly.
    t_phase1_end = start_time + 60
    
    while time.time() < t_phase1_end:
        order = list(range(M))
        random.shuffle(order)
        
        S, D = greedy_pass(order, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S_list = list(S)
            best_D = {k: set(v) for k, v in D.items()}

    # Phase 2: Local Search with Swap Operations
    # Use swap-based moves to escape local optima.
    S = set(best_S_list)
    D = {k: set(v) for k, v in best_D.items()}
    
    no_improve_count = 0
    max_no_improve = 30
    last_improve_time = time.time()
    
    def try_swap_remove_add(S, D, remove_idx, add_idx):
        """
        Try to swap: remove remove_idx from S, then add add_idx.
        Returns True if successful.
        """
        if remove_idx not in S or add_idx in S:
            return False
        
        # Remove remove_idx from S and update D
        S.discard(remove_idx)
        D.pop(remove_idx, None)
        
        # Update D for all remaining points
        for s in S:
            dx = points[remove_idx][0] - points[s][0]
            dy = points[remove_idx][1] - points[s][1]
            d_sq = int(dx*dx + dy*dy)
            if d_sq in D[s]:
                D[s].discard(d_sq)
        
        # Try to add add_idx
        if try_add(S, D, add_idx):
            return True
        else:
            # Rollback: re-add remove_idx
            S.add(remove_idx)
            # Recompute distances from remove_idx to all in S
            S_list = list(S)
            S_pts = points[S_list]
            diff = points[remove_idx] - S_pts
            dists = np.sum(diff * diff, axis=1)
            D[remove_idx] = set(dists.tolist())
            for i, s_idx in enumerate(S_list):
                D[s_idx].add(int(dists[i]))
            return False
    
    while time.time() < deadline - 5:
        improved = False
        
        # First, try simple additions
        candidates = [i for i in range(M) if i not in S]
        random.shuffle(candidates)
        
        for p_idx in candidates:
            if try_add(S, D, p_idx):
                improved = True
                break
        
        if not improved:
            # Try swap operations: remove one point, add another
            if len(S) > 3:
                s_list = list(S)
                random.shuffle(s_list)
                
                for remove_idx in s_list:
                    if time.time() > deadline - 5:
                        break
                    
                    # Find candidates that could replace remove_idx
                    # Points that are close to remove_idx's position tend to be more compatible
                    r_x, r_y = points[remove_idx]
                    
                    # Generate nearby candidates (within a window)
                    window = 8
                    x_min = max(0, r_x - window)
                    x_max = min(N, r_x + window + 1)
                    y_min = max(0, r_y - window)
                    y_max = min(N, r_y + window + 1)
                    
                    nearby = [i for i in range(M) if i not in S and x_min <= points[i][0] < x_max and y_min <= points[i][1] < y_max]
                    random.shuffle(nearby)
                    
                    # Also add some random candidates from the full grid
                    if candidates:
                        random.shuffle(candidates)
                        nearby.extend(candidates[:20])
                    
                    for add_idx in nearby:
                        if time.time() > deadline - 5:
                            break
                        if add_idx in S:
                            continue
                        if try_swap_remove_add(S, D, remove_idx, add_idx):
                            improved = True
                            break
                    
                    if improved:
                        break
        
        if improved:
            no_improve_count = 0
            last_improve_time = time.time()
            if len(S) > best_size:
                best_size = len(S)
                best_S_list = list(S)
                best_D = {k: set(v) for k, v in D.items()}
        else:
            no_improve_count += 1
            
            # If we haven't improved in a while, do a more aggressive perturbation
            if no_improve_count > max_no_improve:
                # Remove 2-3 random points to break out of local optimum
                if len(S) > 5:
                    k = random.randint(2, 3)
                    to_remove = random.sample(list(S), min(k, len(S)))
                    for r in to_remove:
                        S.discard(r)
                        D.pop(r, None)
                        for s in S:
                            dx = points[r][0] - points[s][0]
                            dy = points[r][1] - points[s][1]
                            d_sq = int(dx*dx + dy*dy)
                            if d_sq in D[s]:
                                D[s].discard(d_sq)
                else:
                    # If set is very small, full restart
                    order = list(range(M))
                    random.shuffle(order)
                    S, D = greedy_pass(order, deadline)
                
                no_improve_count = 0
                last_improve_time = time.time()

    # Return the best solution found
    # Ensure we return tuples of (x, y)
    return [tuple(points[i]) for i in best_S_list]

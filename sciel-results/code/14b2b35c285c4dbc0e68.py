import time
import random
import numpy as np

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit

    # Precompute all points as a numpy array for fast distance calculations
    # Shape: (M, 2)
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    best_size = 0
    best_S_list = []
    best_D = {}
    
    def try_add(S, D, p_idx, points_array):
        """
        Attempts to add point p_idx to set S.
        Returns True if successful, updating S and D.
        """
        if p_idx in S:
            return False
            
        p = points_array[p_idx]
        
        if len(S) == 0:
            S.add(p_idx)
            D[p_idx] = set()
            return True
            
        S_list = list(S)
        S_pts = points_array[S_list]
        
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

    def greedy_pass(order, time_limit, points_array):
        """
        Runs a greedy pass with the given order of points.
        """
        S = set()
        D = {}
        
        for p_idx in order:
            if time.time() > time_limit:
                break
            try_add(S, D, p_idx, points_array)
                
        return S, D

    # Phase 1: Random Greedy Restarts
    # The goal is to find a good initial solution quickly.
    t_phase1_end = start_time + 60
    
    while time.time() < t_phase1_end:
        order = list(range(M))
        random.shuffle(order)
        
        S, D = greedy_pass(order, deadline, points)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S_list = list(S)
            best_D = {k: set(v) for k, v in D.items()}

    # Phase 2: Local Search
    # Refine the best solution found so far.
    if not best_S_list:
        return []

    S = set(best_S_list)
    D = {k: set(v) for k, v in best_D.items()}
    
    no_improve_count = 0
    max_no_improve = 100
    last_improve_time = time.time()
    
    # Pre-shuffle candidate list to avoid re-shuffling M items every iteration
    # We will iterate through this list, skipping points already in S.
    # If we get stuck, we reshuffle.
    candidates = list(range(M))
    random.shuffle(candidates)
    candidate_idx = 0
    
    while time.time() < deadline - 5:
        improved = False
        
        # Iterate through candidates
        # We reset candidate_idx to 0 if we reshuffled, otherwise continue
        # To ensure we check all potential additions, we might need to loop through the list
        # But if S changes, the validity of other candidates changes.
        # Simple strategy: try to add from the current candidate pointer onwards.
        # If we reach the end without improvement, reshuffle and restart pointer.
        
        while candidate_idx < M:
            if time.time() > deadline - 5:
                break
                
            p_idx = candidates[candidate_idx]
            candidate_idx += 1
            
            if p_idx in S:
                continue
                
            if try_add(S, D, p_idx, points):
                improved = True
                no_improve_count = 0
                last_improve_time = time.time()
                if len(S) > best_size:
                    best_size = len(S)
                    best_S_list = list(S)
                    best_D = {k: set(v) for k, v in D.items()}
                # After adding, the state changed. We could continue scanning, 
                # but new points might enable others that were previously invalid?
                # No, adding a point only adds constraints. It cannot make an invalid point valid.
                # So we can continue scanning the rest of the list.
                # However, if we added a point, we should probably restart the scan 
                # to see if we can add more from the beginning, as the order might matter 
                # for greedy choices. But restarting is expensive.
                # Let's just continue. If we get stuck, we will perturb.
                pass

        if not improved:
            no_improve_count += 1
            
            # If we haven't improved in a while, perturb
            if no_improve_count > max_no_improve:
                # Aggressive Perturbation: Remove 2-5 points
                if len(S) > 5:
                    k = random.randint(2, 5)
                    to_remove = random.sample(list(S), min(k, len(S)))
                    for r in to_remove:
                        S.discard(r)
                        D.pop(r, None)
                        # Update D for remaining points
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
                    S, D = greedy_pass(order, deadline, points)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S_list = list(S)
                        best_D = {k: set(v) for k, v in D.items()}
                
                no_improve_count = 0
                last_improve_time = time.time()
                
                # Reshuffle candidates and reset index
                random.shuffle(candidates)
                candidate_idx = 0
            else:
                # If we just reached the end of the candidate list without improving,
                # reshuffle to try a different order
                if candidate_idx >= M:
                    random.shuffle(candidates)
                    candidate_idx = 0

    # Return the best solution found
    # Ensure we return tuples of (x, y)
    return [tuple(points[i]) for i in best_S_list]

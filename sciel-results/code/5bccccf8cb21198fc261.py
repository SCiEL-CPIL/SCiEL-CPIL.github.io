import time
import random
import math
import numpy as np

N = 64
NUM_POINTS = N * N
MAX_DIST_SQ = (N-1)**2 + (N-1)**2

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Keep 40s margin for 330s limit
    
    # Precompute coordinates as a numpy array
    # coords[i] = [x, y]
    coords = np.array([(i // N, i % N) for i in range(NUM_POINTS)], dtype=np.int32)
    
    # forbidden_mask[i, d] is True if distance d is in dist_from[i]
    # Size: 4096 * 8192 booleans = ~32MB. Acceptable.
    forbidden_mask = np.zeros((NUM_POINTS, MAX_DIST_SQ + 1), dtype=bool)
    
    S = []  # List of point indices in current set
    in_set = np.zeros(NUM_POINTS, dtype=bool)
    
    best_S = []
    best_size = 0
    
    def calc_d2_vec(p_idx, S_indices):
        """
        Calculate squared distances from point p_idx to all points in S_indices.
        Returns a numpy array of squared distances.
        """
        if len(S_indices) == 0:
            return np.array([], dtype=np.int32)
        p_coord = coords[p_idx]
        S_coords = coords[S_indices]
        # (x_p - x_s)^2 + (y_p - y_s)^2
        diff = p_coord[None, :] - S_coords
        d2s = np.sum(diff * diff, axis=1)
        return d2s.astype(np.int32)

    def is_valid_add(p, S_indices):
        """
        Check if p can be added to the set defined by S_indices.
        """
        if len(S_indices) == 0:
            return True
        
        d2s = calc_d2_vec(p, S_indices)
        
        # Condition 1: All distances from p to S must be distinct
        if len(np.unique(d2s)) != len(d2s):
            return False
        
        # Condition 2: For each s in S, d(p, s) must not be in dist_from[s]
        # forbidden_mask[s, d] is True if d is in dist_from[s]
        # We need to check if any forbidden_mask[s_i, d2s_i] is True
        # S_indices is a list/array of indices. d2s is the corresponding distances.
        if len(S_indices) > 0:
            # Create an array of row indices and column indices for advanced indexing
            # Rows: S_indices, Cols: d2s
            # Note: S_indices might be a list, convert to array for indexing
            S_arr = np.array(S_indices, dtype=np.int32)
            # Check conflicts
            conflicts = forbidden_mask[S_arr, d2s]
            if np.any(conflicts):
                return False
                
        return True

    def do_add(p, S_indices):
        """
        Add p to the set. Updates forbidden_mask.
        Assumes p is not already in S.
        """
        S_indices.append(p)
        in_set[p] = True
        
        if len(S_indices) > 1:
            # Get the previous points
            prev_indices = S_indices[:-1]
            d2s = calc_d2_vec(p, prev_indices)
            
            # Update mask for p: distances to all prev
            forbidden_mask[p, d2s] = True
            
            # Update mask for prev points: distance to p
            # prev_indices is array, d2s is array
            forbidden_mask[prev_indices, d2s] = True

    def do_remove(p, S_indices):
        """
        Remove p from the set. Updates forbidden_mask.
        """
        if p not in S_indices:
            return
        
        # Find index
        try:
            idx = S_indices.index(p)
        except ValueError:
            return
            
        S_indices.pop(idx)
        in_set[p] = False
        
        if len(S_indices) > 0:
            # Get remaining points
            rem_indices = np.array(S_indices, dtype=np.int32)
            d2s = calc_d2_vec(p, rem_indices)
            
            # Clear mask for p
            forbidden_mask[p, d2s] = False
            
            # Clear mask for remaining points
            forbidden_mask[rem_indices, d2s] = False

    def greedy_init(order):
        S_indices.clear()
        in_set[:] = False
        forbidden_mask[:] = False
        
        for p in order:
            if is_valid_add(p, S_indices):
                do_add(p, S_indices)
                
    # 1. Greedy Initialization
    # Try a few random orders
    best_init_size = 0
    best_init_S = []
    
    for _ in range(5):
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        greedy_init(order)
        if len(S_indices) > best_init_size:
            best_init_size = len(S_indices)
            best_init_S = S_indices[:]
            
    # Set current state to best init
    S_indices = best_init_S[:]
    # Rebuild state from scratch to ensure consistency (masks were cleared in greedy_init, 
    # but we copied the list. We need to rebuild the mask for the new S_indices)
    in_set[:] = False
    forbidden_mask[:] = False
    # Re-add points one by one to rebuild mask
    temp_S = []
    for p in S_indices:
        do_add(p, temp_S)
    S_indices = temp_S
    
    best_size = len(S_indices)
    best_S = S_indices[:]
    
    # 2. Simulated Annealing
    T = 1.0
    alpha = 0.999
    min_T = 0.001
    no_improve_count = 0
    max_no_improve = 1000
    
    # Precompute a list of all points for random selection
    all_points = list(range(NUM_POINTS))
    
    while time.time() < deadline and T > min_T:
        if random.random() < 0.3:
            # Try to add a random point
            p = random.choice(all_points)
            if not in_set[p]:
                if is_valid_add(p, S_indices):
                    do_add(p, S_indices)
                    if len(S_indices) > best_size:
                        best_size = len(S_indices)
                        best_S = S_indices[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
        else:
            # Try a swap
            if S_indices:
                p_out = random.choice(S_indices)
                p_in = random.choice(all_points)
                
                if in_set[p_in]:
                    continue
                
                # Check if p_in is valid with S \ {p_out}
                # Create a temporary list of S without p_out
                # To avoid copying the whole list, we can just pass a filtered list to is_valid_add?
                # is_valid_add takes S_indices. 
                # We can construct a new list for the check. 
                # Since |S| is small (< 50), this is fast.
                S_check = [x for x in S_indices if x != p_out]
                
                if is_valid_add(p_in, S_check):
                    # Perform swap
                    do_remove(p_out, S_indices)
                    do_add(p_in, S_indices)
                    
                    no_improve_count += 1
                    # Note: We don't track "energy" change, just accept valid swaps.
                    # This is effectively a random walk on the space of valid sets.
                else:
                    no_improve_count += 1
        
        # Kick mechanism
        if no_improve_count > max_no_improve:
            if S_indices:
                p = random.choice(S_indices)
                do_remove(p, S_indices)
                no_improve_count = 0
        
        T *= alpha
        
        if time.time() > deadline - 5:
            break

    return [tuple(coords[i]) for i in best_S]


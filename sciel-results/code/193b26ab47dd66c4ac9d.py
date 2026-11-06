"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 330s kill
    
    n_points = N * N
    
    # Index: point (x, y) -> index x * N + y
    def deidx(i):
        return (i // N, i % N)
    
    # Precompute squared distances
    coords = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = (diff ** 2).sum(axis=2).astype(np.int32)
    
    # Track forbidden distances: for each selected point i, which distances are "taken"
    # forbidden[i] is a set of squared distances that point i already has to other selected points
    # When adding point p, we need:
    # 1. d(p, i) not in forbidden[i] for all i in selected (apex at i)
    # 2. d(p, i) all distinct (apex at p)
    # 3. d(p, i) not equal to any d(j, k) where j,k in selected, j!=k, and {j,k} doesn't include i
    #    Actually simpler: for each pair (i,j) in selected, d(i,j) is forbidden for apex at i or j.
    #    So if we add p, d(p,i) must not equal d(i,j) for any j != i, and d(p,j) must not equal d(i,j).
    #    This is equivalent to: d(p,i) not in forbidden[i] for all i.
    
    selected = []
    selected_set = set()
    # forbidden[i] = set of distances from i to other selected points
    forbidden = {}
    
    def is_valid_fast(p):
        """Check if p can be added to current selected set."""
        k = len(selected)
        if k < 2:
            return True
        
        # Get distances from p to all selected points
        S_arr = np.array(selected, dtype=np.int32)
        d_p_S = dist_sq[p, S_arr]
        
        # Check 1: All distances from p to S must be distinct (apex at p)
        if len(np.unique(d_p_S)) < k:
            return False
        
        # Check 2: For each selected point i, d(p, i) must not be in forbidden[i]
        # forbidden[i] contains all distances from i to other selected points
        for i, s_idx in enumerate(selected):
            d_val = d_p_S[i]
            if d_val in forbidden[s_idx]:
                return False
        
        return True
    
    def add_point(p):
        """Add point p to selected set, updating forbidden distances."""
        selected.append(p)
        selected_set.add(p)
        forbidden[p] = set()
        
        # Update forbidden sets for all existing selected points
        for s_idx in selected[:-1]:
            d = dist_sq[p, s_idx]
            forbidden[p].add(d)
            forbidden[s_idx].add(d)
    
    # Greedy Phase with multiple random restarts
    random.seed(42)
    
    best_selected = []
    best_size = 0
    
    # Try multiple random orders
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Reset
        selected = []
        selected_set = set()
        forbidden = {}
        
        point_order = list(range(n_points))
        random.shuffle(point_order)
        
        for p in point_order:
            if time.time() > deadline:
                break
            
            if p in selected_set:
                continue
            
            if is_valid_fast(p):
                add_point(p)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = list(selected)
            # Copy state
            best_selected_set = set(selected)
            best_forbidden = {k: set(v) for k, v in forbidden.items()}
    
    # Local Search Phase starting from best found
    selected = best_selected
    selected_set = best_selected_set
    forbidden = best_forbidden
    
    remaining = [p for p in range(n_points) if p not in selected_set]
    
    # Track best during local search
    ls_best_size = len(selected)
    ls_best_selected = list(selected)
    ls_best_selected_set = set(selected)
    ls_best_forbidden = {k: set(v) for k, v in forbidden.items()}
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
        
        improved = False
        
        # Try to add a point
        if remaining:
            # Pick random candidates
            n_candidates = min(50, len(remaining))
            candidates = random.sample(remaining, n_candidates)
            for p in candidates:
                if is_valid_fast(p):
                    add_point(p)
                    remaining.remove(p)
                    ls_best_size = len(selected)
                    ls_best_selected = list(selected)
                    ls_best_selected_set = set(selected)
                    ls_best_forbidden = {k: set(v) for k, v in forbidden.items()}
                    no_improve_count = 0
                    improved = True
                    break
        
        # If no point added, try a swap
        if not improved and len(selected) > 0:
            # Pick a random point to remove
            r_idx = random.randint(0, len(selected) - 1)
            r = selected[r_idx]
            
            # Pick a random point to add
            if remaining:
                p = random.choice(remaining)
                
                # Check if removing r and adding p is valid
                # We need to check if p is valid with (selected - {r})
                # Save state
                saved_selected = list(selected)
                saved_selected_set = set(selected_set)
                saved_forbidden = {k: set(v) for k, v in forbidden.items()}
                
                # Remove r
                selected.pop(r_idx)
                selected_set.discard(r)
                for s_idx in selected:
                    forbidden[s_idx].discard(dist_sq[r, s_idx])
                del forbidden[r]
                
                # Check if p is valid
                if is_valid_fast(p):
                    add_point(p)
                    remaining.remove(p)
                    remaining.append(r)
                    ls_best_size = len(selected)
                    ls_best_selected = list(selected)
                    ls_best_selected_set = set(selected)
                    ls_best_forbidden = {k: set(v) for k, v in forbidden.items()}
                    no_improve_count = 0
                    improved = True
                else:
                    # Restore state
                    selected = saved_selected
                    selected_set = saved_selected_set
                    forbidden = saved_forbidden
        
        if not improved:
            no_improve_count += 1
        
        # Occasionally shuffle remaining
        if random.random() < 0.01:
            random.shuffle(remaining)
    
    # Use the best result found
    final_selected = ls_best_selected if ls_best_size >= best_size else best_selected
    
    result = [deidx(i) for i in final_selected]
    return result
# EVOLVE-BLOCK-END

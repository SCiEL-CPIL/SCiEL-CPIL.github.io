import time
import random
import numpy as np

N = 16
DEADLINE = 320.0 # Leave 10s buffer for 330s limit

# Precompute geometry
# Points are (x, y) in [0, N)^2
# Index: idx = y * N + x
points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
n_points = N * N

# Squared distance matrix
# dist2[i, j] = (points[i] - points[j])^2
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=-1)

def _can_add_np(idx_p, S_idx):
    """
    Check if point idx_p can be added to set S_idx (array of indices) without creating isosceles triangle.
    S_idx must be 1D array of indices.
    Returns True if can add, False otherwise.
    """
    if len(S_idx) < 2:
        return True
    
    # Distances from p to all points in S
    d_ps = dist2[idx_p, S_idx] # shape (k,)
    
    # Pairwise distances within S
    # We only need upper triangle i < j
    k = len(S_idx)
    if k == 2:
        # Only one pair (0, 1)
        d_s0_s1 = dist2[S_idx[0], S_idx[1]]
        d_p_s0 = d_ps[0]
        d_p_s1 = d_ps[1]
        if d_p_s0 == d_p_s1 or d_p_s0 == d_s0_s1 or d_p_s1 == d_s0_s1:
            return False
        return True
        
    # General case
    # Generate indices for all pairs i < j
    i_idx, j_idx = np.triu_indices(k, k=1)
    
    d_p_i = d_ps[i_idx]
    d_p_j = d_ps[j_idx]
    d_ij = dist2[S_idx[i_idx], S_idx[j_idx]]
    
    # Check isosceles conditions
    is_iso = (d_p_i == d_p_j) | (d_p_i == d_ij) | (d_p_j == d_ij)
    
    if np.any(is_iso):
        return False
        
    return True

def _is_maximal_np(S_idx, all_indices):
    """
    Check if S_idx is a maximal no-isosceles set.
    S_idx: array of indices of points in S.
    all_indices: array of all point indices.
    """
    S_set = set(S_idx)
    # Check all points not in S
    for p_idx in all_indices:
        if p_idx in S_set:
            continue
        if _can_add_np(p_idx, S_idx):
            return False
    return True

def _reduce_to_minimal_np(S_idx, all_indices, rng, time_limit=None, start_time=None):
    """
    Stochastic reduction.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 1.0
        
    S_list = list(S_idx)
    current_size = len(S_list)
    best_size = current_size
    best_set = S_list[:]
    
    # Randomize order of removal attempts
    while time.time() - start_time < time_limit:
        # Try to remove a random point
        # We pick a random point, check if removing it keeps maximality.
        # If yes, remove it.
        # To be stochastic, we might try multiple points or restart.
        
        # Strategy: Pick random point, try to remove. If fails, pick another.
        # If we remove one, restart the loop (or continue).
        # To avoid getting stuck, if no removal works in a pass, we might shuffle and try again?
        # Or just accept the local minimum.
        
        # Let's try a single pass of random removals.
        # Shuffle indices
        indices_to_try = list(range(len(S_list)))
        rng.shuffle(indices_to_try)
        
        removed_any = False
        for i in indices_to_try:
            if time.time() - start_time > time_limit:
                break
            if i >= len(S_list):
                continue
                
            p_to_remove = S_list[i]
            # Create new set without p_to_remove
            # S_list is a list of indices
            S_test = S_list[:i] + S_list[i+1:]
            
            if len(S_test) < 2:
                continue
                
            # Check if S_test is maximal
            # Note: S_test is a list, convert to np array for _is_maximal_np?
            # _is_maximal_np expects S_idx as array/list.
            # It uses set(S_idx) internally.
            
            if _is_maximal_np(S_test, all_indices):
                S_list = S_test
                removed_any = True
                best_size = len(S_list)
                best_set = S_list[:]
                # Restart the pass? Or continue?
                # Continuing is faster, but might miss better removals.
                # Let's continue but note that indices are now shifted.
                # Actually, S_list changed, so the loop indices are invalid.
                # Break and restart loop.
                break
                
        if not removed_any:
            # No removal found in this pass.
            # We could try a "swap" move here to escape local minimum?
            # Or just stop.
            # For now, stop reduction for this set.
            break
            
    return best_set

def _greedy_coverage_np(all_indices, rng):
    """
    Greedy construction using coverage heuristic.
    """
    remaining = set(all_indices)
    chosen = []
    
    start = rng.choice(list(remaining))
    remaining.remove(start)
    chosen.append(start)
    chosen_np = np.array(chosen)
    
    while remaining:
        best_p = None
        best_score = -1
        
        candidates = list(remaining)
        if len(candidates) > 100:
            candidates = rng.sample(candidates, 100)
        
        for p in candidates:
            # Check if can add
            if not _can_add_np(p, chosen_np):
                continue
                
            # Score: how many new points does this block?
            # A point q is blocked if {p, a, b} is isosceles for some a, b in chosen.
            # Or if {p, q, a} is isosceles for some a in chosen.
            # Actually, q is blocked if adding q to (chosen + p) is not possible.
            # i.e., _can_add_np(q, chosen_np + [p]) is False.
            
            new_chosen_np = np.append(chosen_np, p)
            
            score = 0
            # Sample some q to estimate score?
            # Or check all q in remaining?
            # remaining is up to 256.
            # _can_add_np is fast now.
            for q in remaining:
                if q == p:
                    continue
                if not _can_add_np(q, new_chosen_np):
                    score += 1
                    
            if score > best_score:
                best_score = score
                best_p = p
                
        if best_p is None or best_score <= 0:
            break
            
        remaining.remove(best_p)
        chosen.append(best_p)
        chosen_np = np.array(chosen)
        
    return chosen

def construct():
    start_time = time.time()
    
    all_indices = np.arange(N * N)
    
    best_set = None
    best_size = N * N + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Random greedy + reduction
    # We run multiple random greedy constructions.
    # Greedy: pick random point, if can add, add it.
    # This creates a maximal set (by construction, we add until no more can be added? No, we add if we *can*. 
    # If we iterate through all points and add those that can be added, the result is maximal.
    # Because if a point p could be added, we would have added it when we encountered it.
    # Wait, if we add A, then B might become non-addable. If we encounter B before A, we add B. Then A might become non-addable.
    # So a single pass through all points in random order, adding if possible, results in a maximal set.
    
    for attempt in range(500): # Increased attempts due to speedup
        if time.time() - start_time > DEADLINE - 5:
            break
            
        order = all_indices.copy()
        rng.shuffle(order)
        
        chosen = []
        chosen_np = np.array([], dtype=np.int32)
        
        for p in order:
            if _can_add_np(p, chosen_np):
                chosen.append(p)
                chosen_np = np.append(chosen_np, p)
                
        # Reduce to minimal
        reduced = _reduce_to_minimal_np(chosen, all_indices, rng, time_limit=2.0, start_time=start_time)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 6:
            break
            
    # Strategy 2: Coverage greedy + reduction
    for attempt in range(100):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        chosen = _greedy_coverage_np(all_indices, rng)
        
        if not chosen:
            continue
            
        # Ensure maximality (greedy coverage might stop early if score <= 0)
        # So we do a final pass to add any remaining valid points
        chosen_np = np.array(chosen)
        remaining_set = set(all_indices) - set(chosen)
        
        # Iterate through remaining points and add if possible
        # To be safe, shuffle them
        rem_list = list(remaining_set)
        rng.shuffle(rem_list)
        
        for p in rem_list:
            if _can_add_np(p, chosen_np):
                chosen.append(p)
                chosen_np = np.append(chosen_np, p)
                
        reduced = _reduce_to_minimal_np(chosen, all_indices, rng, time_limit=2.0, start_time=start_time)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 6:
            break
            
    # Strategy 3: Local Search on best found
    if best_set is not None and time.time() - start_time < DEADLINE - 10:
        S_list = best_set[:]
        improved = True
        while improved and time.time() - start_time < DEADLINE - 5:
            improved = False
            S_np = np.array(S_list)
            # Try to swap one point for another
            # Remove i, add j.
            for i in range(len(S_list)):
                if time.time() - start_time > DEADLINE - 5:
                    break
                p_remove = S_list[i]
                S_without = S_list[:i] + S_list[i+1:]
                S_without_np = np.array(S_without)
                
                # Find a point to add
                # We want a point p_add such that:
                # 1. _can_add_np(p_add, S_without_np) is True
                # 2. _is_maximal_np(S_without + [p_add], all_indices) is True
                # 3. Ideally, the new set can be reduced to size < len(S_list)
                
                # To speed up, we can just check if S_without + [p_add] is maximal.
                # If it is, we reduce it.
                
                # Iterate through candidates for p_add
                # We can sample candidates to save time
                candidates = list(set(all_indices) - set(S_without))
                rng.shuffle(candidates)
                
                for p_add in candidates[:20]: # Limit candidates
                    S_new = S_without + [p_add]
                    S_new_np = np.array(S_new)
                    
                    if _is_maximal_np(S_new, all_indices):
                        # Try to reduce
                        reduced = _reduce_to_minimal_np(S_new, all_indices, rng, time_limit=1.0, start_time=start_time)
                        if len(reduced) < len(S_list):
                            S_list = reduced
                            improved = True
                            best_size = len(S_list)
                            best_set = S_list[:]
                            break # Restart outer loop with new S_list
                            
                if improved:
                    break
                    
    if best_set is None:
        # Fallback
        chosen = []
        for p in all_indices:
            if _can_add_np(p, np.array(chosen)):
                chosen.append(p)
        return chosen
        
    return best_set

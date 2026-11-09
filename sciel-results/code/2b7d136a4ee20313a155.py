"""Combined approach: fast vectorized greedy construction with minimum-degree heuristic,
plus local search (remove-and-re-maximize) to escape local optima.
Uses NumPy for speed, multi-start, and deadline guard."""

import time
import random
import itertools
import numpy as np

N = 16
DEADLINE_SECONDS = 290  # safety margin before 300s kill

# Precompute all points and indices
ALL_POINTS = np.array(list(itertools.product(range(N), repeat=2)), dtype=np.int32)
NUM_POINTS = len(ALL_POINTS)

# Precompute squared distances between all pairs
diff = ALL_POINTS[:, np.newaxis, :] - ALL_POINTS[np.newaxis, :, :]
DIST_MATRIX = np.sum(diff * diff, axis=2)

def _can_add_indices(p_idx, S_indices):
    """
    Check if point p_idx can be added to set S_indices without creating an isosceles triangle.
    Returns True if no isosceles triangle is formed.
    """
    if len(S_indices) < 2:
        return True
    
    S_indices = np.array(S_indices)
    if len(S_indices) == 0:
        return True
        
    # Distances from p to all points in S
    d_p_s = DIST_MATRIX[p_idx, S_indices]
    
    # Check 1: Is p equidistant to any two points in S?
    # i.e., any duplicate in d_p_s
    if len(np.unique(d_p_s)) < len(d_p_s):
        return False

    # Distances between all pairs in S
    D_S = DIST_MATRIX[np.ix_(S_indices, S_indices)]
    
    # Check 2 & 3: Is d(p, S[i]) == d(S[i], S[j]) or d(p, S[j]) == d(S[i], S[j])?
    # Construct match matrix:
    # match_matrix[i, j] is True if d_p_s[i] == D_S[i, j] OR d_p_s[j] == D_S[i, j]
    
    # d_p_s shape (k,)
    # D_S shape (k, k)
    
    # We need to check off-diagonal elements effectively, but diagonal D_S[i,i]=0 and d_p_s[i]>0 (since p not in S)
    # So diagonal will never match. We can just check np.any.
    
    match_matrix = (d_p_s[:, np.newaxis] == D_S) | (d_p_s[np.newaxis, :] == D_S)
    
    if np.any(match_matrix):
        return False
        
    return True

def _block_count(p_idx, S_indices, addable_indices):
    """
    Count how many points in addable_indices would become non-addable if p_idx is added to S_indices.
    This is the 'degree' heuristic.
    """
    new_S = S_indices + [p_idx]
    count = 0
    for q in addable_indices:
        if q == p_idx:
            continue
        if not _can_add_indices(q, new_S):
            count += 1
    return count

def _build_greedy_min_degree(order_seed=0):
    """
    Greedy construction using minimum-degree heuristic.
    At each step, pick the addable point that blocks the fewest other addable points.
    """
    rng = random.Random(order_seed)
    S_indices = []
    addable = set(range(NUM_POINTS))
    
    while addable:
        best_p = None
        best_score = float('inf')
        
        candidates = list(addable)
        rng.shuffle(candidates)
        
        for p_idx in candidates:
            if not _can_add_indices(p_idx, S_indices):
                continue
            
            # Compute block count
            # Optimization: if block_count is already >= best_score, we can't improve (unless we want tie-breaking)
            # But block count calculation is expensive.
            # Let's just compute it.
            bc = _block_count(p_idx, S_indices, addable)
            
            # Add small random noise for tie-breaking to avoid symmetry traps
            score = bc + rng.random() * 0.01
            
            if score < best_score:
                best_score = score
                best_p = p_idx
                
        if best_p is None:
            break
            
        S_indices.append(best_p)
        addable.discard(best_p)
        
        # Update addable set
        new_addable = set()
        for q in addable:
            if _can_add_indices(q, S_indices):
                new_addable.add(q)
        addable = new_addable
        
    return S_indices

def _build_greedy_random(order_seed=0):
    """
    Simple random greedy construction. Faster, sometimes finds different structures.
    """
    rng = random.Random(order_seed)
    S_indices = []
    addable = set(range(NUM_POINTS))
    
    while addable:
        candidates = list(addable)
        rng.shuffle(candidates)
        
        found = False
        for p_idx in candidates:
            if _can_add_indices(p_idx, S_indices):
                S_indices.append(p_idx)
                addable.discard(p_idx)
                found = True
                break
        
        if not found:
            break
            
        # Update addable
        new_addable = set()
        for q in addable:
            if _can_add_indices(q, S_indices):
                new_addable.add(q)
        addable = new_addable
        
    return S_indices

def _local_search(S_indices, rng, max_iters=10):
    """
    Local search: try to remove a point and re-maximize. If the resulting maximal set is smaller, keep it.
    """
    current_S = list(S_indices)
    current_size = len(current_S)
    
    for _ in range(max_iters):
        if not current_S:
            break
            
        removed = False
        # Try removing each point
        indices_to_try = list(range(len(current_S)))
        rng.shuffle(indices_to_try)
        
        for i in indices_to_try:
            p_idx = current_S[i]
            temp_S = [x for j, x in enumerate(current_S) if j != i]
            
            # Re-maximize temp_S
            S_max = list(temp_S)
            addable = set(range(NUM_POINTS)) - set(S_max)
            
            # Greedy addition to maximize
            # Use random order for diversity
            while addable:
                candidates = list(addable)
                rng.shuffle(candidates)
                added = False
                for q in candidates:
                    if _can_add_indices(q, S_max):
                        S_max.append(q)
                        addable.discard(q)
                        added = True
                        break
                if not added:
                    break
                    
            if len(S_max) < current_size:
                current_S = S_max
                current_size = len(S_max)
                removed = True
                break
                
        if not removed:
            break
            
    return current_S

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = float('inf')
    
    # 1. Try minimum-degree greedy with multiple seeds
    # This is the primary structural approach.
    num_md_attempts = 20
    for seed in range(num_md_attempts):
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
            
        S_indices = _build_greedy_min_degree(order_seed=seed)
        
        # Local search refinement
        rng = random.Random(seed * 100)
        S_indices = _local_search(S_indices, rng, max_iters=5)
        
        if len(S_indices) < best_size:
            best_size = len(S_indices)
            best_set = S_indices
            
    # 2. Try random greedy with multiple seeds
    # This is faster and might find different local optima.
    num_rg_attempts = 50
    for seed in range(num_rg_attempts):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
            
        S_indices = _build_greedy_random(order_seed=seed)
        
        # Light local search
        rng = random.Random(seed * 100 + 50)
        S_indices = _local_search(S_indices, rng, max_iters=3)
        
        if len(S_indices) < best_size:
            best_size = len(S_indices)
            best_set = S_indices

    # 3. If we have time, do a few more local search passes on the best found set
    if best_set is not None and time.time() - start_time < DEADLINE_SECONDS - 20:
        rng = random.Random(999)
        # Try to improve the current best further
        improved = _local_search(best_set, rng, max_iters=10)
        if len(improved) < best_size:
            best_set = improved
            best_size = len(best_set)

    if best_set is None:
        # Fallback: simple greedy
        S_indices = _build_greedy_random(order_seed=0)
        best_set = S_indices

    return [tuple(ALL_POINTS[i]) for i in best_set]

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")

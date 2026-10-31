"""Improved program for kissing number in dimension 11."""
import time
import numpy as np
from itertools import combinations, product

D = 11
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _generate_candidates():
    """Generate candidate integer vectors of various weights."""
    candidates = []
    
    # Weight 1: ±1 in one position
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Weight 2: ±1 in two positions
    for i, j in combinations(range(D), 2):
        for si, sj in product((1, -1), repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[i], v[j] = si, sj
            candidates.append(v)
    
    # Weight 3: ±1 in three positions
    for i, j, k in combinations(range(D), 3):
        for si, sj, sk in product((1, -1), repeat=3):
            v = np.zeros(D, dtype=np.int64)
            v[i], v[j], v[k] = si, sj, sk
            candidates.append(v)
    
    # Weight 4: ±1 in four positions
    for i, j, k, l in combinations(range(D), 4):
        for si, sj, sk, sl in product((1, -1), repeat=4):
            v = np.zeros(D, dtype=np.int64)
            v[i], v[j], v[k], v[l] = si, sj, sk, sl
            candidates.append(v)
    
    return candidates

def _check_pairwise_constraint(vectors, new_vec):
    """Check if new_vec satisfies min_dist >= max_norm with all existing vectors."""
    if len(vectors) == 0:
        return True, 0.0
    
    # Compute max norm
    max_norm = 0.0
    for v in vectors:
        n = np.linalg.norm(v)
        if n > max_norm:
            max_norm = n
    new_norm = np.linalg.norm(new_vec)
    if new_norm > max_norm:
        max_norm = new_norm
    
    # Check all pairwise distances
    for v in vectors:
        d = np.linalg.norm(new_vec - v)
        if d < max_norm - 1e-9:
            return False, max_norm
    return True, max_norm

def _compute_max_norm(vectors):
    """Compute the maximum norm among vectors."""
    if len(vectors) == 0:
        return 0.0
    max_norm = 0.0
    for v in vectors:
        n = np.linalg.norm(v)
        if n > max_norm:
            max_norm = n
    return max_norm

def _is_valid_set(vectors):
    """Check if the set of vectors satisfies the constraint."""
    if len(vectors) <= 1:
        return True
    max_norm = _compute_max_norm(vectors)
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            d = np.linalg.norm(vectors[i] - vectors[j])
            if d < max_norm - 1e-9:
                return False
    return True

def _greedy_construct(candidates, deadline):
    """Greedy construction: add vectors if they satisfy the constraint."""
    selected = []
    selected_set = set()
    
    # Precompute norms
    norms = [np.linalg.norm(v) for v in candidates]
    
    for idx, v in enumerate(candidates):
        if time.time() > deadline:
            break
        
        v_tuple = tuple(v)
        if v_tuple in selected_set:
            continue
        
        # Quick check: if this vector's norm exceeds current max, 
        # we need to recheck all pairs
        current_max_norm = max([np.linalg.norm(sv) for sv in selected] + [norms[idx]]) if selected else norms[idx]
        
        compatible = True
        for sv in selected:
            d = np.linalg.norm(v - sv)
            if d < current_max_norm - 1e-9:
                compatible = False
                break
        
        if compatible:
            selected.append(v)
            selected_set.add(v_tuple)
    
    return selected

def _local_search(vectors, candidates, deadline):
    """Local search: try to replace one vector with another to increase count."""
    if len(vectors) == 0:
        return vectors
    
    selected = list(vectors)
    selected_set = set(tuple(v) for v in selected)
    n = len(selected)
    
    # Build a list of candidate indices not in selected
    available = []
    for idx, v in enumerate(candidates):
        v_tuple = tuple(v)
        if v_tuple not in selected_set:
            available.append(idx)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Try removing selected[i] and adding a candidate
            remaining = [selected[j] for j in range(n) if j != i]
            max_norm_remaining = max([np.linalg.norm(v) for v in remaining] + [0.0]) if remaining else 0.0
            
            for idx in available:
                if time.time() > deadline:
                    break
                v = candidates[idx]
                v_tuple = tuple(v)
                if v_tuple in selected_set:
                    continue
                
                new_max_norm = max(max_norm_remaining, np.linalg.norm(v))
                
                # Check if v is compatible with all remaining
                compatible = True
                for rv in remaining:
                    d = np.linalg.norm(v - rv)
                    if d < new_max_norm - 1e-9:
                        compatible = False
                        break
                
                if compatible:
                    # This swap doesn't increase count, so skip
                    # We only want to ADD vectors, not swap 1-for-1
                    pass
            
            # Try adding a new vector without removing any
            # (This is what greedy does, but local search can try different orders)
        
        # Try a different approach: for each pair of candidates, see if we can 
        # add one that's compatible with all selected
        for idx in available:
            if time.time() > deadline:
                break
            v = candidates[idx]
            v_tuple = tuple(v)
            if v_tuple in selected_set:
                continue
            
            new_max_norm = max(_compute_max_norm(selected), np.linalg.norm(v))
            compatible = True
            for sv in selected:
                d = np.linalg.norm(v - sv)
                if d < new_max_norm - 1e-9:
                    compatible = False
                    break
            
            if compatible:
                selected.append(v)
                selected_set.add(v_tuple)
                n += 1
                available.remove(idx)
                improved = True
                break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate candidates
    candidates = _generate_candidates()
    
    # Greedy construction
    selected = _greedy_construct(candidates, deadline)
    
    # Local search improvement
    if time.time() < deadline - 10:  # Leave 10s safety margin
        selected = _local_search(selected, candidates, deadline)
    
    return np.array(selected, dtype=np.int64)

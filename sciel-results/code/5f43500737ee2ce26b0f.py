import numpy as np
import itertools
import time

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def generate_weight_1_vectors():
    """Generate all weight-1 vectors in {-2, 0, 2}^D (norm^2 = 4)"""
    candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def greedy_mis(conflicts, deadline, rng):
    """
    Greedy Maximum Independent Set with degree-based ordering.
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    selected_indices = []
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
            
        # Find min degree among alive
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        
        # Random tie-breaking
        chosen = candidates[rng.integers(len(candidates))]
        selected_indices.append(chosen)
        
        # Find neighbors to remove
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        # Update alive
        alive[to_remove] = False
        
        # Update degrees for remaining alive nodes
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts

    return selected_indices

def local_improve(selected, vectors, dots, conflicts, deadline, rng):
    """
    Improved local search: try to add multiple compatible vectors at once.
    """
    N = len(vectors)
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected] = True
    current_selected = list(selected)
    current_size = len(current_selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Find all unselected vectors that don't conflict with current selected
        if np.any(selected_mask):
            has_conflict = conflicts[:, selected_mask].any(axis=1)
        else:
            has_conflict = np.zeros(N, dtype=bool)
            
        candidates = ~has_conflict & ~selected_mask
        
        if not np.any(candidates):
            break
            
        cand_indices = np.where(candidates)[0]
        
        # Sort candidates by degree (prefer low degree)
        cand_degrees = conflicts[cand_indices].sum(axis=1)
        sorted_cands = cand_indices[np.argsort(cand_degrees)]
        
        # Try to add as many as possible greedily
        added = []
        for c in sorted_cands:
            if time.time() > deadline:
                break
            if not np.any(conflicts[c] & selected_mask):
                selected_mask[c] = True
                added.append(c)
                
        if len(added) > 0:
            current_selected.extend(added)
            current_size += len(added)
            improved = True
            
            # If we added vectors, we might want to try to remove some to add more?
            # For now, just keep adding.
            
            # Check if we can remove some selected to add more?
            # This is complex. Let's stick to adding.
            
    return current_selected

def construct():
    t0 = time.time()
    deadline = t0 + 115  # Safety margin before 150s kill
    
    # Generate weight-4 vectors
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    # Generate weight-1 vectors (all compatible with weight-4 and each other)
    weight1_vectors = generate_weight_1_vectors()
    
    if N == 0:
        return weight1_vectors
        
    # Compute dot products for weight-4 vectors
    dots = vectors @ vectors.T
    
    # Conflict if dot > 2
    conflicts = (dots > 2).astype(np.int8)
    
    # Self-conflicts don't matter
    np.fill_diagonal(conflicts, 0)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Run multiple greedy MIS restarts
    mis_deadline = deadline - 30  # Leave 30s for local improvement
    attempt = 0
    while time.time() < mis_deadline:
        rng_attempt = np.random.default_rng(attempt * 12345 + 67890)
        current_set = greedy_mis(conflicts, mis_deadline, rng_attempt)
        attempt += 1
        
        if len(current_set) > len(best_set):
            best_set = current_set
    
    # Do local improvement on the best set found
    if len(best_set) > 0 and time.time() < deadline - 3:
        improved_set = local_improve(list(best_set), vectors, dots, conflicts, deadline - 2, rng)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    if not best_set:
        return weight1_vectors
        
    result_vectors_w4 = vectors[best_set].astype(np.int64)
    
    # Combine weight-4 and weight-1 vectors
    final_vectors = np.vstack([result_vectors_w4, weight1_vectors])
    
    return final_vectors

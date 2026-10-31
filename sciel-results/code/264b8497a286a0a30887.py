import numpy as np
import itertools
import time
import sys

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
    Try to improve the selected set by swapping out one vector for another.
    """
    N = len(vectors)
    selected_set = set(selected)
    current_size = len(selected)
    
    # Precompute: for each unselected vector, which selected vectors conflict with it
    # conflicts is (N, N) boolean
    unselected = np.array([i for i in range(N) if i not in selected_set])
    if len(unselected) == 0:
        return selected
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # For each selected vector, try to swap it out
        for idx in range(len(selected)):
            if time.time() > deadline:
                return selected
            v_out = selected[idx]
            # Find unselected vectors that don't conflict with any other selected vector
            # except v_out
            other_selected = [s for s in selected if s != v_out]
            # An unselected vector u can replace v_out if:
            # 1. u doesn't conflict with any other_selected
            # 2. u is not in selected_set
            
            # Check conflicts between unselected and other_selected
            if len(other_selected) == 0:
                compatible_unselected = unselected
            else:
                # conflicts[unselected, other_selected] is (len(unselected), len(other_selected))
                # u is compatible if no conflict with any other_selected
                conflict_mask = conflicts[np.ix_(unselected, other_selected)]
                compatible_unselected = unselected[~conflict_mask.any(axis=1)]
            
            if len(compatible_unselected) > 0:
                # Pick a random compatible unselected vector
                u_in = compatible_unselected[rng.integers(len(compatible_unselected))]
                # Swap
                selected[idx] = u_in
                selected_set.discard(v_out)
                selected_set.add(u_in)
                # Update unselected
                unselected = np.array([i for i in range(N) if i not in selected_set])
                improved = True
                break
    
    return selected

def construct():
    t0 = time.time()
    deadline = t0 + 115  # Safety margin
    
    # Generate weight-4 vectors
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    if N == 0:
        return np.empty((0, D), dtype=np.int64)
        
    # Compute dot products
    dots = vectors @ vectors.T
    
    # Conflict if dot > 2 (since norm^2 = 4, need dist^2 >= 4, so 4+4-2*dot >= 4 => dot <= 2)
    conflicts = (dots > 2).astype(np.int8)
    
    # Self-conflicts don't matter since we pick distinct vectors
    np.fill_diagonal(conflicts, 0)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Run multiple greedy MIS restarts
    n_restarts = 0
    while time.time() < deadline - 10:  # Leave 10s for local improvement
        current_set = greedy_mis(conflicts, deadline - 10, rng)
        n_restarts += 1
        
        if len(current_set) > len(best_set):
            best_set = current_set
        
        # If we found a large set, do local improvement
        if len(best_set) > 0 and time.time() < deadline - 5:
            improved_set = local_improve(list(best_set), vectors, dots, conflicts, deadline - 2, rng)
            if len(improved_set) > len(best_set):
                best_set = improved_set
    
    if not best_set:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    result_vectors = vectors[best_set]
    return result_vectors.astype(np.int64)

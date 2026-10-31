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

def generate_weight_1_vectors():
    """Generate all vectors of weight 1 with entry +/- 2"""
    candidates = []
    for i in range(D):
        for s in [2, -2]:
            v = [0] * D
            v[i] = s
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

def local_improve(conflicts, initial_set, deadline, rng):
    """
    Local search to improve the independent set.
    Tries to swap vertices to increase set size.
    """
    N = len(conflicts)
    current_set = set(initial_set)
    in_set = np.zeros(N, dtype=bool)
    for i in current_set:
        in_set[i] = True
    
    best_size = len(current_set)
    best_set = list(current_set)
    
    # Precompute conflicts for faster access
    # conflicts is a boolean matrix or int8 matrix
    
    while time.time() < deadline:
        # Try to add a random vertex not in set
        # If it conflicts, try to remove one of its neighbors and add it?
        # This is a 1-swap. Size stays same.
        # To increase size, we need to find a vertex that has 0 conflicts with the set.
        # Greedy MIS already does this.
        # Local search can try to "shake" the set.
        
        # Strategy: Randomly remove a few vertices, then greedily add back.
        # Or: Try to swap a vertex u in set with v not in set, if v has fewer conflicts 
        # with the remaining set, allowing more future additions.
        
        # Simpler: Random Restart Greedy
        # We'll just run greedy multiple times in the main loop.
        
        # For now, a simple hill climb:
        # Pick a random vertex u NOT in set.
        # Find its neighbors in set.
        # If neighbors is empty, add u. (Greedy should have caught this, but maybe not if order was bad)
        u = rng.integers(N)
        if not in_set[u]:
            neighbors_in_set = np.where(conflicts[u] & in_set)[0]
            if len(neighbors_in_set) == 0:
                in_set[u] = True
                current_set.add(u)
                best_size += 1
                best_set = list(current_set)
                continue
        
        # Try swap: remove one neighbor, add u?
        # If u has 1 neighbor v in set.
        # Remove v, add u. Size same.
        # But maybe u allows adding more others?
        # This is complex.
        
        # Let's just do a random restart of greedy within the time budget in the main function.
        # local_improve is not very effective here without a better heuristic.
        
    return best_set

def construct():
    t0 = time.time()
    deadline = t0 + 115
    
    # 1. Generate Weight-4 vectors
    vectors_w4 = generate_weight_4_vectors()
    N = len(vectors_w4)
    
    if N == 0:
        # Fallback
        w1 = generate_weight_1_vectors()
        return w1.astype(np.int64)
        
    # 2. Build Conflict Graph for W4
    dots = vectors_w4 @ vectors_w4.T
    conflicts = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflicts, 0)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # 3. Run Greedy MIS multiple times with different seeds to find a good set
    # We have ~115s. Greedy is fast.
    # Let's run greedy for the first 60s, keeping the best result.
    greedy_deadline = t0 + 60
    
    while time.time() < greedy_deadline:
        current_set = greedy_mis(conflicts, greedy_deadline, rng)
        if len(current_set) > len(best_set):
            best_set = current_set
            # Optional: Local improve?
            
    # 4. Generate Weight-1 vectors
    vectors_w1 = generate_weight_1_vectors()
    
    # 5. Combine
    # The W1 vectors are compatible with all W4 vectors.
    # They are compatible with each other.
    # So we just take the union.
    
    if not best_set:
        # If MIS failed, just return W1
        return vectors_w1.astype(np.int64)
        
    result_vectors = np.vstack([vectors_w4[best_set], vectors_w1])
    
    # 6. Verification (Optional, but good for safety)
    # Check if min pairwise distance >= max norm
    # Max norm is 2 (squared 4).
    # We need d(u,v)^2 >= 4 for all u != v.
    # This is equivalent to u.v <= 2 for all u,v with norm 2.
    # For W1 and W4, we proved this.
    # For W1 and W1, we proved this.
    # For W4 and W4, MIS ensures this.
    
    return result_vectors.astype(np.int64)

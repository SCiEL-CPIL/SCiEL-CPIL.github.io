import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict graph: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Use a more sophisticated approach: local search / hill climbing
    # Start with a greedy solution, then try to improve it
    
    # First, get a greedy initial solution
    alive = np.ones(N4, dtype=bool)
    selected = np.zeros(N4, dtype=bool)
    
    # Greedy: repeatedly pick the vertex with minimum degree among alive
    degrees = conflicts.sum(axis=1).astype(np.int64)
    rng = np.random.default_rng(42)
    
    while np.any(alive):
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        alive_degrees = degrees[alive_indices]
        min_deg = np.min(alive_degrees)
        cands = alive_indices[alive_degrees == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        selected[chosen] = True
        # Remove chosen and its neighbors
        to_remove = np.where(conflicts[chosen] | (np.arange(N4) == chosen))[0]
        alive[to_remove] = False
        # Update degrees
        remaining = np.where(alive)[0]
        if len(remaining) > 0:
            degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
    
    # Now try local search: try to swap in a non-selected vertex and swap out its neighbors
    # This is a simple 1-swap local search
    
    best_selected = selected.copy()
    best_count = best_selected.sum()
    
    # Precompute: for each non-selected vertex, how many selected vertices conflict with it?
    # And for each selected vertex, how many non-selected vertices conflict with it?
    
    # Let's iterate: try each non-selected vertex v. If we add v, we must remove all selected neighbors of v.
    # Net gain = 1 - (number of selected neighbors of v)
    # So we want v with negative net gain... wait, that's never positive.
    # Actually, we need to remove selected neighbors of v, and add v.
    # Net change = 1 - |selected_neighbors(v)|
    # This is only positive if v has no selected neighbors, which means v can be added freely.
    # But if v has no selected neighbors, we should have added it already in greedy.
    
    # So 1-swap doesn't help directly. Let's try 2-swap: add two vertices, remove their selected neighbors.
    # This is more complex.
    
    # Alternative: use a different heuristic. Let's try the "degree-based" with tie-breaking by number of non-neighbors.
    # Or use a maximum independent set algorithm with better heuristics.
    
    # Let's try: sort by degree, then use a different ordering.
    # Or use the "minimum degree" but with a different initial selection.
    
    # Actually, let's try multiple random restarts of the greedy algorithm and keep the best.
    
    best_result = best_selected
    best_count = best_count
    
    # Try a few more random restarts
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        rng2 = np.random.default_rng(42 + restart)
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int64)
        
        # Shuffle the order in which we consider vertices
        order = rng2.permutation(N4)
        
        for i in order:
            if not alive[i]:
                continue
            # Check if i is still the minimum degree among alive
            # This is expensive, so let's just pick the first alive vertex
            selected[i] = True
            to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
        
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_result = selected.copy()
    
    result_v4 = vectors_v4[best_result]
    
    # Add all V1 vectors (they're all compatible with V4 and each other)
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

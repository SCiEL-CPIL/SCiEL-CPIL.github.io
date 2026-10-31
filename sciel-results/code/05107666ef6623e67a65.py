import numpy as np
import itertools
import time
import sys
from collections import defaultdict

D = 11

def generate_weight_4_vectors():
    """Generate all weight-4 vectors in R^11 (exactly 4 non-zero entries, each ±1)."""
    vectors = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second margin before 150s kill
    
    # Generate all weight-4 vectors
    candidates = generate_weight_4_vectors()
    n = len(candidates)
    
    if n == 0:
        return np.empty((0, D), dtype=np.int64)
    
    # Precompute conflict matrix: conflict if dot > 2 (i.e., dot >= 3)
    # For weight-4 vectors, dot product ranges from -4 to 4
    # Two vectors are compatible if dot <= 2
    dots = candidates @ candidates.T
    conflict = dots > 2
    
    # Remove self-loops
    np.fill_diagonal(conflict, False)
    
    # Precompute degree of each vertex
    degrees = np.sum(conflict, axis=1)
    
    best_result = np.empty((0, D), dtype=np.int64)
    best_size = 0
    
    # Multiple greedy restarts with different orderings
    for seed in range(200):
        if time.time() > deadline:
            break
        
        # Create ordering: primarily by degree, with random perturbation
        rng = np.random.default_rng(seed)
        if seed == 0:
            order = np.argsort(degrees, kind='stable')
        else:
            # Mix of degree and random
            perturbed = degrees.astype(float) + rng.random(n) * 5
            order = np.argsort(perturbed, kind='stable')
        
        # Greedy independent set
        selected = []
        selected_bool = np.zeros(n, dtype=bool)
        available = np.ones(n, dtype=bool)
        
        for idx in order:
            if time.time() > deadline:
                break
            if not available[idx]:
                continue
            
            # Check if idx conflicts with any already selected
            if selected_bool[idx] or np.any(conflict[idx] & selected_bool):
                continue
            
            # Select idx
            selected.append(int(idx))
            selected_bool[idx] = True
            # Remove all neighbors from available
            available[conflict[idx]] = False
            available[idx] = False
        
        # Convert to array
        if len(selected) == 0:
            continue
        
        selected_arr = candidates[np.array(selected, dtype=np.int64)]
        cur_size = len(selected)
        
        # Local improvement: try to add vectors
        # Build set of selected vectors for quick lookup
        selected_set = set(map(tuple, selected_arr.tolist()))
        
        # Compute which candidates can be added
        # A candidate j can be added if dot with all selected <= 2
        dots_sel = selected_arr @ candidates.T  # shape (cur_size, n)
        max_dots = np.max(dots_sel, axis=0) if cur_size > 0 else np.zeros(n)
        addable = max_dots <= 2
        
        # Try to add addable candidates
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for j in range(n):
                if time.time() > deadline:
                    break
                if addable[j] and tuple(candidates[j].tolist()) not in selected_set:
                    # Add candidate j
                    selected_arr = np.vstack([selected_arr, candidates[j]])
                    selected_set.add(tuple(candidates[j].tolist()))
                    cur_size += 1
                    improved = True
                    # Recompute addable
                    dots_sel = selected_arr @ candidates.T
                    max_dots = np.max(dots_sel, axis=0)
                    addable = max_dots <= 2
        
        # Local improvement: try 1-swap
        # For each selected vector i, try removing it and adding a different vector j
        # This allows escaping local minima
        for _ in range(3):  # Limit swap iterations for time
            if time.time() > deadline:
                break
            
            swapped = False
            # Recompute addable for each possible removal
            for i in range(len(selected_arr)):
                if time.time() > deadline:
                    break
                
                # Remove i
                remaining = np.delete(selected_arr, i, axis=0)
                if len(remaining) == 0:
                    continue
                
                # Compute which candidates can be added to remaining
                dots_rem = remaining @ candidates.T
                max_dots_rem = np.max(dots_rem, axis=0)
                addable_rem = max_dots_rem <= 2
                
                # Find a candidate j that:
                # 1. Is not in selected_set
                # 2. Can be added to remaining
                # 3. Is not the removed vector i
                removed_vec = selected_arr[i]
                removed_tuple = tuple(removed_vec.tolist())
                
                best_j = -1
                for j in range(n):
                    if addable_rem[j] and tuple(candidates[j].tolist()) not in selected_set:
                        if tuple(candidates[j].tolist()) != removed_tuple:
                            best_j = j
                            break
                
                if best_j >= 0:
                    # Perform the swap
                    selected_arr = np.delete(selected_arr, i, axis=0)
                    selected_set.discard(removed_tuple)
                    selected_arr = np.vstack([selected_arr, candidates[best_j]])
                    selected_set.add(tuple(candidates[best_j].tolist()))
                    swapped = True
                    break
            
            if not swapped:
                break
        
        if cur_size > best_size:
            best_size = cur_size
            best_result = selected_arr
    
    return best_result

# EVOLVE-BLOCK-END

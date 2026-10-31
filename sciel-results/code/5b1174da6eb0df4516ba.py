"""Rewritten program for kissing number in dimension 11."""
import time
import numpy as np
import itertools

D = 11

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Generate all nonzero vectors in {-1, 0, 1}^11 with norm^2 = 4 (weight-4 vectors)
    # These have norm 2, so the constraint is: pairwise distance >= 2, i.e., dot product <= 2
    # For two weight-4 vectors with entries in {-1,0,1}, dot product ranges from -4 to 4.
    # We need dot product <= 2 for the distance constraint (since ||v||^2=4, ||w||^2=4,
    # ||v-w||^2 = 8 - 2*v.w >= 4 => v.w <= 2)
    
    # Generate all weight-4 vectors
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Compute the Gram matrix (dot products)
    # Two vectors conflict if their dot product > 2
    # We want a maximum independent set in the conflict graph
    # Greedy approach: sort by some heuristic, then add if compatible with all selected
    
    # Precompute dot product matrix
    # This is n_candidates x n_candidates, which is 5280 x 5280 = ~28M entries
    # That's manageable memory-wise (28M * 8 bytes = 224MB for int64)
    
    # Actually, let's be smarter. We'll use a greedy approach with randomization.
    # First, compute all dot products
    dot_matrix = candidates @ candidates.T
    
    # Conflict: dot product > 2
    conflict = dot_matrix > 2
    
    # Greedy MIS with random tie-breaking
    # Try multiple random orderings and keep the best
    best_set = []
    best_size = 0
    
    n_tries = 50
    rng = np.random.RandomState(42)
    
    for trial in range(n_tries):
        if time.time() > deadline - 10.0:
            break
        
        # Random ordering
        order = rng.permutation(n_candidates)
        
        selected = []
        selected_indices = set()
        
        for idx in order:
            if time.time() > deadline - 5.0:
                break
            
            # Check if idx is compatible with all selected
            # conflict[idx, selected_indices] should all be False
            if selected:
                sel_arr = np.array(selected)
                if np.any(conflict[idx, sel_arr]):
                    continue
            
            selected.append(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # If we have a good set, return it
    if best_set:
        result = candidates[best_set]
        return result
    else:
        # Fallback: return a small valid set
        return candidates[:1]

# EVOLVE-BLOCK-END

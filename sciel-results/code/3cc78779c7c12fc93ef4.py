import numpy as np
import itertools

D = 11

def construct():
    # Generate candidates of norm squared 2 and 3
    
    # Norm squared 2: +/- e_i +/- e_j
    # 220 vectors
    candidates = []
    
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
                    
    # Norm squared 3: +/- e_i +/- e_j +/- e_k
    # 1320 vectors
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
                            
    candidates = np.array(candidates)
    
    # Greedy Independent Set Selection
    # We want to maximize the number of vectors such that min pairwise dist >= max norm.
    # If we mix norms, the constraint is harder.
    # Strategy 1: Only norm 2. (220 vectors)
    # Strategy 2: Only norm 3. (Up to ~440 vectors)
    # Strategy 3: Mixed.
    
    # Let's try to build a set from Norm 3 vectors first, as they have higher potential count.
    # If we select a set S of norm 3 vectors, max norm is sqrt(3).
    # We need u dot v <= 1 for all u, v in S.
    
    # We will perform a greedy selection on the 1320 norm 3 vectors.
    # To improve the result, we can try different orderings or a simple greedy.
    
    norm3_candidates = candidates[220:]
    
    selected = []
    # Vectorized greedy:
    # Maintain a list of selected vectors.
    # For each candidate, check if it conflicts with any selected.
    # Conflict if dot > 1.
    
    # To speed up, we can use a matrix of dot products, but 1320x1320 is small.
    # However, checking against growing list is O(N^2).
    
    # Let's use a simple loop.
    # Sort candidates by some heuristic? 
    # Maybe randomize? Or just sequential.
    
    # We will try to pick as many as possible.
    # Since the graph is regular-ish, a simple greedy should get a decent fraction.
    
    # To ensure we don't time out, we limit the check.
    # 1320 candidates.
    
    # We will store selected vectors in a list.
    sel = []
    
    # Optimization: Precompute dot products? 
    # 1320 * 11 = 14520 floats. 
    # Dot product matrix 1320x1320 is 1.7M floats. ~13MB. OK.
    
    dot_matrix = norm3_candidates @ norm3_candidates.T
    
    # We need to select a subset of indices such that dot_matrix[i,j] <= 1 for all i!=j.
    # Greedy:
    # 1. Start with empty set.
    # 2. Iterate through candidates.
    # 3. If candidate is compatible with all in set, add it.
    
    # To get a better result, we can try multiple random permutations or just a fixed one.
    # Given the symmetry, the result might be stable.
    
    # Let's just do a single greedy pass.
    # To make it deterministic, we use the natural order.
    
    selected_indices = []
    
    for i in range(len(norm3_candidates)):
        # Check compatibility with all selected
        # selected_indices is a list of indices into norm3_candidates
        # We need dot_matrix[i, j] <= 1 for all j in selected_indices
        
        # If selected is empty, add i
        if not selected_indices:
            selected_indices.append(i)
            continue
            
        # Check conflicts
        # Get dot products of i with all selected
        dots = dot_matrix[i, selected_indices]
        if np.all(dots <= 1):
            selected_indices.append(i)
            
    # Convert to vectors
    if selected_indices:
        result = norm3_candidates[selected_indices]
    else:
        result = np.array([]).reshape(0, D)
        
    # Compare with the norm 2 set (220 vectors)
    # The norm 2 set is valid and has 220 vectors.
    # If our norm 3 set is larger, use it.
    
    if len(result) > 220:
        return result
    else:
        # Return the 220 norm 2 vectors
        return candidates[:220]

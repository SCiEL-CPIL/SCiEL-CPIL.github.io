import numpy as np
import itertools

D = 11

def construct():
    # We will try to find a large set of integer vectors
    # such that min pairwise distance >= max norm.
    
    # Candidate norms squared to try
    # We restrict to small norms to keep candidate set manageable.
    # Norm squared 1: 22 vectors
    # Norm squared 2: 220 vectors
    # Norm squared 4: Weight 4 (16*11C4=11880) + Weight 1 (44) + Weight 2 (0? no, 1^2+1^2=2)
    # Norm squared 5: Weight 2 (1^2+2^2=5). 11*10*4 = 440 vectors.
    
    # Let's generate candidates for norm squared <= 5.
    candidates = []
    
    # Norm squared 1: (±1, 0, ...)
    for i in range(D):
        for sign in [1, -1]:
            v = [0]*D
            v[i] = sign
            candidates.append(v)
            
    # Norm squared 2: (±1, ±1, 0, ...)
    for i in range(D):
        for j in range(i+1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0]*D
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    # Norm squared 4: 
    # Case 1: (±2, 0, ...)
    for i in range(D):
        for sign in [1, -1]:
            v = [0]*D
            v[i] = 2*sign
            candidates.append(v)
            
    # Case 2: (±1, ±1, ±1, ±1, 0, ...)
    # This is the current program's set.
    # 11C4 * 16 = 330 * 16 = 5280 vectors.
    # We might not need all of them if we mix with others.
    # But let's include them.
    for subset in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0]*D
            for idx, s in zip(subset, signs):
                v[idx] = s
            candidates.append(v)
                
    # Norm squared 5: (±1, ±2, 0, ...)
    for i in range(D):
        for j in range(i+1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0]*D
                    v[i] = s1
                    v[j] = 2*s2
                    candidates.append(v)
                    # Also swap positions? No, i, j are distinct indices.
                    # v[i]=1, v[j]=2 is distinct from v[i]=2, v[j]=1?
                    # Yes, but we iterate i, j and assign s1 to i, s2 to j.
                    # We need to cover both (1,2) and (2,1) on indices i,j.
                    # My loop assigns s1 to i, s2 to j.
                    # So (1,2) and (2,1) are generated if we vary s1,s2?
                    # s1=1, s2=1 -> 1, 2.
                    # s1=1, s2=-1 -> 1, -2.
                    # s1=-1, s2=1 -> -1, 2.
                    # s1=-1, s2=-1 -> -1, -2.
                    # We miss 2, 1 etc?
                    # Ah, the values are 1 and 2.
                    # I should generate (1,2) and (2,1) explicitly.
                    pass # Will fix in code

    # Actually, let's just generate all integer vectors with norm squared <= 5 directly by looping ranges?
    # D=11, range -2..2 for each coord? 5^11 is too big.
    # Better to generate by weight.
    
    # Let's refine the generation for Norm 5.
    # Vectors with two non-zero entries, one is ±1, other is ±2.
    # Indices i, j.
    # Values: (1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1).
    # So for each pair (i, j), 8 vectors.
    
    # Let's rebuild candidates list properly.
    candidates = []
    
    # Norm 1
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D; v[i] = s; candidates.append(v)
            
    # Norm 2
    for i in range(D):
        for j in range(i+1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0]*D; v[i] = s1; v[j] = s2; candidates.append(v)
                    
    # Norm 4
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D; v[i] = 2*s; candidates.append(v)
            
    for subset in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0]*D
            for idx, s in zip(subset, signs):
                v[idx] = s
            candidates.append(v)
            
    # Norm 5
    for i in range(D):
        for j in range(i+1, D):
            # (1, 2) and (2, 1) with signs
            for val_i in [1, -1, 2, -2]:
                for val_j in [1, -1, 2, -2]:
                    if val_i**2 + val_j**2 == 5:
                        v = [0]*D
                        v[i] = val_i
                        v[j] = val_j
                        candidates.append(v)

    # Greedy packing
    # We want to maximize k such that min_dist >= max_norm.
    # We can try different max_norm thresholds.
    # Let's just greedily add vectors, maintaining the constraint dynamically.
    # Sort candidates by norm? Or random?
    # Random shuffling multiple times is good for greedy.
    
    best_set = []
    best_score = 0
    
    # To speed up, we can pre-compute norms.
    # But D=11, candidates ~ 6000.
    # 6000 * 6000 distance checks is 36M. Fast in NumPy?
    # We can do it in a loop.
    
    import random
    
    # Try a few random orderings
    for _ in range(10):
        random.shuffle(candidates)
        current_set = []
        max_norm_sq = 0
        
        for v in candidates:
            v = np.array(v)
            v_norm_sq = np.dot(v, v)
            new_max_norm_sq = max(max_norm_sq, v_norm_sq)
            
            # Check if v is compatible with current_set
            # Distance to all u in current_set must be >= new_max_norm_sq
            # Wait, if we add v, the constraint for existing pairs (u1, u2) is still valid?
            # No, the condition is global: min_dist >= max_norm.
            # If max_norm increases, the required min_dist increases.
            # So existing pairs might violate the new constraint.
            # This makes dynamic greedy difficult.
            
            # Alternative: Fix the max_norm R.
            # Find max subset with norms <= R and pairwise dist >= R.
            # Try R = 1, 2, 4, 5 (squared norms).
            
            pass

    # Let's implement the "Fix R" approach.
    # R_sq values: 1, 2, 4, 5
    
    for R_sq in [1, 2, 4, 5]:
        # Filter candidates with norm_sq <= R_sq
        valid_cands = []
        for v in candidates:
            v_arr = np.array(v)
            if np.dot(v_arr, v_arr) <= R_sq:
                valid_cands.append(v_arr)
                
        if not valid_cands:
            continue
            
        # Greedy with fixed R_sq
        # We need pairwise dist >= R_sq.
        # Note: If all norms are <= R, and we ensure dist >= R, 
        # then the condition min_dist >= max_norm (which is <= R) is satisfied?
        # Yes, if min_dist >= R and max_norm <= R, then min_dist >= max_norm.
        
        # Sort by norm descending? Or random?
        # Usually, starting with larger vectors (closer to R) is better?
        # Or random.
        
        best_subset_for_R = []
        
        for _ in range(5): # 5 random trials per R
            random.shuffle(valid_cands)
            current_subset = []
            
            for v in valid_cands:
                # Check distance to all in current_subset
                ok = True
                for u in current_subset:
                    diff = v - u
                    if np.dot(diff, diff) < R_sq:
                        ok = False
                        break
                if ok:
                    current_subset.append(v)
                    
            if len(current_subset) > len(best_subset_for_R):
                best_subset_for_R = current_subset
                
        if len(best_subset_for_R) > best_score:
            best_score = len(best_subset_for_R)
            best_set = best_subset_for_R
            
    if not best_set:
        # Fallback to current logic or empty
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return np.array(best_set, dtype=np.int64)

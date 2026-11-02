"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct():
    # Generate Weight 3 candidates
    w3_candidates = []
    for i, j, k in combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    w3_candidates.append(v)
    
    # Generate Weight 4 candidates
    w4_candidates = []
    for i, j, k, l in combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        w4_candidates.append(v)
                        
    all_candidates = w3_candidates + w4_candidates
    all_candidates = np.array(all_candidates, dtype=np.int8)
    n_candidates = len(all_candidates)
    
    # Norms squared
    norms_sq = np.sum(all_candidates ** 2, axis=1)
    
    best_set = []
    best_size = 0
    
    # Multi-start greedy
    # Use a fixed seed for reproducibility, but vary the shuffle seed for trials
    base_seed = 42
    
    for trial in range(50):
        rng = np.random.RandomState(base_seed + trial)
        perm = rng.permutation(n_candidates)
        shuffled_candidates = all_candidates[perm]
        shuffled_norms = norms_sq[perm]
        
        selected_indices = []
        selected_vectors = []
        
        for idx in range(n_candidates):
            v = shuffled_candidates[idx]
            norm_v = shuffled_norms[idx]
            
            if not selected_vectors:
                selected_vectors.append(v)
                selected_indices.append(perm[idx])
                continue
            
            selected_matrix = np.array(selected_vectors, dtype=np.int8)
            # Compute dot products
            dots = selected_matrix @ v
            
            # Determine constraints
            # For each u in selected:
            # if norm_u == 3 and norm_v == 3: dot <= 1
            # if norm_u == 4 and norm_v == 4: dot <= 2
            # if mixed: dot <= 1
            
            # We can create a mask of allowed dots
            # norms_u = np.array([np.sum(u**2) for u in selected_vectors]) # Expensive
            # Instead, we can store norms with selected vectors
            
            # Let's optimize by storing norms in a list
            # Actually, let's just compute the max allowed dot for each u
            # Since v is fixed, we check each u.
            
            is_compatible = True
            # Vectorized check:
            # We need to compare dots against limits.
            # limits = np.where(norms_selected == 3, 1, 2) if norm_v == 4?
            # No, limits depend on both.
            
            # Let's store norms of selected vectors in a numpy array
            # selected_norms = np.array(selected_norms_list)
            
            # For now, simple loop inside the vectorized dot product?
            # dots is an array of shape (k,)
            # We need to check dots[i] <= limit(i, norm_v)
            
            # Let's maintain selected_norms as a list, convert to array for check
            # To avoid conversion cost, we can just iterate if k is small?
            # k grows up to ~500.
            # 500 iterations in Python is fast.
            
            # Let's use the Python loop for clarity and speed for small k
            # But we already computed dots.
            
            # Re-derive limits:
            # If norm_v == 3:
            #   limit is 1 if norm_u == 3, else 1 (mixed) -> always 1?
            #   Wait, if norm_u == 4, mixed limit is 1.
            #   So if norm_v == 3, limit is always 1.
            # If norm_v == 4:
            #   limit is 1 if norm_u == 3 (mixed).
            #   limit is 2 if norm_u == 4.
            
            if norm_v == 3:
                if np.any(dots > 1):
                    is_compatible = False
            else: # norm_v == 4
                # We need to know which selected vectors have norm 3 and which have 4
                # We can store this info.
                # For now, let's assume we have selected_norms array
                # We need to pass selected_norms to this block.
                
                # Let's refactor to keep selected_norms array
                pass # Placeholder

        # This structure is getting complex. Let's simplify.
        # We will store selected vectors and their norms in parallel lists.
        
    # Let's rewrite the inner loop properly.
    
    best_set = []
    best_size = 0
    
    for trial in range(50):
        rng = np.random.RandomState(base_seed + trial)
        perm = rng.permutation(n_candidates)
        shuffled_candidates = all_candidates[perm]
        shuffled_norms = norms_sq[perm]
        
        selected_vectors = []
        selected_norms = []
        
        for idx in range(n_candidates):
            v = shuffled_candidates[idx]
            norm_v = shuffled_norms[idx]
            
            if not selected_vectors:
                selected_vectors.append(v)
                selected_norms.append(norm_v)
                continue
            
            selected_matrix = np.array(selected_vectors, dtype=np.int8)
            dots = selected_matrix @ v
            
            if norm_v == 3:
                # All limits are 1
                if np.any(dots > 1):
                    continue
            else: # norm_v == 4
                # Limits: 1 if norm_u==3, 2 if norm_u==4
                # Create limit array
                # selected_norms is a list of ints
                sel_norms_arr = np.array(selected_norms)
                limits = np.where(sel_norms_arr == 3, 1, 2)
                if np.any(dots > limits):
                    continue
            
            selected_vectors.append(v)
            selected_norms.append(norm_v)
            
        if len(selected_vectors) > best_size:
            best_size = len(selected_vectors)
            best_set = selected_vectors
            
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END

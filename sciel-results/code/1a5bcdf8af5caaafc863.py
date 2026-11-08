"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    deadline = time.time() + 115
    
    # We will search over pools of integer vectors with small norms.
    # The constraint is: min pairwise distance >= max norm.
    # Let N be the max norm squared. We filter vectors with norm_sq <= N.
    # We want to select a subset S such that for all u, v in S, ||u-v||^2 >= N.
    # Note: ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v.
    # Since ||u||^2, ||v||^2 <= N, the condition is ||u||^2 + ||v||^2 - 2 u.v >= N.
    
    # Strategy:
    # 1. Generate candidate vectors with norm_sq <= N_max for various N.
    # 2. For a fixed N, we have a graph where vertices are valid vectors (norm<=N)
    #    and edges connect vectors that are "compatible" (distance^2 >= N).
    #    We want the Maximum Independent Set (MIS) of the incompatibility graph,
    #    or equivalently the Maximum Clique of the compatibility graph.
    # 3. Use a greedy heuristic with random restarts and local improvement (swap).
    
    # Candidate norms: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    # For D=11, vectors with entries in {-1, 0, 1} have norm_sq from 1 to 11.
    # We can also consider vectors with entries in {-2, -1, 0, 1, 2} but norm grows fast.
    # Let's stick to {-1, 0, 1} first as it's dense and likely optimal for small N.
    
    # Pre-generate all vectors in {-1, 0, 1}^11
    n_vectors = 3 ** D
    vectors = np.zeros((n_vectors, D), dtype=np.int64)
    for i in range(n_vectors):
        val = i
        for d in range(D):
            vectors[i, d] = val % 3 - 1
            val //= 3
            
    # Remove zero vector
    norms_sq = np.sum(vectors * vectors, axis=1)
    nonzero_mask = norms_sq > 0
    vectors = vectors[nonzero_mask]
    norms_sq = norms_sq[nonzero_mask]
    
    best_set = []
    best_score = 0
    
    # Try different max norm thresholds N
    # If N is small, the distance requirement is low, but the pool is restricted to low norm vectors.
    # If N is large, the pool is larger, but distance requirement is high.
    # The kissing number in 11D is known to be 460? No, bounds are tighter.
    # Actually, for D=11, the kissing number is between 460 and 545? 
    # Wait, the seed got 220+ with a specific construction.
    # Let's see what N gives the best result.
    
    candidate_Ns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    rng = np.random.default_rng(42)
    
    for N in candidate_Ns:
        if time.time() > deadline:
            break
            
        # Filter pool: norm_sq <= N
        mask = norms_sq <= N
        pool = vectors[mask]
        pool_norms = norms_sq[mask]
        n_pool = len(pool)
        
        if n_pool == 0:
            continue
        
        # Pre-compute compatibility matrix? 
        # n_pool can be large. For N=2, pool is weight 1 and 2.
        # Weight 1: 22. Weight 2: 220. Total 242.
        # For N=3, add weight 3: C(11,3)*2^3 = 165*8 = 1320. Total 1562.
        # For N=4, add weight 4: C(11,4)*2^4 = 330*16 = 5280. Total 6842.
        # Matrix 6842x6842 is ~47M entries. Might be too slow/memory heavy for Python loops.
        # We will use a greedy approach with on-the-fly distance checks, or vectorized checks.
        
        # Greedy with random restarts
        n_restarts = 10
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Random order
            indices = rng.permutation(n_pool)
            shuffled_pool = pool[indices]
            shuffled_norms = pool_norms[indices]
            
            selected = []
            selected_norms = []
            selected_indices = []
            
            for i in range(n_pool):
                v = shuffled_pool[i]
                v_norm = shuffled_norms[i]
                
                if not selected:
                    selected.append(v)
                    selected_norms.append(v_norm)
                    selected_indices.append(indices[i])
                    continue
                
                # Check distance to all selected
                # dist_sq = v_norm + sel_norms - 2 * dot(v, sel)
                # We need min(dist_sq) >= N
                # To speed up, we can stop early if any dist < N
                
                # Vectorized dot product
                dots = np.dot(selected, v) # Shape (M,)
                dists_sq = v_norm + np.array(selected_norms) - 2.0 * dots
                
                # Check if all are >= N
                if np.all(dists_sq >= N):
                    selected.append(v)
                    selected_norms.append(v_norm)
                    selected_indices.append(indices[i])
            
            current_score = len(selected)
            
            # Local improvement: try to swap out a vector to allow adding more?
            # Or just accept if better.
            if current_score > best_score:
                best_score = current_score
                best_set = selected.copy()
                
    # If we didn't find anything, fallback to weight-2 vectors
    if not best_set:
        mask = norms_sq == 2
        best_set = vectors[mask].tolist()
        best_score = len(best_set)
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END

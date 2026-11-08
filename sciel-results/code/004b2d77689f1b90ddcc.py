"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    deadline = time.time() + 115
    
    # Generate all integer vectors with entries in {-1, 0, 1}
    # Total 3^11 = 177147 vectors
    # This is a dense pool, unlike the sparse weight-based pool of the seed.
    
    # Efficient generation using base-3 representation
    n_vectors = 3 ** D
    vectors = np.zeros((n_vectors, D), dtype=np.int64)
    
    # Unpack base-3 digits
    # Instead of looping 177k times in Python, we can use broadcasting or simple loops
    # D=11 is small, so a loop is fine, or we can use np.repeat/tile tricks.
    # Let's just loop for clarity and safety, it's fast enough.
    for i in range(n_vectors):
        val = i
        for d in range(D):
            vectors[i, d] = val % 3 - 1
            val //= 3
            
    # Compute norms squared
    norms_sq = np.sum(vectors * vectors, axis=1)
    
    # We want to maximize k such that min_dist_sq >= max_norm_sq
    # Let max_norm_sq = N. We filter vectors with norm_sq <= N.
    # Then greedily pick vectors such that pairwise dist_sq >= N.
    
    best_set = []
    best_score = 0
    best_N = 0
    
    # Candidate max norms to try
    # Norms squared for {-1,0,1} vectors are integers from 1 to 11.
    # We should probably try N=2, 4, 6, 8, 10, 11?
    # If N is too small, we can't pack many. If N is too large, distance requirement is high.
    # The ratio is what matters, but since we fix N as the max norm, we are effectively
    # checking packings where the furthest point is at distance sqrt(N) and min distance is sqrt(N).
    
    candidate_Ns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    # To improve the greedy result, we do multiple random restarts for each N
    # and also try different ordering strategies (random, norm-sorted)
    
    rng = np.random.default_rng(42)
    
    for N in candidate_Ns:
        # Filter vectors with norm_sq <= N
        mask = norms_sq <= N
        pool = vectors[mask]
        pool_norms = norms_sq[mask]
        
        if len(pool) == 0:
            continue
            
        # Try a few random shuffles
        for restart in range(3):
            if time.time() > deadline:
                return np.array(best_set, dtype=np.int64) if best_set else np.array([], dtype=np.int64).reshape(0, D)
                
            # Randomize order
            indices = rng.permutation(len(pool))
            shuffled_pool = pool[indices]
            shuffled_norms = pool_norms[indices]
            
            selected = []
            selected_norms = []
            
            for i in range(len(shuffled_pool)):
                v = shuffled_pool[i]
                v_norm = shuffled_norms[i]
                
                if not selected:
                    selected.append(v)
                    selected_norms.append(v_norm)
                    continue
                
                # Check distance to all selected
                # D^2 = v_norm + sel_norms - 2 * dot(v, sel)
                # dot(v, sel) is (1, D) dot (M, D).T -> (1, M)
                dots = np.dot(selected, v) # Shape (M,)
                dists_sq = v_norm + np.array(selected_norms) - 2.0 * dots
                
                if np.min(dists_sq) >= N:
                    selected.append(v)
                    selected_norms.append(v_norm)
                    
            if len(selected) > best_score:
                best_score = len(selected)
                best_set = selected
                best_N = N
                
    # One more check: The seed program found 220+ vectors.
    # Our pool includes all weight-2 vectors.
    # If N=2, we can only pick weight-1 and weight-2 vectors.
    # Weight-1 vectors (norm 1) are too close to each other?
    # e1 and e2: dist^2 = 2. N=2. 2>=2 OK.
    # e1 and e1: same.
    # So we can potentially pick more than just weight-2?
    # But if we pick e1 (norm 1), and then e1+e2 (norm 2, dist^2=1 < 2). Fail.
    # So we can't mix weight 1 and weight 2 easily if N=2?
    # If we pick e1, we need dist >= sqrt(2).
    # e1 and e2: dist^2=2. OK.
    # e1 and -e2: dist^2=2. OK.
    # e1 and e3: dist^2=2. OK.
    # So we can pick all 22 weight-1 vectors?
    # e1, e2, ..., e11, -e1, ..., -e11.
    # Dist between e1 and e2 is sqrt(2).
    # Dist between e1 and -e2 is sqrt(2).
    # Dist between e1 and e1 is 0.
    # So yes, 22 vectors of weight 1 work with N=1?
    # If N=1, max norm is 1. Min dist must be >= 1.
    # e1 and e2: dist^2=2 >= 1.
    # So 22 vectors is valid for N=1.
    # But we want max k.
    # The weight-2 set has 220 vectors.
    
    # Ensure we return a valid array
    if not best_set:
        # Fallback to seed-like behavior if something failed
        # But the pool should contain the seed vectors.
        # Let's just return an empty array or a single vector to be safe?
        # The task says "nonzero integer vectors".
        # Let's return the 220 weight-2 vectors as a fallback if the greedy failed.
        # Actually, the greedy should find at least some.
        pass
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END

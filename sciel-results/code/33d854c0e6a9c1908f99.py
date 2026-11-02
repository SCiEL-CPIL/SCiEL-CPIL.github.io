"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # 1. Generate all roots of D11 lattice (vectors with two non-zero entries of +/- 1)
    # Norm squared = 2.
    # Condition for pairwise distance >= norm (sqrt(2)):
    # ||u-v||^2 >= 2 => 2+2 - 2u.v >= 2 => u.v <= 1.
    # For D11 roots, distinct roots have dot product in {-2, -1, 0, 1}.
    # So all 220 roots are mutually compatible.
    
    roots = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    roots.append(v)
                    
    roots_arr = np.array(roots, dtype=np.float64)
    n_roots = len(roots_arr)
    
    # 2. Generate candidates with squared norm 4
    # Type A: one entry +/- 2
    # Type B: four entries +/- 1
    
    candidates = []
    
    # Type A
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
            
    # Type B
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(support, signs):
                v[pos] = sign
            candidates.append(v)
            
    cands = np.array(candidates, dtype=np.float64)
    n_cands = len(cands)
    
    # 3. Filter candidates that are compatible with ALL roots
    # Condition: root . cand <= 1
    # Compute dot products: (n_roots, D) @ (D, n_cands) -> (n_roots, n_cands)
    # This is 220 * 5302 * 11 ops ~ 13M. Fast.
    
    if time.time() < deadline - 10:
        # Compute max dot product of each candidate with any root
        # If max dot > 1, candidate is invalid.
        # We want to check if ALL roots have dot <= 1.
        # So we check max(roots @ cand.T) <= 1.
        
        # Matrix multiplication might be memory heavy if done all at once?
        # 220 * 5302 floats is ~ 1M floats = 8MB. Fine.
        
        dots_roots_cands = roots_arr @ cands.T
        max_dots = np.max(dots_roots_cands, axis=0)
        
        # Keep only candidates where max dot <= 1
        valid_cand_mask = max_dots <= 1
        valid_cands = cands[valid_cand_mask]
        
    else:
        valid_cands = np.array([], dtype=np.float64)
        
    # 4. Greedily select from valid_cands
    # Condition between selected norm-4 vectors: dot <= 2
    # (Since max norm in set is now 2, min dist >= 2 => dot <= 2)
    
    selected_norm4 = []
    selected_norm4_dots = np.zeros((0, D), dtype=np.float64)
    
    # Sort valid candidates by some heuristic? 
    # Maybe random shuffle for diversity, or just iterate.
    # Given the strict constraint from roots, the pool might be small.
    
    if len(valid_cands) > 0:
        # Shuffle to avoid bias
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(valid_cands))
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
            v = valid_cands[idx]
            
            # Check compatibility with already selected norm-4 vectors
            if len(selected_norm4) > 0:
                dots = selected_norm4_dots @ v
                if np.any(dots > 2):
                    continue
            
            selected_norm4.append(idx)
            selected_norm4_dots = np.vstack([selected_norm4_dots, v])
            
    # Combine roots and selected norm-4 vectors
    if len(selected_norm4) > 0:
        norm4_vectors = valid_cands[selected_norm4]
        result = np.vstack([roots_arr, norm4_vectors]).astype(np.int64)
    else:
        result = roots_arr.astype(np.int64)
        
    # 5. Verification
    # Check min pairwise distance >= max norm
    if len(result) > 1:
        norms_sq = np.sum(result * result, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        # For verification, we can check a sample or all if small
        # 220 + few extra is small enough for O(N^2) check if optimized or if N < 1000
        # N ~ 220. N^2/2 = 24k pairs. 11 dims. Very fast.
        
        n = len(result)
        valid = True
        for i in range(n):
            if time.time() > deadline - 2:
                break
            for j in range(i + 1, n):
                diff = result[i] - result[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < max_norm_sq - 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            # Fallback to roots only if something went wrong (shouldn't happen)
            result = roots_arr.astype(np.int64)
            
    return result
# EVOLVE-BLOCK-END

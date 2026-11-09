"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Generate candidate vectors with small integer components
    # We'll consider vectors with norm squared up to 4 (i.e., up to 2 non-zero components with value 1, or 1 component with value 2)
    # Actually, let's be more systematic. Start with vectors that have small norms.
    
    # Strategy: Generate all vectors with entries in {-1, 0, 1} that have at least 2 non-zero entries
    # and norm squared >= 2. Then try to find a large independent set.
    
    # But that's too many. Let's be smarter.
    # The kissing number in 11D is 558. We need to get close to that.
    
    # Let's use a greedy approach with a priority based on how many conflicts each vector has.
    
    # First, generate a reasonable set of candidates.
    # Use vectors with 2 or 3 non-zero components, each being +/-1.
    
    candidates = []
    
    # All vectors with exactly 2 non-zero components (each +/-1)
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # All vectors with exactly 3 non-zero components (each +/-1)
    for combo in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # Compute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # We need min pairwise distance >= max norm
    # Let's try different max norm thresholds and see what works.
    
    # For now, let's use a greedy algorithm:
    # 1. Sort candidates by norm (ascending)
    # 2. For each candidate, check if it can be added to the current set
    #    (i.e., distance to all existing vectors >= current max norm)
    
    # But this is O(n^2 * D) which might be slow.
    
    # Let's use a different approach: build a conflict graph and use greedy independent set.
    
    # Actually, let's just do a simple greedy with early termination.
    
    # Sort by norm squared (ascending)
    order = np.argsort(norms_sq)
    sorted_candidates = candidates[order]
    sorted_norms_sq = norms_sq[order]
    
    selected = []
    max_norm_sq_current = 0
    
    for i in range(len(sorted_candidates)):
        if time.time() > deadline:
            break
            
        v = sorted_candidates[i]
        v_norm_sq = sorted_norms_sq[i]
        
        # New max norm squared would be max(max_norm_sq_current, v_norm_sq)
        new_max_norm_sq = max(max_norm_sq_current, v_norm_sq)
        
        # Check if v can be added: distance to all selected vectors >= new_max_norm_sq
        can_add = True
        for u in selected:
            diff = v - u
            dist_sq = np.sum(diff * diff)
            if dist_sq < new_max_norm_sq:
                can_add = False
                break
        
        if can_add:
            selected.append(v)
            max_norm_sq_current = new_max_norm_sq
    
    if len(selected) > 0:
        return np.array(selected, dtype=np.int64)
    else:
        # Fallback: return the base 220 vectors
        base_vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        base_vectors.append(np.array(v, dtype=np.int64))
        return np.array(base_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END

import time
import numpy as np
from itertools import product

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Generate a diverse pool of candidate integer vectors.
    # The seed program only used vectors with two non-zero entries of magnitude 1.
    # We expand this to include vectors with up to 3 non-zero entries, 
    # and magnitudes up to 2. This allows for "thicker" or differently angled vectors
    # that might fit better in the gaps of the initial configuration.
    
    candidates = []
    
    # Type 1: 2 non-zero entries (covers seed program's space)
    # Magnitudes 1, 2
    for i in range(D):
        for j in range(i + 1, D):
            for m1 in (1, -1, 2, -2):
                for m2 in (1, -1, 2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = m1
                    v[j] = m2
                    candidates.append(v)
    
    # Type 2: 3 non-zero entries (New geometry)
    # Magnitudes 1 only to keep norms manageable and avoid huge gaps
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for s1 in (1, -1):
                    for s2 in (1, -1):
                        for s3 in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates.append(v)
    
    # Convert to array for vectorized operations
    candidates = np.array(candidates, dtype=np.int64)
    
    # Remove duplicates (though construction above is mostly unique, signs might overlap in logic if not careful, 
    # but here indices are ordered i<j<k so unique combos. 
    # However, (1,1,0..) and (-1,-1,0..) are distinct. 
    # Let's ensure no exact duplicates just in case)
    candidates = np.unique(candidates, axis=0)
    
    # 2. Greedy Selection Strategy
    # We want to select a subset S of candidates such that 
    # min_{u,v in S, u!=v} ||u-v|| >= max_{u in S} ||u||
    
    # Note: The condition is ||u-v|| >= R_max.
    # If we normalize by R_max, it's equivalent to min ||u-v||/R_max >= 1.
    
    # Start with the most "standard" high-symmetry vectors from the seed program to ensure a valid baseline.
    # The seed program's vectors are the "edges" of the hypercube face, norm sqrt(2).
    # Let's initialize with the seed program's set to guarantee score >= 1.0 immediately.
    
    # Reconstruct seed set quickly
    seed_vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    seed_vectors.append(v)
    
    selected = np.array(seed_vectors, dtype=np.int64)
    
    # Calculate norms of selected
    norms_sq = np.sum(selected**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    min_dist_sq = np.inf
    
    # Function to check if a new vector can be added
    def can_add(v, current_selected, current_max_norm_sq):
        if len(current_selected) == 0:
            return True, np.sum(v**2)
        
        # Distance to all existing
        diffs = current_selected - v
        dists_sq = np.sum(diffs**2, axis=1)
        min_d_sq = np.min(dists_sq)
        
        # New max norm
        v_norm_sq = np.sum(v**2)
        new_max_norm_sq = max(current_max_norm_sq, v_norm_sq)
        
        # Condition: min pairwise distance >= max norm
        # So min_d_sq >= new_max_norm_sq
        if min_d_sq >= new_max_norm_sq:
            return True, new_max_norm_sq
        return False, current_max_norm_sq

    # We will try to add vectors from the candidate pool that are NOT in the seed set.
    # To make this efficient, we can sort candidates by "potential" or just iterate.
    # Since D=11, the pool is small enough (approx 11*10/2 * 16 + 11*10*9/6 * 8 ~ 880 + 1320 = 2200 vectors).
    # Checking each against ~220 selected vectors is fast.
    
    # Create a set of tuples for O(1) lookup of what's already selected
    selected_set = {tuple(v) for v in selected}
    
    # Candidates not in selected
    new_candidates = []
    for v in candidates:
        t = tuple(v)
        if t not in selected_set:
            new_candidates.append(v)
            
    new_candidates = np.array(new_candidates, dtype=np.int64)
    
    # Strategy: Iterate through new candidates. 
    # To maximize k, we want to add as many as possible.
    # Order matters. Vectors with norms close to the current max are easier to add 
    # because they don't increase the "threshold" (max_norm) as much as high-norm vectors would.
    # The current max norm is sqrt(2) (norm_sq=2).
    # Candidates with norm_sq=2 are the most compatible.
    # Candidates with norm_sq=3, 4, etc. require larger minimum distances.
    
    # Sort new candidates by norm_sq ascending
    norms_sq_new = np.sum(new_candidates**2, axis=1)
    sort_idx = np.argsort(norms_sq_new)
    sorted_new_candidates = new_candidates[sort_idx]
    
    added_count = 0
    for v in sorted_new_candidates:
        if time.time() > deadline:
            break
            
        # Check if v can be added
        # We need to check distance to ALL selected vectors
        diffs = selected - v
        dists_sq = np.sum(diffs**2, axis=1)
        min_d_sq = np.min(dists_sq) if len(selected) > 0 else np.inf
        
        v_norm_sq = np.sum(v**2)
        current_max_norm_sq = np.max(np.sum(selected**2, axis=1)) if len(selected) > 0 else 0
        new_max_norm_sq = max(current_max_norm_sq, v_norm_sq)
        
        if min_d_sq >= new_max_norm_sq:
            # Add it
            selected = np.vstack([selected, v])
            added_count += 1
            selected_set.add(tuple(v))
            
    # Final check: Ensure the result is valid.
    # Due to floating point or logic errors, we should verify.
    if len(selected) > 0:
        # Calculate min pairwise distance squared
        # For D=11 and k~200, O(k^2) is fine.
        min_pair_dist_sq = np.inf
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                d_sq = np.sum((selected[i] - selected[j])**2)
                if d_sq < min_pair_dist_sq:
                    min_pair_dist_sq = d_sq
        
        max_norm_sq = np.max(np.sum(selected**2, axis=1))
        
        # If invalid (shouldn't happen with greedy check, but safety first)
        if min_pair_dist_sq < max_norm_sq:
             # Fallback to seed only
             return np.array(seed_vectors, dtype=np.int64)
             
    return selected

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We need to ensure the interface matches.
# The prompt says: "Keep the function names and signatures of the seed program."
# Seed program: def construct(): return np.array(...)
# My code defines construct() correctly.
# EVOLVE-BLOCK-END

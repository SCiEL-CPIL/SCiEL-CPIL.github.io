import time
import numpy as np
from itertools import product

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Generate Candidate Pool
    # We consider integer vectors with norm squared 2 and 3.
    # Norm squared 2: two non-zero entries, each +/- 1.
    # Norm squared 3: three non-zero entries, each +/- 1.
    
    candidates = []
    
    # Type 1: Norm squared 2
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Type 2: Norm squared 3
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
                            
    candidates = np.array(candidates, dtype=np.int64)
    # Ensure uniqueness (though construction is unique by index order)
    candidates = np.unique(candidates, axis=0)
    
    # Pre-compute norms squared
    norms_sq = np.sum(candidates**2, axis=1)
    
    # The constraint is min pairwise distance squared >= max norm squared in the set.
    # If we only use vectors with norm squared <= 3, and we include at least one with norm squared 3,
    # then max_norm_sq = 3.
    # So we require d_min_sq >= 3.
    # If we only use norm squared 2, max_norm_sq = 2, require d_min_sq >= 2.
    # But we know we can get 220 with norm squared 2.
    # We suspect we can get > 220 by mixing or using norm squared 3.
    # Let's target max_norm_sq = 3.
    # This means ALL vectors in the final set must be at distance squared >= 3 from each other.
    
    # Greedy Randomized Search
    best_set = None
    best_k = 0
    
    # We will run multiple greedy passes with different random orderings
    # to try to find a larger independent set.
    
    num_trials = 200  # Adjust based on speed
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        # Randomly shuffle candidates
        idx = np.arange(len(candidates))
        np.random.shuffle(idx)
        shuffled_candidates = candidates[idx]
        
        selected = []
        # For efficiency, we can keep a list of selected vectors
        # and check distance to all of them.
        # Since k is expected to be ~250, and pool is ~1500, 
        # each check is O(k). Total O(k * pool) ~ 250 * 1500 = 375,000 ops per trial.
        # 200 trials * 375k = 75M ops. Should be fast in NumPy/Python?
        # Python loops are slow. Let's vectorize the check or optimize.
        
        # Optimization: 
        # We can maintain a matrix of distances? No, memory.
        # We can batch check?
        
        # Let's stick to a simple loop but optimize the inner check.
        # Actually, 1500 candidates * 250 selected = 375k distance calcs.
        # Each distance calc is dot product.
        # We can pre-compute candidate dot products with selected?
        
        # Alternative: Use a bitset or simple list.
        # Let's try a vectorized approach for the greedy step.
        
        # Actually, a simpler greedy:
        # 1. Start with empty set.
        # 2. For each candidate in shuffled order:
        #    a. Compute distance to all selected.
        #    b. If min dist >= 3, add to selected.
        
        selected_list = []
        for v in shuffled_candidates:
            if time.time() > deadline:
                break
                
            if not selected_list:
                selected_list.append(v)
                continue
                
            # Check if v can be added
            # Compute distances to all selected
            # selected_list is a list of arrays. Convert to array for vectorization?
            # Or just loop?
            # k is small (<300). Loop is fine.
            
            can_add = True
            for u in selected_list:
                diff = v - u
                d_sq = np.dot(diff, diff)
                if d_sq < 3:
                    can_add = False
                    break
            
            if can_add:
                selected_list.append(v)
                
        k = len(selected_list)
        if k > best_k:
            best_k = k
            best_set = np.array(selected_list, dtype=np.int64)
            
        # Optional: If we found 253, we can stop early? 
        # Theoretical max is 253.
        if best_k >= 253:
            break

    # If best_set is None (shouldn't happen), fallback to seed
    if best_set is None:
        # Fallback to seed program logic (norm 2 only)
        seed_vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        seed_vectors.append(v)
        return np.array(seed_vectors, dtype=np.int64)
        
    # Final verification
    # Ensure min pairwise distance squared >= max norm squared
    # Since we enforced d_sq >= 3 during construction, and all vectors have norm_sq <= 3,
    # max_norm_sq is either 2 (if no norm-3 vectors added) or 3.
    # If max_norm_sq = 2, we required d_sq >= 3? No, we required d_sq >= 3 always.
    # So if max_norm_sq = 2, condition d_sq >= 2 is satisfied by d_sq >= 3.
    # If max_norm_sq = 3, condition d_sq >= 3 is satisfied.
    
    # However, if we only added norm-2 vectors, max_norm_sq = 2.
    # Our check was d_sq >= 3. This is stricter than needed (d_sq >= 2).
    # This might limit the number of norm-2 vectors we can pack if we enforce d_sq >= 3.
    # But we are mixing with norm-3 vectors, so max_norm_sq will likely be 3.
    # If we end up with ONLY norm-2 vectors, we might have missed the 220 limit?
    # No, 220 vectors of norm 2 have min dist squared 2.
    # If we enforce d_sq >= 3, we cannot pack all 220.
    # We can only pack a subset of norm-2 vectors that are mutually at distance >= 3.
    # Distance between two norm-2 vectors:
    # If they are orthogonal (disjoint supports), d_sq = 4. OK.
    # If they share 1 index, d_sq = 2 (if signs differ) or 0 (if same, but they are distinct so not 0? 
    # (1,1,0) and (1,-1,0) -> diff (0,2,0) -> d_sq=4.
    # (1,1,0) and (-1,1,0) -> diff (2,0,0) -> d_sq=4.
    # (1,1,0) and (1,1,0) -> same vector.
    # Wait, distance between (1,1,0) and (1,-1,0) is 2. d_sq=4.
    # Distance between (1,1,0) and (-1,-1,0) is 2sqrt(2). d_sq=8.
    # Distance between (1,1,0) and (1,0,1) is sqrt(1+1+1)=sqrt(3). d_sq=3.
    # So norm-2 vectors can be at distance squared 2, 4, 6, 8...
    # Min distance squared for distinct norm-2 vectors is 2 (e.g., (1,1,0) and (1,-1,0) is 4? No.
    # (1,1,0) and (1,1,0) is 0.
    # (1,1,0) and (1,-1,0): diff (0,2,0), d_sq=4.
    # (1,1,0) and (-1,1,0): diff (2,0,0), d_sq=4.
    # (1,1,0) and (0,1,1): diff (1,0,-1), d_sq=2.
    # So yes, d_sq can be 2.
    # By enforcing d_sq >= 3, we exclude pairs with d_sq=2.
    # This reduces the capacity for norm-2 vectors.
    # But we are aiming for k > 220. The 220 set has many d_sq=2 pairs.
    # So we cannot simply take the 220 set.
    # We must rely on the norm-3 vectors to increase the count.
    
    return best_set

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END

import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Strategy:
    # Instead of a greedy independent set on a large pool, we use a hybrid pool of weight-1 and weight-4 vectors.
    # This leverages the structure of the Leech lattice / kissing number constructions in high dimensions.
    # Weight-1 vectors: 2 * 11 = 22 vectors (norm 1)
    # Weight-4 vectors: 11 choose 4 * 16 = 5280 vectors (norm 2)
    # Total pool ~ 5302 vectors.
    # 
    # The key insight is that weight-4 vectors with disjoint supports are distance 4 (norm 2), which is >= max norm (2).
    # Weight-1 vectors are distance sqrt(2) from each other (if disjoint) or 0 (if same) or 2 (if opposite).
    # If we include weight-1 vectors, max norm is 2.
    # dist(weight-1, weight-1) >= sqrt(2) < 2. So weight-1 vectors conflict with each other unless they are disjoint?
    # No, dist((1,0..), (0,1..)) = sqrt(2). sqrt(2) < 2. Conflict.
    # So we can't have many weight-1 vectors.
    # But we can have weight-4 vectors.
    # Let's focus on weight-4 vectors primarily, and maybe add some weight-1 if they fit.
    # Actually, the inspiration says "hybrid pool of weight-1 and weight-4 vectors".
    # Let's generate both and use local improvement.
    
    # 1. Generate Candidate Pool
    candidates = []
    
    # Weight 1
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int8)
            v[i] = s
            candidates.append(v)
            
    # Weight 4
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    N = len(candidates)
    
    # 2. Precompute Norms and Conflict Matrix
    norms_sq = np.sum(candidates * candidates, axis=1)
    max_norm_sq = np.max(norms_sq)  # Should be 4
    
    # Conflict if dist_sq < max_norm_sq
    dot = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot
    conflict = dist_sq < max_norm_sq
    
    # 3. Greedy Independent Set to get an initial set
    # Use a simple greedy: pick nodes with fewest conflicts first (minimum degree heuristic)
    # Or just random order. Let's try minimum degree first.
    degrees = np.sum(conflict, axis=1)
    
    best_indices = np.array([], dtype=int)
    best_size = 0
    
    # Run multiple greedy passes with different randomizations
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Randomize order slightly to escape local optima
        # We can shuffle the order of picking, or add small noise to degrees
        noise = np.random.rand(N) * 0.1
        priority = degrees + noise
        
        order = np.argsort(priority)
        
        current_selected = np.zeros(N, dtype=bool)
        count = 0
        
        for idx in order:
            if count == 0:
                current_selected[idx] = True
                count += 1
            else:
                if not np.any(conflict[idx] & current_selected):
                    current_selected[idx] = True
                    count += 1
        
        if count > best_size:
            best_size = count
            best_indices = np.where(current_selected)[0]
            
    # 4. Local Improvement (Swap-based Optimization)
    # Try to swap out one selected vector for one unselected vector to increase the set size or maintain it while allowing further additions.
    # This is a heuristic local search.
    
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[best_indices] = True
    unselected_mask = ~selected_mask
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # For each selected index, try to replace it with an unselected index
        # such that the new set is still valid and potentially allows more additions.
        # This is expensive, so we do a limited search.
        
        # A simpler local improvement: try to add any unselected vector that doesn't conflict.
        # This is already done in greedy, but maybe we can add more after swaps?
        
        # Let's try a simple hill climbing: 
        # Pick a random selected index i and a random unselected index j.
        # If replacing i with j allows the set to remain valid (i.e., j doesn't conflict with any other selected),
        # and if this swap enables adding another vector later, we do it.
        # But checking "enables adding" is hard.
        
        # Alternative: just try to add any compatible unselected vector.
        # If we can't add any, try swapping.
        
        # First, check if we can add any more vectors.
        added = False
        for idx in np.where(unselected_mask)[0]:
            if not np.any(conflict[idx] & selected_mask):
                selected_mask[idx] = True
                unselected_mask[idx] = False
                added = True
                improved = True
                # Break to restart the scan? Or continue?
                # Continue to add as many as possible in this pass.
        
        if not added:
            # Try swaps
            sel_idx = np.where(selected_mask)[0]
            unsel_idx = np.where(unselected_mask)[0]
            
            if len(sel_idx) == 0 or len(unsel_idx) == 0:
                break
                
            # Random sample of swaps to try
            num_swaps_to_try = 100
            for _ in range(num_swaps_to_try):
                if time.time() > deadline - 5:
                    break
                    
                i = sel_idx[np.random.randint(len(sel_idx))]
                j = unsel_idx[np.random.randint(len(unsel_idx))]
                
                # Check if j conflicts with any selected except i
                # We need to check conflict[j, selected_mask & ~onehot(i)]
                # This is O(N) per swap. 100 swaps * O(N) = 100*5000 = 500k ops. Fast.
                
                # Create a mask of selected excluding i
                temp_mask = selected_mask.copy()
                temp_mask[i] = False
                
                if not np.any(conflict[j] & temp_mask):
                    # Swap is valid
                    selected_mask[i] = False
                    selected_mask[j] = True
                    unselected_mask[i] = True
                    unselected_mask[j] = False
                    improved = True
                    # Restart the loop to try adding more
                    break
    
    # Finalize
    best_indices = np.where(selected_mask)[0]
    result_vectors = candidates[best_indices]
    
    return result_vectors.astype(np.int64)

import numpy as np
import itertools
import time

D = 11
DEADLINE = 110.0  # Safety margin for 150s wall-clock limit


# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    # Part 1: All 220 vectors of Type 1 (Two +/- 2s, rest zeros)
    # Norm^2 = 8
    vectors = []

    # Type 1: two coordinates are +/-2, rest are 0
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)

    # Part 2: Type 2 vectors (Eight +/-1s, three zeros)
    # Norm^2 = 8
    all_type2 = []
    for support in itertools.combinations(range(D), 8):
        base = [0] * D
        for idx in support:
            base[idx] = 1
        for signs in itertools.product([-1, 1], repeat=8):
            v = base.copy()
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            all_type2.append(v)

    all_type2 = np.array(all_type2, dtype=np.int8)
    n_type2 = len(all_type2)

    # Shuffle for diversity
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_type2)
    all_type2 = all_type2[perm]

    # Current set of accepted vectors
    accepted = np.array(vectors, dtype=np.int8)
    n_accepted = len(accepted)

    idx = 0
    # Use a smaller batch size to make the internal greedy check feasible and safe
    batch_size = 100
    
    while idx < n_type2:
        if elapsed() > DEADLINE:
            break

        end = min(idx + batch_size, n_type2)
        batch = all_type2[idx:end]  # shape (batch_len, D)
        
        if len(accepted) == 0:
             # If no accepted, take all? No, they must be mutually compatible.
             # But Type 1 are already accepted.
             pass

        # Check compatibility against all currently accepted vectors
        # Dots: (batch_len, n_accepted)
        dots = batch @ accepted.T
        # A candidate is compatible with accepted if all dots <= 4
        compatible_with_accepted = np.all(dots <= 4, axis=1)
        
        # Get indices of candidates compatible with the global set
        cand_indices = np.where(compatible_with_accepted)[0]
        
        if len(cand_indices) == 0:
            idx = end
            continue
            
        # Now, we need to select a subset of these candidates that are mutually compatible.
        # Greedy approach:
        # 1. Pick the first one.
        # 2. Remove all others that conflict with it.
        # 3. Repeat.
        
        # Extract the candidate vectors
        candidates = batch[cand_indices]
        
        # To speed up mutual checks, we can do it in a loop
        # Since batch_size is small (100), and we filter, the number of candidates
        # is likely small.
        
        selected_in_batch = []
        remaining = list(range(len(candidates)))
        
        while remaining:
            if elapsed() > DEADLINE:
                break
            
            # Pick the first remaining candidate
            pick_idx = remaining[0]
            v_pick = candidates[pick_idx]
            
            # Check which other remaining candidates conflict with v_pick
            # Dot products of v_pick with all remaining candidates
            # remaining_cands = candidates[remaining]
            # dots_pick = remaining_cands @ v_pick
            # conflict_mask = dots_pick > 4
            
            # Optimization: Only compute if len(remaining) > 1
            if len(remaining) == 1:
                selected_in_batch.append(candidates[pick_idx])
                break
                
            remaining_cands = candidates[remaining]
            dots_pick = remaining_cands @ v_pick
            conflict_mask = dots_pick > 4
            
            # Keep those that do NOT conflict
            # conflict_mask is True where conflict exists.
            # We want to keep indices where conflict is False.
            # But we also need to remove the picked one (index 0 in remaining list, which is at some position in candidates)
            
            # Let's map back to the 'remaining' list indices
            # remaining is a list of indices into 'candidates'
            # We want to update 'remaining'
            
            new_remaining = []
            for i, rem_idx in enumerate(remaining):
                if i == 0:
                    continue # This is the one we picked, so we add it to selected
                if not conflict_mask[i]:
                    new_remaining.append(rem_idx)
            
            selected_in_batch.append(candidates[pick_idx])
            remaining = new_remaining
            
        if selected_in_batch:
            new_vecs = np.array(selected_in_batch, dtype=np.int8)
            accepted = np.vstack([accepted, new_vecs])
            n_accepted = len(accepted)

        idx = end

    # Convert to list of integer vectors for return
    result = accepted.tolist()
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END

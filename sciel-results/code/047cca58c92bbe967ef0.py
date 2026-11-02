import numpy as np
import itertools
import time
import math

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate Candidates
    candidates = []
    
    # Type 1: One 2, rest 0: ±2e_i (Norm sq 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l (Norm sq 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type 3: Three 1s, rest 0: ±e_i ± e_j ± e_k (Norm sq 3)
    for idxs in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # 2. Compute Conflict Matrix
    # M[i,j] = 1 if i and j conflict (cannot be in same set)
    # Condition: dot(i,j) > threshold
    # threshold = min(norm_sq[i], norm_sq[j]) / 2
    
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Dot product matrix
    M = candidates @ candidates.T
    
    # Threshold matrix
    min_norms = np.minimum(norms_sq[:, None], norms_sq[None, :])
    # Threshold is min_norms / 2. Since dot products are integers, 
    # conflict if dot > min_norms/2 
    # Equivalent to 2*dot > min_norms
    # Or dot > floor(min_norms/2) if min_norms is odd?
    # If min_norms=3, thresh=1.5. dot > 1.5 => dot >= 2.
    # If min_norms=4, thresh=2.0. dot > 2.0 => dot >= 3.
    # So conflict if dot >= floor(min_norms/2) + 1
    
    thresh_int = min_norms // 2
    # Conflict if M > thresh_int
    conflict = (M > thresh_int).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # 3. Initial Greedy Solution
    # Use "least conflicts" heuristic for a strong start
    conflict_counts = np.sum(conflict, axis=1)
    order = np.argsort(conflict_counts, kind='stable')
    
    selected_mask = np.zeros(N, dtype=bool)
    selected_count = 0
    
    for idx in order:
        if not np.any(conflict[idx, selected_mask]):
            selected_mask[idx] = True
            selected_count += 1
            
    best_mask = selected_mask.copy()
    best_count = selected_count
    
    # 4. Simulated Annealing
    rng = np.random.default_rng(42)
    
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.99
    
    # Precompute lists of indices for faster random selection?
    # Or just use rng.integers
    
    steps = 0
    max_steps = 20000 # Adjust based on time
    
    while time.time() < deadline - 5 and steps < max_steps:
        T = T_start * (T_end / T_start) ** (steps / max_steps)
        
        # Pick a move
        # 50% Add, 20% Remove, 30% Swap
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try to Add
            unselected = np.where(~selected_mask)[0]
            if len(unselected) == 0:
                continue
            idx = unselected[rng.integers(len(unselected))]
            
            # Check compatibility
            if not np.any(conflict[idx, selected_mask]):
                # Valid add
                selected_mask[idx] = True
                new_count = selected_count + 1
                # Always accept if improves
                selected_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_mask = selected_mask.copy()
            else:
                # Invalid add, do nothing (or could try to remove something first, but that's a swap)
                pass
                
        elif move_type < 0.7:
            # Try to Remove
            selected = np.where(selected_mask)[0]
            if len(selected) == 0:
                continue
            idx = selected[rng.integers(len(selected))]
            
            # Always valid to remove
            selected_mask[idx] = False
            new_count = selected_count - 1
            
            # Accept with probability exp(-1/T) since delta = -1 (cost increase)
            # Wait, we maximize count. Cost = -count.
            # Delta cost = - (new_count - old_count) = - (-1) = 1.
            # Accept if exp(-1/T) > rand
            if rng.random() < math.exp(-1.0 / T):
                selected_count = new_count
            else:
                # Revert
                selected_mask[idx] = True
                
        else:
            # Try to Swap
            selected = np.where(selected_mask)[0]
            unselected = np.where(~selected_mask)[0]
            if len(selected) == 0 or len(unselected) == 0:
                continue
            
            idx_out = selected[rng.integers(len(selected))]
            idx_in = unselected[rng.integers(len(unselected))]
            
            # Check if idx_in is compatible with (selected_mask - idx_out)
            # We can check compatibility with all selected, and if it conflicts only with idx_out, it's valid.
            # Or simply:
            # Temporarily remove idx_out
            selected_mask[idx_out] = False
            
            if not np.any(conflict[idx_in, selected_mask]):
                # Valid swap (count stays same)
                selected_mask[idx_in] = True
                # Count is unchanged. 
                # We accept this to move the landscape. 
                # Since count is same, we always accept? Or with prob 1?
                # In SA for max clique, moves that preserve size are often accepted to explore.
                # Let's always accept valid swaps to explore the manifold of cliques of same size.
                pass
            else:
                # Revert
                selected_mask[idx_out] = True
            
        steps += 1
        
        # Periodically check time
        if steps % 1000 == 0 and time.time() > deadline - 10:
            break

    # 5. Final Local Search (Greedy Add)
    # Try to add any remaining compatible vertices
    unselected = np.where(~best_mask)[0]
    for idx in unselected:
        if time.time() > deadline - 2:
            break
        if not np.any(conflict[idx, best_mask]):
            best_mask[idx] = True
            best_count += 1

    if best_count == 0:
        # Fallback
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return candidates[best_mask].astype(np.int64)

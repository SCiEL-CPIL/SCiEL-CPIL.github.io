import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # 1. Generate Candidates
    candidates = []
    norms_sq = []
    
    # Type 1: Norm squared 4
    # +/- 2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            norms_sq.append(4)
            
    # Type 2: Norm squared 4
    # +/- e_i +/- e_j +/- e_k +/- e_l
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            norms_sq.append(4)
            
    # Type 3: Norm squared 8
    # +/- 2e_i +/- 2e_j
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    candidates.append(v)
                    norms_sq.append(8)

    candidates = np.array(candidates, dtype=np.int32)
    norms_sq = np.array(norms_sq, dtype=np.int32)
    N = len(candidates)
    
    # 2. Compute Compatibility
    # Dot products
    M = candidates @ candidates.T
    
    # Threshold T[i,j] = min(norms_sq[i], norms_sq[j]) // 2
    # We need M[i,j] <= T[i,j] for i != j
    # For i == j, M[i,i] = norms_sq[i], T[i,i] = norms_sq[i]//2. 
    # We don't care about self-compatibility for selection, but for bitsets it's easier to handle carefully.
    
    min_norms = np.minimum(norms_sq[:, None], norms_sq[None, :])
    thresholds = min_norms // 2
    
    # Compatibility matrix: True if compatible
    # Exclude self: for i==j, we want False in the "can add" logic, but True in "is compatible with self" logic?
    # Actually, if we have a set S, we check if v is compatible with all u in S.
    # If v is in S, we don't check.
    # So C[i,j] should be True if v_i and v_j can coexist.
    C = (M <= thresholds)
    # Diagonal: C[i,i] is True (4<=2 is False? No. 4 <= 2 is False. Wait.)
    # If i==j, M[i,i]=4. T[i,i]=2. 4<=2 is False.
    # So C[i,i] is False. This is good because a vector cannot be compatible with itself in the context of "adding to a set containing itself".
    # But for the logic "is v compatible with set S", if v is in S, we skip.
    
    # 3. Greedy Initialization
    # Use a few random greedy passes to get a good start
    rng = np.random.default_rng(42)
    
    best_mask = np.zeros(N, dtype=bool)
    best_count = 0
    
    def greedy_init():
        mask = np.zeros(N, dtype=bool)
        count = 0
        # Random order
        order = rng.permutation(N)
        
        for idx in order:
            if count == 0:
                mask[idx] = True
                count += 1
                continue
            
            # Check compatibility with all selected
            # C[idx, mask] will be a boolean array of length count
            # We need all to be True
            if np.all(C[idx, mask]):
                mask[idx] = True
                count += 1
        return mask, count

    # Run a few greedy inits
    for _ in range(20):
        if time.time() > deadline - 20:
            break
        m, c = greedy_init()
        if c > best_count:
            best_count = c
            best_mask = m.copy()
            
    # 4. Simulated Annealing / Local Search
    # State: best_mask
    # Moves: 
    # 1. Add a random unselected vector if compatible.
    # 2. Remove a random selected vector.
    # 3. Swap: Remove one, Add one (if the new one is compatible with the rest).
    
    current_mask = best_mask.copy()
    current_count = best_count
    
    # Precompute lists of indices for faster access?
    # We can just use np.where
    
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline - 5:
        # 1. Try to add a random unselected vector
        unselected_idx = np.where(~current_mask)[0]
        if len(unselected_idx) > 0:
            u = unselected_idx[rng.integers(len(unselected_idx))]
            if np.all(C[u, current_mask]):
                current_mask[u] = True
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_mask = current_mask.copy()
                temperature = max(temperature * 0.99, 0.1)
                continue
        
        # 2. Try to swap
        # Pick a random selected vector s
        selected_idx = np.where(current_mask)[0]
        if len(selected_idx) > 0:
            s = selected_idx[rng.integers(len(selected_idx))]
            # Pick a random unselected vector u
            unselected_idx = np.where(~current_mask)[0]
            if len(unselected_idx) > 0:
                u = unselected_idx[rng.integers(len(unselected_idx))]
                
                # Check if u is compatible with (current_mask \ {s})
                # Temporarily remove s
                current_mask[s] = False
                if np.all(C[u, current_mask]):
                    current_mask[u] = True
                    # current_count remains same
                    # Accept swap? 
                    # Swapping doesn't change count, but might lead to better future additions.
                    # We accept swap with probability 1 if it doesn't reduce count (it doesn't).
                    # But we should be careful not to oscillate.
                    # Let's just accept it.
                    temperature = max(temperature * 0.99, 0.1)
                    continue
        
        # 3. Try to remove a vector (to escape local minimum? No, we want to maximize count)
        # Removing reduces count, so only do if we expect to add more later.
        # For now, skip pure removals unless stuck.
        
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.5 # Reheat occasionally? Or just restart greedy?
            
    # 5. Final Polish: Try to add any remaining compatible vectors
    # Scan all unselected
    unselected_idx = np.where(~best_mask)[0]
    for u in unselected_idx:
        if time.time() > deadline - 2:
            break
        if np.all(C[u, best_mask]):
            best_mask[u] = True
            best_count += 1
            
    return candidates[best_mask].astype(np.int64)

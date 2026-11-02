import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # 1. Generate Candidates
    candidates = []
    norms_sq = []
    
    # Type 1: Norm sq 4 (One 2)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            norms_sq.append(4)
            
    # Type 2: Norm sq 4 (Four 1s)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            norms_sq.append(4)
            
    # Type 3: Norm sq 3 (Three 1s)
    for idxs in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            norms_sq.append(3)
            
    # Type 4: Norm sq 2 (Two 1s)
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            norms_sq.append(2)
            
    # Type 5: Norm sq 1 (One 1)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = s
            candidates.append(v)
            norms_sq.append(1)

    candidates = np.array(candidates, dtype=np.int32)
    norms_sq = np.array(norms_sq, dtype=np.int32)
    N = len(candidates)
    
    # 2. Compute Dot Products and Conflicts
    M = candidates @ candidates.T
    norms_sq_col = norms_sq[:, None]
    norms_sq_row = norms_sq[None, :]
    
    # Constraint: 2 * M[i,j] <= norms_sq[i] + norms_sq[j] - R_sq
    # We assume R_sq = 4. If the final set has max norm < 4, the constraint is looser, 
    # so using R_sq=4 is safe (conservative) and valid.
    R_sq = 4
    allowed_dot = (norms_sq_col + norms_sq_row - R_sq) // 2
    # Note: If (sum - R_sq) is odd, e.g., 3, allowed is 1. 
    # 2*M <= 3 => M <= 1.5 => M <= 1. Correct.
    # If sum - R_sq is negative, allowed is negative.
    # e.g., Norm 1 vs Norm 1: 1+1-4 = -2. allowed = -1.
    # 2*M <= -2 => M <= -1.
    # Since M is integer, M <= -1.
    
    conflict = (M > allowed_dot).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    best_set = None
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Helper: Check if vector idx can be added to current selected mask
    def can_add(idx, selected_mask):
        # selected_mask is bool array
        # Check if any selected node conflicts with idx
        if not np.any(conflict[idx] & selected_mask):
            return True
        return False

    # Helper: Local Search 1-2 Swap
    def local_search(selected_mask, count, max_iters=1000):
        # selected_mask: bool array
        # count: int
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add any compatible node
            unselected = np.where(~selected_mask)[0]
            if len(unselected) > 0:
                # Shuffle unselected to break symmetry
                rng.shuffle(unselected)
                for idx in unselected:
                    if can_add(idx, selected_mask):
                        selected_mask[idx] = True
                        count += 1
                        improved = True
                        break # Restart scan from beginning or continue? 
                        # Usually restart to ensure we don't miss interactions, 
                        # but for speed, we can just continue and do another pass.
                        # Let's restart the outer loop to be safe with 1-2 swaps later.
            
            if improved:
                continue
            
            # Try 1-2 Swap: Remove one, add two
            # Select a random node in S to remove
            selected_indices = np.where(selected_mask)[0]
            if len(selected_indices) < 2:
                break
            
            # To be efficient, we might limit swap attempts per iteration
            attempts = 0
            max_attempts = 50
            while attempts < max_attempts and time.time() < deadline - 5:
                attempts += 1
                # Pick a random node u to remove
                u = selected_indices[rng.integers(len(selected_indices))]
                
                # Tentatively remove u
                selected_mask[u] = False
                count -= 1
                
                # Now try to add two nodes v, w from unselected
                # Ideally v and w should conflict with u (that's why we removed it)
                # and be compatible with the rest of S.
                
                # Find candidates that were blocked by u or just random unselected
                # To be fast, pick random unselected nodes
                unselected = np.where(~selected_mask)[0]
                if len(unselected) < 2:
                    selected_mask[u] = True
                    count += 1
                    break
                
                # Pick two random candidates
                v_idx, w_idx = rng.choice(unselected, size=2, replace=False)
                
                # Check if v and w are compatible with current S (which excludes u)
                if can_add(v_idx, selected_mask) and can_add(w_idx, selected_mask):
                    # Check if v and w are compatible with each other
                    if not conflict[v_idx, w_idx]:
                        selected_mask[v_idx] = True
                        selected_mask[w_idx] = True
                        count += 2
                        improved = True
                        break
                    else:
                        # v and w conflict, undo
                        pass
                # If not successful, we don't undo u yet? 
                # No, if we fail to find a pair, we should probably restore u 
                # or try another pair. 
                # For simplicity, if this specific pair fails, we restore u and try another u?
                # Or keep u removed and try more pairs?
                # Keeping u removed allows adding 1 node potentially.
                # But we are in "1-2 swap" mode.
                # Let's just restore u if we don't find a pair in a few tries.
                
            if not improved:
                # Restore u if we removed it and didn't add 2
                if not selected_mask[u]:
                    selected_mask[u] = True
                    count += 1
                break # No improvement found in this iteration of local search

        return selected_mask, count

    # Strategy 1: Randomized Greedy + Local Search
    n_starts = 50
    for start in range(n_starts):
        if time.time() > deadline - 10:
            break
            
        if start == 0:
            order = np.arange(N)
        else:
            order = rng.permutation(N)
            
        selected_mask = np.zeros(N, dtype=bool)
        count = 0
        
        # Greedy Phase
        for idx in order:
            if can_add(idx, selected_mask):
                selected_mask[idx] = True
                count += 1
                
        # Local Search Phase
        selected_mask, count = local_search(selected_mask, count)
        
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()

    # Strategy 2: Greedy by Degree (Least Conflicts) + Local Search
    if time.time() < deadline - 20:
        # Sort by number of conflicts (ascending)
        conflict_counts = np.sum(conflict, axis=1)
        order = np.argsort(conflict_counts, kind='stable')
        
        selected_mask = np.zeros(N, dtype=bool)
        count = 0
        
        for idx in order:
            if can_add(idx, selected_mask):
                selected_mask[idx] = True
                count += 1
                
        selected_mask, count = local_search(selected_mask, count)
        
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
            
    # Strategy 3: Greedy by Degree (Most Compatible) + Local Search
    if time.time() < deadline - 20:
        compat_counts = N - 1 - np.sum(conflict, axis=1)
        order = np.argsort(-compat_counts, kind='stable')
        
        selected_mask = np.zeros(N, dtype=bool)
        count = 0
        
        for idx in order:
            if can_add(idx, selected_mask):
                selected_mask[idx] = True
                count += 1
                
        selected_mask, count = local_search(selected_mask, count)
        
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()

    if best_set is None:
        # Fallback: Return just Type 1
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return candidates[best_set].astype(np.int64)

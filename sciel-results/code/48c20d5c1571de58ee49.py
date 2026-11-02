import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # Generate candidates
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
    
    # Compute dot product matrix M
    M = candidates @ candidates.T
    
    # Compute squared norms
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    R_sq = 4
    
    # Threshold matrix: M[i,j] <= floor((norms_sq[i] + norms_sq[j] - R_sq) / 2)
    num = norms_sq[:, None] + norms_sq[None, :] - R_sq
    Thresh = num // 2
    
    # Conflict matrix: True where M > Thresh (cannot both be selected)
    conflict = (M > Thresh).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    best_count = 0
    best_set = None
    
    rng = np.random.default_rng(42)
    
    def greedy_select(order, selected_mask=None):
        """Greedy: select vertices in given order if compatible with all previously selected."""
        if selected_mask is None:
            selected_mask = np.zeros(N, dtype=bool)
        
        count = int(np.sum(selected_mask))
        
        for idx in order:
            if selected_mask[idx]:
                continue
            # Check compatibility with all currently selected using vectorized conflict check
            if not np.any(conflict[idx, selected_mask]):
                selected_mask[idx] = True
                count += 1
        
        return selected_mask, count
    
    # Strategy A: Multiple random restarts
    n_trials = 300
    for trial in range(n_trials):
        if time.time() > deadline - 20:
            break
        
        if trial == 0:
            order = np.arange(N)
        elif trial == 1:
            order = np.arange(N)[::-1]
        elif trial == 2:
            # Sort by norm (prefer higher norm vectors first)
            order = np.argsort(-norms_sq, kind='stable')
        elif trial == 3:
            # Sort by conflict count (least conflicts first)
            cc = np.sum(conflict, axis=1)
            order = np.argsort(cc, kind='stable')
        elif trial == 4:
            # Sort by compat count (most compatible first)
            cc = np.sum(conflict, axis=1)
            order = np.argsort(-(N - 1 - cc), kind='stable')
        else:
            order = rng.permutation(N)
        
        selected_mask = np.zeros(N, dtype=bool)
        _, count = greedy_select(order, selected_mask)
        
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy B: Greedy with "least conflicts" priority
    if time.time() < deadline - 30:
        conflict_counts = np.sum(conflict, axis=1)
        order_b = np.argsort(conflict_counts, kind='stable')
        
        selected_mask = np.zeros(N, dtype=bool)
        _, count = greedy_select(order_b, selected_mask)
        
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy C: Greedy with "most central" (highest compatibility degree)
    if time.time() < deadline - 25:
        compat_counts = N - 1 - conflict_counts
        order_c = np.argsort(-compat_counts, kind='stable')
        
        selected_mask = np.zeros(N, dtype=bool)
        _, count = greedy_select(order_c, selected_mask)
        
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy D: Iterative improvement - try adding unselected vertices
    if best_set is not None and time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            unselected = np.where(~best_set)[0]
            for idx in unselected:
                if time.time() > deadline - 8:
                    break
                if not np.any(conflict[idx, best_set]):
                    best_set[idx] = True
                    best_count += 1
                    improved = True
    
    # Strategy E: Swap-based local search
    if best_set is not None and time.time() < deadline - 5:
        selected_idx = np.where(best_set)[0]
        unselected_idx = np.where(~best_set)[0]
        
        for i in range(len(selected_idx)):
            if time.time() > deadline - 5:
                break
            si = selected_idx[i]
            # Temporarily remove si
            best_set[si] = False
            new_count = best_count - 1
            
            # Try adding unselected vertices
            added = []
            for uj in unselected_idx:
                if not np.any(conflict[uj, best_set]):
                    best_set[uj] = True
                    new_count += 1
                    added.append(uj)
            
            if new_count > best_count:
                best_count = new_count
            else:
                # Revert
                best_set[si] = True
                for u in added:
                    best_set[u] = False
    
    # Final scan: add any remaining compatible vertices
    if best_set is not None and time.time() < deadline - 2:
        unselected = np.where(~best_set)[0]
        for idx in unselected:
            if time.time() > deadline - 2:
                break
            if not np.any(conflict[idx, best_set]):
                best_set[idx] = True
                best_count += 1
    
    if best_set is None:
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return candidates[best_set].astype(np.int64)

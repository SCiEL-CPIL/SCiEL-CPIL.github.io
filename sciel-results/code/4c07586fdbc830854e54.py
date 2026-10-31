import numpy as np
import itertools
import time

def construct():
    D = 11
    deadline = time.time() + 110  # Safety margin
    
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
    
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Strategy 1: All norm^2=2 vectors (they all satisfy dist_sq >= 2 = max_norm^2)
    norm2_mask = norms_sq == 2
    k_strategy1 = norm2_mask.sum()
    
    # Strategy 2: Find max set with dist_sq >= 4 for all pairs (max norm = 2)
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    best_size = 0
    best_selected = None
    rng = np.random.RandomState(42)
    
    # Fast greedy with random order + local search
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        order = rng.permutation(n)
        selected = []
        selected_set = set()
        
        for idx in order:
            if time.time() > deadline:
                break
            if not np.any(conflict[idx, selected]):
                selected.append(idx)
                selected_set.add(idx)
        
        size = len(selected)
        if size > best_size:
            best_size = size
            best_selected = selected.copy()
        
        # Quick local search: try to add compatible vectors
        if size >= best_size - 3:
            current = selected.copy()
            cur_set = set(current)
            
            # Add phase
            while True:
                if time.time() > deadline:
                    break
                unselected = np.array([i for i in range(n) if i not in cur_set])
                if len(unselected) == 0:
                    break
                compatible = ~conflict[unselected][:, current].any(axis=1)
                if not compatible.any():
                    break
                to_add = unselected[np.where(compatible)[0][0]]
                current.append(to_add)
                cur_set.add(to_add)
            
            # Swap phase
            for _ in range(50):
                if time.time() > deadline:
                    break
                improved = False
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    idx = rng.randint(len(current))
                    to_remove = current[idx]
                    remaining = [x for x in current if x != to_remove]
                    
                    unselected = np.array([i for i in range(n) if i not in cur_set or i == to_remove])
                    if len(unselected) == 0:
                        break
                    
                    if remaining:
                        compatible = ~conflict[unselected][:, remaining].any(axis=1)
                    else:
                        compatible = np.ones(len(unselected), dtype=bool)
                    
                    if compatible.any():
                        to_add = unselected[np.where(compatible)[0][0]]
                        current[idx] = to_add
                        cur_set.discard(to_remove)
                        cur_set.add(to_add)
                        improved = True
                        break
                if not improved:
                    break
            
            if len(current) > best_size:
                best_size = len(current)
                best_selected = current
    
    # Choose better strategy
    if k_strategy1 > best_size:
        result = candidates[norm2_mask]
    else:
        result = candidates[best_selected]
    
    return result

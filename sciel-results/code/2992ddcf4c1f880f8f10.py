import time
import random
import numpy as np

_LCM = 232792560
_WEIGHTS = [_LCM // i for i in range(1, 23)]

def _find_collision(indices):
    """
    Returns (is_dissociated, conflict_indices).
    conflict_indices is a set of indices involved in a collision, or None.
    """
    k = len(indices)
    if k == 0:
        return True, None
    
    # Generate sums in mask-index order
    sums = np.array([0], dtype=np.int64)
    for v in [_WEIGHTS[i-1] for i in indices]:
        sums = np.concatenate((sums, sums + v))
        
    # Sort and find duplicates
    order = np.argsort(sums)
    sorted_sums = sums[order]
    
    diffs = np.diff(sorted_sums)
    dup_indices = np.where(diffs == 0)[0]
    
    if len(dup_indices) == 0:
        return True, None
        
    # Take the first duplicate
    idx = dup_indices[0]
    m1 = order[idx]
    m2 = order[idx+1]
    
    # Symmetric difference of masks gives the conflicting elements
    diff_mask = m1 ^ m2
    
    n = len(indices)
    conflict_local_indices = []
    for i in range(n):
        if (diff_mask >> i) & 1:
            conflict_local_indices.append(indices[i])
            
    return False, set(conflict_local_indices)

def _try_add_back(S, removed_elements, deadline):
    """
    Try to add back removed elements one at a time if possible.
    Returns the improved set.
    """
    S = set(S)
    improved = False
    
    # Try adding back each removed element in random order
    candidates = list(removed_elements)
    random.shuffle(candidates)
    
    for elem in candidates:
        if time.time() > deadline:
            break
        if elem in S:
            continue
            
        test_S = S | {elem}
        is_diss, _ = _find_collision(list(test_S))
        if is_diss:
            S = test_S
            improved = True
            
    return S, improved

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    all_elements = list(range(1, 23))
    best = []
    best_len = 0
    
    # Strategies for removal
    # 0: min weight
    # 1: max weight
    # 2: random
    # 3: max index
    # 4: min index
    
    strategies = [0, 1, 2, 3, 4]
    
    for strat in strategies:
        for trial in range(15): # 15 trials per strategy
            if time.time() > deadline:
                break
                
            S = set(all_elements)
            removed = []
            
            while time.time() < deadline:
                is_diss, conflict = _find_collision(list(S))
                
                if is_diss:
                    # Try to add back removed elements
                    S, improved = _try_add_back(S, removed, deadline)
                    if improved:
                        # Re-check and continue trying to add
                        is_diss, conflict = _find_collision(list(S))
                        if not is_diss and conflict:
                            # Continue removal loop
                            pass
                        else:
                            if len(S) > best_len:
                                best = list(S)
                                best_len = len(S)
                            break
                    else:
                        if len(S) > best_len:
                            best = list(S)
                            best_len = len(S)
                        break # Found a valid set for this trial
                    
                if conflict is None:
                    break # Error?
                    
                # Choose element to remove
                if strat == 0:
                    # Min weight
                    x = min(conflict, key=lambda i: _WEIGHTS[i-1])
                elif strat == 1:
                    # Max weight
                    x = max(conflict, key=lambda i: _WEIGHTS[i-1])
                elif strat == 2:
                    # Random
                    x = random.choice(list(conflict))
                elif strat == 3:
                    # Max index
                    x = max(conflict)
                else:
                    # Min index
                    x = min(conflict)
                    
                S.remove(x)
                removed.append(x)
                
            # If we exited due to deadline, check if current S is valid
            if time.time() >= deadline:
                if len(S) > 0:
                    is_diss, _ = _find_collision(list(S))
                    if is_diss:
                        # Try to add back
                        S, _ = _try_add_back(S, removed, deadline)
                        is_diss, _ = _find_collision(list(S))
                        if is_diss and len(S) > best_len:
                            best = list(S)
                            best_len = len(S)
                        
    return best

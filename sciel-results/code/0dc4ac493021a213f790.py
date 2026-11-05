<<<<<<< SEARCH
import time
import random

N = 40
_DEADLINE = time.time() + 85

def _find_diff_basis(target_size, max_val, seed=42):
    """Find a set S of size target_size from 1..max_val-1 such that
    pairwise differences cover 1..max_val-2. Must include 1 and max_val-1."""
    target = set(range(1, max_val - 1))  # 1..38
    rng = random.Random(seed)
    
    # Bitmask for differences 1..38
    # Bit i represents difference i+1? Or just use a set for clarity and switch to int if slow.
    # 38 bits fits in an int.
    # Let's use an integer bitmask where bit d is set if difference d is covered.
    
    def compute_covered(S):
        s_list = sorted(S)
        mask = 0
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                d = s_list[j] - s_list[i]
                if 1 <= d <= max_val - 2:
                    mask |= (1 << d)
        return mask
    
    target_mask = 0
    for d in range(1, max_val - 1):
        target_mask |= (1 << d)
        
    # Always include 1 and max_val-1
    fixed = {1, max_val - 1}
    remaining_size = target_size - len(fixed)
    
    best = None
    best_score = -1
    best_mask = 0
    
    for attempt in range(2000):
        if time.time() > _DEADLINE - 5:
            break
        # Choose remaining elements
        candidates = [x for x in range(2, max_val - 1) if x not in fixed]
        S = set(fixed)
        S.update(rng.sample(candidates, remaining_size))
        mask = compute_covered(S)
        score = bin(mask & target_mask).count('1')
        
        if score > best_score:
            best_score = score
            best = set(S)
            best_mask = mask
        
        if (mask & target_mask) == target_mask:
            return sorted(S)
        
        # Local search
        improved = True
        current_mask = mask
        current_score = score
        while improved and time.time() < _DEADLINE - 5:
            improved = False
            s_list = sorted(S)
            missing = target_mask & ~current_mask
            if not missing:
                return sorted(S)
            
            # Try to add a new element that covers the most missing diffs
            # Or swap an existing element
            # Swap is better for fixed size
            
            for idx in range(len(s_list)):
                old = s_list[idx]
                if old in fixed:
                    continue
                for nv in range(2, max_val - 1):
                    if nv in S and nv != old:
                        continue
                    S.remove(old)
                    S.add(nv)
                    new_mask = compute_covered(S)
                    new_score = bin(new_mask & target_mask).count('1')
                    if new_score > current_score:
                        current_score = new_score
                        current_mask = new_mask
                        improved = True
                        break
                    else:
                        S.remove(nv)
                        S.add(old)
                if (current_mask & target_mask) == target_mask:
                    return sorted(S)
                if improved:
                    break # Restart loop with new state
    
    if best is not None and (best_mask & target_mask) == target_mask:
        return sorted(best)
    return None
=======
import time
import random

N = 40
_DEADLINE = time.time() + 85

def _compute_mask(s_list, max_val):
    """Compute bitmask of covered differences for sorted list s_list."""
    mask = 0
    n = len(s_list)
    for i in range(n):
        for j in range(i + 1, n):
            d = s_list[j] - s_list[i]
            if 1 <= d <= max_val - 2:
                mask |= (1 << d)
    return mask

def _find_diff_basis(target_size, max_val, seed=42):
    """Find a set S of size target_size from 1..max_val-1 such that
    pairwise differences cover 1..max_val-2. Must include 1 and max_val-1."""
    target_mask = 0
    for d in range(1, max_val - 1):
        target_mask |= (1 << d)
    
    # Always include 1 and max_val-1
    fixed = [1, max_val - 1]
    remaining_size = target_size - len(fixed)
    candidates = [x for x in range(2, max_val - 1)]
    
    rng = random.Random(seed)
    
    best = None
    best_score = -1
    best_mask = 0
    best_set = None
    
    for attempt in range(5000):
        if time.time() > _DEADLINE - 5:
            break
        
        # Choose remaining elements
        S_list = fixed + rng.sample(candidates, remaining_size)
        S_list.sort()
        S_set = set(S_list)
        mask = _compute_mask(S_list, max_val)
        score = bin(mask & target_mask).count('1')
        
        if score > best_score:
            best_score = score
            best = list(S_list)
            best_mask = mask
            best_set = S_set
        
        if (mask & target_mask) == target_mask:
            return S_list
        
        # Local search: try swaps
        improved = True
        current_mask = mask
        current_score = score
        while improved and time.time() < _DEADLINE - 5:
            improved = False
            missing = target_mask & ~current_mask
            if not missing:
                return S_list
            
            # Try each non-fixed element for swap
            for idx in range(len(S_list)):
                old = S_list[idx]
                if old in fixed:
                    continue
                # Try swapping with each candidate not in S
                for nv in candidates:
                    if nv in S_set:
                        continue
                    # Perform swap
                    new_S_list = S_list[:]
                    new_S_list[idx] = nv
                    new_S_list.sort()
                    new_S_set = set(new_S_list)
                    new_mask = _compute_mask(new_S_list, max_val)
                    new_score = bin(new_mask & target_mask).count('1')
                    
                    if new_score > current_score:
                        S_list = new_S_list
                        S_set = new_S_set
                        current_mask = new_mask
                        current_score = new_score
                        improved = True
                        if (current_mask & target_mask) == target_mask:
                            return S_list
                        break
                if improved:
                    break  # Restart with new state
    
    if best is not None and (best_mask & target_mask) == target_mask:
        return best
    return None
>>>>>>> REPLACE


import numpy as np
import time

N = 512
K = 4
DEADLINE = 85.0  # seconds, safety margin before 90s

def _is_4ap_free(mask):
    """Check if a boolean mask of length N contains no 4-AP."""
    # For each element i in the set, for each d, check if i, i+d, i+2d, i+3d are all in set
    # Use bit operations for speed
    # Convert mask to numpy uint8 array
    arr = mask.astype(np.uint8)
    # For each d, check all i where i+3d < N
    for d in range(1, (N - 1) // 3 + 1):
        # indices: i, i+d, i+2d, i+3d for i in 0..N-1-3d
        max_i = N - 1 - 3 * d
        if max_i < 0:
            break
        # Vectorized check
        a = arr[0:max_i+1]
        b = arr[d:max_i+1+d]
        c = arr[2*d:max_i+1+2*d]
        e = arr[3*d:max_i+1+3*d]
        if np.any(a & b & c & e):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d) with a, a+3d in [1, N]
    # We'll use 0-indexed internally
    ap_list = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # Greedy construction: sort elements by degree (number of 4-APs they're in), add if no conflict
    # But first, let's do a simple greedy: try to add each element in order of smallest "conflict"
    
    # Compute degree of each element
    degree = np.zeros(N, dtype=np.int32)
    for ap in ap_list:
        for idx in ap:
            degree[idx] += 1
    
    # Greedy: process elements in order of increasing degree
    order = np.argsort(degree)
    
    mask = np.zeros(N, dtype=np.uint8)
    # For each AP, track if it's "hit" (has an element selected)
    # Actually, we need to track if adding an element would complete any AP
    # Better: maintain for each element, which APs it participates in
    # When we select an element, mark all its APs as "partially selected"
    # An element can be added if none of its APs are already fully selected
    
    # Alternative: maintain conflict count per AP
    # When we select element e, for each AP containing e, increment count
    # Element e can be added if for all APs containing e, the other 3 elements are NOT all selected
    
    # Simpler approach: just check each time using the mask
    # But that's O(N * num_aps) which might be slow
    
    # Let's use a different approach: backtracking with heuristic
    # Or: local search from a greedy start
    
    # Greedy start
    for idx in order:
        # Check if adding idx would create a 4-AP
        can_add = True
        for ap in ap_list:
            if idx in ap:
                others = [x for x in ap if x != idx]
                if all(mask[x] for x in others):
                    can_add = False
                    break
        if can_add:
            mask[idx] = 1
    
    best_mask = mask.copy()
    best_size = int(np.sum(best_mask))
    
    # Local search: try to add elements, then swap
    # First, try to add any element that doesn't create a 4-AP
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(N):
            if not mask[idx]:
                can_add = True
                for ap in ap_list:
                    if idx in ap:
                        others = [x for x in ap if x != idx]
                        if all(mask[x] for x in others):
                            can_add = False
                            break
                if can_add:
                    mask[idx] = 1
                    changed = True
                    best_mask = mask.copy()
                    best_size += 1
                    break
    
    # Swap-based local search
    # Try swapping out one element and in another
    # Or: try removing one element and adding two others
    
    # Simpler: random restarts with greedy
    import random
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Random greedy
        random_order = list(range(N))
        rng.shuffle(random_order)
        
        # Sort by random degree or just use random order
        rm = np.zeros(N, dtype=np.uint8)
        for idx in random_order:
            can_add = True
            for ap in ap_list:
                if idx in ap:
                    others = [x for x in ap if x != idx]
                    if all(rm[x] for x in others):
                        can_add = False
                        break
            if can_add:
                rm[idx] = 1
        
        # Try to improve by adding
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(N):
                if not rm[idx]:
                    can_add = True
                    for ap in ap_list:
                        if idx in ap:
                            others = [x for x in ap if x != idx]
                            if all(rm[x] for x in others):
                                can_add = False
                                break
                    if can_add:
                        rm[idx] = 1
                        improved = True
                        break
        
        if int(np.sum(rm)) > best_size:
            best_mask = rm.copy()
            best_size = int(np.sum(rm))
        
        if time.time() >= deadline:
            break
    
    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if best_mask[i]]
    return result

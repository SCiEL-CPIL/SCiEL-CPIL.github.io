import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 55.0
    n = 12
    m = 40
    total = 1 << n

    # Precompute comparability matrix as a numpy bool array: comp_mat[i,j] = True if i is strict subset of j
    # comp_mat[i,j] means i ⊂ j (strict)
    masks = np.arange(total, dtype=np.int32)
    # For each pair (i,j), check if i is strict subset of j: (i & j) == i and i != j
    # This is O(total^2) but done once in numpy
    # Actually, let's think differently: i is subset of j iff (i | j) == j
    # Strict subset: (i | j) == j and i != j
    # We can compute: for each i, the supersets are those j where (j & i) == i and j != i
    # Let's build a list of comparable pairs more efficiently
    
    # Build comparability: for each mask i, list of masks j where i ⊂ j or j ⊂ i
    # Use the fact that i ⊂ j iff (j & ~i) != 0 and (j & i) == i
    # i ⊃ j iff (i & ~j) != 0 and (i & j) == j
    
    # Precompute popcount for ranking
    popcount = np.zeros(total, dtype=np.int32)
    for i in range(1, total):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # For fast 1-swap: precompute for each candidate new_mask, its comparability with all other masks
    # comp_count[new_mask, existing_mask] = 1 if comparable
    # We'll compute gain on the fly using numpy
    
    # Build initial solution: use a chain-like structure
    # Strategy: pick sets that form many comparabilities
    # A good structure: for k=0..5, take all subsets of {0,...,k} of size <= k
    # But we need exactly 40 distinct sets
    
    # Better initial: use a "layered" approach
    # Layer 0: empty set
    # Layer 1: singletons {0}, {0,1}, {0,1,2}, ..., up to size 5
    # This creates a chain plus branches
    
    initial = []
    # Chain: {}, {0}, {0,1}, {0,1,2}, ..., {0,1,2,3,4,5}
    for k in range(7):
        mask = (1 << k) - 1
        initial.append(mask)
    # Add sets that are comparable with multiple chain elements
    # {0,1,2,3,4,5,6}, {0,1,2,3,4,5,7}, ...
    for extra in range(6, 12):
        mask = (1 << 6) - 1
        mask |= (1 << extra)
        initial.append(mask)
    # That's 7 + 6 = 13, need 27 more
    # Add: {0,1,2,3,4}, {0,1,2,3,5}, {0,1,2,4,5}, etc. (size 5 subsets of {0..5})
    # C(6,5) = 6
    for i in range(6):
        mask = (1 << 6) - 1
        mask &= ~(1 << i)
        initial.append(mask)
    # That's 19, need 21 more
    # Add size-4 subsets of {0..5}: C(6,4)=15
    from itertools import combinations
    for combo in combinations(range(6), 4):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        initial.append(mask)
        if len(initial) >= m:
            break
    # That should be enough: 19 + 15 = 34, need 6 more
    # Add size-3 subsets of {0..5}: C(6,3)=20, take 6
    count = 0
    for combo in combinations(range(6), 3):
        if count >= 6:
            break
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if mask not in initial:
            initial.append(mask)
            count += 1
    
    # Deduplicate and ensure exactly 40
    seen = set()
    selected = []
    for mask in initial:
        if mask not in seen:
            seen.add(mask)
            selected.append(mask)
    if len(selected) < m:
        # Fill with random masks
        import random
        rng = random.Random(42)
        while len(selected) < m:
            mask = rng.randint(0, total - 1)
            if mask not in seen:
                seen.add(mask)
                selected.append(mask)
    selected = selected[:m]
    
    sel_set = set(selected)
    sel_arr = np.array(selected, dtype=np.int32)
    
    def score_of(sel_list):
        s = 0
        for idx in range(len(sel_list)):
            for jdx in range(idx + 1, len(sel_list)):
                a = sel_list[idx]
                b = sel_list[jdx]
                if (a & b) == a or (a & b) == b:
                    s += 1
        return s
    
    current = score_of(selected)
    best = current
    best_sel = list(selected)
    
    # Precompute for each mask in selected, which other selected masks it's comparable with
    def compute_comparable_in_sel(sel_set_local, sel_list_local):
        # Returns a list of lists: comp_in_sel[i] = list of indices j where sel[i] and sel[j] are comparable
        comp_list = [[] for _ in range(len(sel_list_local))]
        for i in range(len(sel_list_local)):
            for j in range(i + 1, len(sel_list_local)):
                a = sel_list_local[i]
                b = sel_list_local[j]
                if (a & b) == a or (a & b) == b:
                    comp_list[i].append(j)
                    comp_list[j].append(i)
        return comp_list
    
    # 1-swap with numpy acceleration
    # For a given new_mask, gain = number of selected masks (excluding the one being replaced) that are comparable with new_mask
    def compute_gain(new_mask, sel_set_local, exclude_mask):
        gain = 0
        for y in sel_set_local:
            if y == exclude_mask:
                continue
            if (new_mask & y) == new_mask or (new_mask & y) == y:
                gain += 1
        return gain
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # 1-swap pass
        for idx in range(m):
            if time.time() >= deadline:
                break
            old = selected[idx]
            # Compute loss: how many selected masks are comparable with old (excluding itself)
            loss = 0
            for y in sel_set:
                if y != old and ((old & y) == old or (old & y) == y):
                    loss += 1
            
            # Find best new mask
            best_gain = -1
            best_new = -1
            for new in range(total):
                if new in sel_set:
                    continue
                gain = 0
                for y in sel_set:
                    if y != old and ((new & y) == new or (new & y) == y):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_new = new
            
            if best_new >= 0 and best_gain > loss:
                sel_set.remove(old)
                sel_set.add(best_new)
                selected[idx] = best_new
                current += best_gain - loss
                if current > best:
                    best = current
                    best_sel = list(selected)
                improved = True
    
    # Convert to lists
    result = []
    for mask in best_sel:
        result.append([int(b) for b in range(n) if (mask >> b) & 1])
    return result

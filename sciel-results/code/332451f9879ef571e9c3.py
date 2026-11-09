def construct():
    from itertools import combinations
    import time
    import random
    import numpy as np

    start_time = time.time()
    deadline = start_time + 85

    all_sets = list(combinations(range(1, 17), 4))
    n_total = len(all_sets)

    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m

    masks = np.array([to_mask(s) for s in all_sets], dtype=np.int64)
    
    # Precompute intersections
    inter = np.zeros((n_total, n_total), dtype=np.int64)
    for i in range(n_total):
        inter[i, i+1:] = masks[i] & masks[i+1:]
        inter[i+1:, i] = inter[i, i+1:]

    # Initial family: all sets intersecting {1, 2}
    # Indices of sets containing 1 or 2
    init_indices = []
    for i in range(n_total):
        if (masks[i] & (1 << 0)) or (masks[i] & (1 << 1)):
            init_indices.append(i)
    
    family = set(init_indices)
    
    # Check for sunflowers and remove
    # To speed up, we can check pairs and see if they form a sunflower with any existing set
    # But we need to remove one set from each sunflower.
    # Greedy: Find a sunflower, remove the set with highest "degree" or random.
    
    # Efficient check:
    # For a pair (a, b), core K = inter[a][b].
    # Find c in family such that inter[a][c] == K and inter[b][c] == K.
    # If found, (a, b, c) is a sunflower.
    
    # We will iterate and remove.
    # To avoid O(N^3), we can maintain a list of potential conflicts.
    
    # Simpler: Randomly sample pairs until we find a sunflower, then remove one.
    # But we need to ensure termination.
    
    # Better: Iterate through all pairs in family.
    # If a sunflower is found, remove the third set.
    # Repeat until no sunflowers.
    
    # This might be slow if we do it naively.
    # Let's try a few iterations.
    
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        fam_list = list(family)
        n_fam = len(fam_list)
        
        # Shuffle to avoid bias
        random.shuffle(fam_list)
        
        for i in range(n_fam):
            if time.time() > deadline - 5:
                break
            a = fam_list[i]
            if a not in family:
                continue
            for j in range(i + 1, n_fam):
                if time.time() > deadline - 5:
                    break
                b = fam_list[j]
                if b not in family:
                    continue
                K = inter[a][b]
                # Find c in family such that inter[a][c] == K and inter[b][c] == K
                # We can check a few random candidates or iterate
                # To be safe, iterate all
                for k in range(n_fam):
                    if k == i or k == j:
                        continue
                    c = fam_list[k]
                    if c not in family:
                        continue
                    if inter[a][c] == K and inter[b][c] == K:
                        # Sunflower found: a, b, c
                        # Remove c
                        family.remove(c)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
                
    return [sorted(all_sets[i]) for i in family]

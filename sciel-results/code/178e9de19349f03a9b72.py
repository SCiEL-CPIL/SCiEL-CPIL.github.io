def construct():
    import itertools
    import time
    import random
    import numpy as np

    start_time = time.time()
    deadline = start_time + 85

    all_sets = list(itertools.combinations(range(1, 17), 4))
    n_total = len(all_sets)
    
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m

    masks = np.array([to_mask(s) for s in all_sets], dtype=np.int64)
    
    # Precompute intersection matrix
    # inter[i][j] = masks[i] & masks[j]
    inter = np.zeros((n_total, n_total), dtype=np.int64)
    for i in range(n_total):
        inter[i, i:] = masks[i] & masks[i:]
        inter[i:, i] = inter[i, i:]

    best_family = []
    
    # Try multiple random orders
    for _ in range(5):
        if time.time() > deadline - 10:
            break
            
        indices = list(range(n_total))
        random.shuffle(indices)
        
        current_family = []
        
        for s in indices:
            # Check if s can be added to current_family
            can_add = True
            n_fam = len(current_family)
            
            # Optimization: If n_fam is large, this loop is slow.
            # But we expect n_fam to stay under 250.
            
            for i in range(n_fam):
                a = current_family[i]
                K = inter[a][s]
                for j in range(i + 1, n_fam):
                    b = current_family[j]
                    if inter[a][b] == K and inter[b][s] == K:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                current_family.append(s)
                
        if len(current_family) > len(best_family):
            best_family = current_family

    return [sorted(all_sets[i]) for i in best_family]

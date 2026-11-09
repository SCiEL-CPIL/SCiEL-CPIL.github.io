"""Optimized construction of large 4-uniform sunflower-free family."""

def construct():
    from itertools import combinations
    import time
    import numpy as np
    from collections import defaultdict

    deadline = time.time() + 85  # Safety margin before 90s kill

    # Generate all 4-subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n_total = len(all_sets)  # 1820
    
    # Precompute 3-cores for each set
    # set_idx -> list of 3-core indices
    # 3-core index can be the bitmask of the 3-set
    three_cores = []
    for s in all_sets:
        cores = []
        for i in range(4):
            core_set = tuple(sorted([x for j, x in enumerate(s) if j != i]))
            # Convert to bitmask for fast comparison
            mask = 0
            for x in core_set:
                mask |= (1 << (x - 1))
            cores.append(mask)
        three_cores.append(cores)
        
    # Convert sets to bitmasks for intersection checks
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m
        
    masks = np.array([to_mask(s) for s in all_sets], dtype=np.int64)

    # Greedy with 3-core constraint check
    # 1. Fast check: 3-core saturation
    # 2. Slow check: Full sunflower check
    
    def check_conflict_full(c_mask, chosen_masks, chosen_indices):
        """Check if c_mask conflicts with any pair in chosen."""
        n = len(chosen_masks)
        if n < 2:
            return False
        
        # I_i = chosen[i] & c_mask
        I = chosen_masks & c_mask
        
        # K_{i,j} = chosen[i] & chosen[j]
        K = chosen_masks[:, None] & chosen_masks[None, :]
        
        # Conflict if K[i,j] == I[i] AND I[j] == I[i]
        SameI = (I[:, None] == I[None, :])
        K_eq_I = (K == I[:, None])
        
        diag = np.eye(n, dtype=bool)
        conflict_mask = K_eq_I & SameI & ~diag
        
        return np.any(conflict_mask)

    # Greedy initialization
    best_size = 0
    best_indices = []
    
    # Try sequential order first
    chosen_indices = []
    chosen_masks_list = []
    core_counts = defaultdict(int)
    
    for idx in range(n_total):
        if time.time() > deadline:
            break
        cm = masks[idx]
        
        # Fast 3-core check
        valid = True
        for core in three_cores[idx]:
            if core_counts[core] >= 2:
                valid = False
                break
                
        if not valid:
            continue
            
        if len(chosen_masks_list) < 2:
            chosen_indices.append(idx)
            chosen_masks_list.append(cm)
            for core in three_cores[idx]:
                core_counts[core] += 1
            continue
        
        chosen_arr = np.array(chosen_masks_list, dtype=np.int64)
        if not check_conflict_full(cm, chosen_arr, chosen_indices):
            chosen_indices.append(idx)
            chosen_masks_list.append(cm)
            for core in three_cores[idx]:
                core_counts[core] += 1

    best_size = len(chosen_indices)
    best_indices = list(chosen_indices)

    # Local Search / Improvement
    # Try to swap or add if possible
    # Given the tight bound, we might not improve much with simple swaps.
    # Let's try a few random restarts with the fast filter.
    
    rng = np.random.default_rng(42)
    restarts = 0
    while time.time() < deadline - 5 and restarts < 50:
        order = rng.permutation(n_total)
        chosen_indices_r = []
        chosen_masks_r = []
        core_counts_r = defaultdict(int)
        
        for idx in order:
            if time.time() > deadline - 2:
                break
            cm = masks[idx]
            
            # Fast 3-core check
            valid = True
            for core in three_cores[idx]:
                if core_counts_r[core] >= 2:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            if len(chosen_masks_r) < 2:
                chosen_indices_r.append(idx)
                chosen_masks_r.append(cm)
                for core in three_cores[idx]:
                    core_counts_r[core] += 1
                continue
            
            chosen_arr = np.array(chosen_masks_r, dtype=np.int64)
            if not check_conflict_full(cm, chosen_arr, chosen_indices_r):
                chosen_indices_r.append(idx)
                chosen_masks_r.append(cm)
                for core in three_cores[idx]:
                    core_counts_r[core] += 1
        
        restarts += 1
        if len(chosen_indices_r) > best_size:
            best_size = len(chosen_indices_r)
            best_indices = list(chosen_indices_r)

    return [sorted(all_sets[i]) for i in best_indices]

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
    
    # Convert to bitmasks
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m
    
    masks = np.array([to_mask(s) for s in all_sets], dtype=np.int64)
    
    # Precompute element frequencies for better ordering
    elem_freq = np.zeros(16, dtype=np.int64)
    for m in masks:
        for e in range(16):
            if m & (1 << e):
                elem_freq[e] += 1
    
    def check_conflict_fast(c_mask, chosen_masks):
        """Check if c_mask conflicts with any pair in chosen_masks.
        
        Conflict: exist i != j such that:
          A_i & A_j == A_i & c  AND  A_j & c == A_i & c
        Group chosen sets by their intersection with c.
        For group G_t = {i : A_i & c == t}, check if any pair i,j in G_t has A_i & A_j == t.
        """
        n = len(chosen_masks)
        if n < 2:
            return False
        
        I = chosen_masks & c_mask  # shape (n,)
        
        # Group indices by their intersection value with c
        # Using np.unique for efficiency
        unique_vals, inverse = np.unique(I, return_inverse=True)
        
        for u_idx, t in enumerate(unique_vals):
            idxs = np.where(inverse == u_idx)[0]
            if len(idxs) < 2:
                continue
            
            sub = chosen_masks[idxs]
            # Compute pairwise intersections
            # sub shape (k,), result (k, k)
            inter = sub[:, None] & sub[None, :]
            # Mask for i < j
            mask = np.triu(np.ones((len(sub), len(sub)), dtype=bool), k=1)
            
            if np.any((inter == t) & mask):
                return True
        
        return False
    
    def greedy_build(order, deadline):
        """Greedy build following given order."""
        chosen_indices = []
        chosen_masks_list = []
        
        for idx in order:
            if time.time() > deadline:
                break
            cm = masks[idx]
            
            if len(chosen_masks_list) < 2:
                chosen_indices.append(idx)
                chosen_masks_list.append(cm)
                continue
            
            chosen_arr = np.array(chosen_masks_list, dtype=np.int64)
            if not check_conflict_fast(cm, chosen_arr):
                chosen_indices.append(idx)
                chosen_masks_list.append(cm)
        
        return chosen_indices, chosen_masks_list
    
    def local_improve(chosen_indices, chosen_masks_list, deadline):
        """Try to improve by swapping: remove one, try to add multiple new ones."""
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            n_chosen = len(chosen_indices)
            chosen_set = set(chosen_indices)
            
            # Try removing each chosen element and see if we can add more
            for remove_pos in range(n_chosen):
                if time.time() > deadline - 2:
                    break
                
                # Remove element at remove_pos
                remaining_indices = chosen_indices[:remove_pos] + chosen_indices[remove_pos+1:]
                remaining_masks = [chosen_masks_list[i] for i in range(n_chosen) if i != remove_pos]
                
                if len(remaining_masks) < 2:
                    continue
                
                remaining_arr = np.array(remaining_masks, dtype=np.int64)
                
                # Try to add elements not in chosen
                added = []
                for idx in range(n_total):
                    if idx in chosen_set:
                        continue
                    if time.time() > deadline - 3:
                        break
                    cm = masks[idx]
                    if not check_conflict_fast(cm, remaining_arr):
                        added.append(idx)
                
                # If we added more than we removed (1), it's an improvement
                if len(added) > 1:
                    new_indices = remaining_indices + added
                    new_masks = remaining_masks + [masks[i] for i in added]
                    chosen_indices = new_indices
                    chosen_masks_list = new_masks
                    improved = True
                    break
        
        return chosen_indices, chosen_masks_list

    def simulated_annealing(initial_indices, initial_masks, deadline):
        """Simulated Annealing to escape local minima."""
        current_indices = list(initial_indices)
        current_masks = list(initial_masks)
        current_size = len(current_indices)
        best_indices = list(current_indices)
        best_size = current_size
        
        selected_set = set(current_indices)
        
        rng = np.random.default_rng(42)
        T = 5.0
        cooling = 0.99
        min_T = 0.1
        
        while time.time() < deadline - 5 and T > min_T:
            # Pick move
            if rng.random() < 0.6: # Bias towards adding
                # Try add
                unselected = [i for i in range(n_total) if i not in selected_set]
                if not unselected: 
                    T *= cooling
                    continue
                idx = unselected[rng.integers(0, len(unselected))]
                cm = masks[idx]
                
                if len(current_masks) < 2 or not check_conflict_fast(cm, np.array(current_masks, dtype=np.int64)):
                    # Accept
                    current_indices.append(idx)
                    current_masks.append(cm)
                    selected_set.add(idx)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_indices = list(current_indices)
            else:
                # Try remove
                if current_size == 0:
                    T *= cooling
                    continue
                remove_pos = rng.integers(0, current_size)
                idx = current_indices[remove_pos]
                
                # Delta E = +1 (size decreases)
                # Prob = exp(-1/T)
                if rng.random() < np.exp(-1.0 / T):
                    # Accept
                    current_indices.pop(remove_pos)
                    current_masks.pop(remove_pos)
                    selected_set.remove(idx)
                    current_size -= 1
            
            T *= cooling
        
        return best_indices, best_size

    # Strategy 1: Order by element frequency (rare elements first)
    # Sets with rarer elements are less likely to conflict
    freq_order = sorted(range(n_total), key=lambda i: sum(1/elem_freq[e] for e in range(16) if masks[i] & (1 << e)))
    
    chosen_indices, chosen_masks_list = greedy_build(freq_order, deadline)
    
    best_size = len(chosen_indices)
    best_indices = list(chosen_indices)
    
    # Local improvement on frequency-ordered result
    if time.time() < deadline - 10:
        ci, cm = local_improve(chosen_indices, chosen_masks_list, deadline)
        if len(ci) > best_size:
            best_size = len(ci)
            best_indices = list(ci)
            
    # Simulated Annealing on the best found so far
    if time.time() < deadline - 15:
        sa_indices, sa_size = simulated_annealing(best_indices, [masks[i] for i in best_indices], deadline)
        if sa_size > best_size:
            best_size = sa_size
            best_indices = list(sa_indices)

    # Strategy 2: Random restarts with different orderings
    rng = np.random.default_rng(42)
    restarts = 0
    while time.time() < deadline - 15 and restarts < 20:
        # Mix of random and frequency-based perturbations
        if restarts % 3 == 0:
            order = rng.permutation(n_total)
        elif restarts % 3 == 1:
            # Perturb frequency order
            order = freq_order.copy()
            for _ in range(100):
                i, j = rng.integers(0, n_total, 2)
                order[i], order[j] = order[j], order[i]
        else:
            # Weighted random: prefer rarer sets
            weights = np.array([sum(1.0/elem_freq[e] for e in range(16) if masks[i] & (1 << e)) for i in range(n_total)])
            weights = weights / weights.sum()
            order = rng.choice(n_total, size=n_total, replace=False, p=weights)
        
        chosen_r, masks_r = greedy_build(order, deadline)
        
        # Local improvement
        if time.time() < deadline - 10 and len(chosen_r) >= best_size - 10:
            ci, cm = local_improve(chosen_r, masks_r, deadline)
            if len(ci) > len(chosen_r):
                chosen_r = ci
                
        # SA on restart result if promising
        if time.time() < deadline - 15 and len(chosen_r) >= best_size - 5:
             sa_r, sa_sz = simulated_annealing(chosen_r, masks_r, deadline)
             if sa_sz > len(chosen_r):
                 chosen_r = sa_r

        restarts += 1
        if len(chosen_r) > best_size:
            best_size = len(chosen_r)
            best_indices = list(chosen_r)
    
    return [sorted(all_sets[i]) for i in best_indices]

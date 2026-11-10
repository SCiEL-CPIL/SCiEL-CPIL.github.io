import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n_max = 512
    size = n_max
    
    # Precompute compatibility matrix
    a = np.arange(1, size + 1, dtype=np.int64)
    gcd_mat = np.gcd(a[:, None], a[None, :])
    lcm_mat = (a[:, None] * a[None, :]) // gcd_mat
    compat = lcm_mat <= n_max  # (512, 512) boolean
    
    def greedy_remove(compat, size, deadline, start_time, active_init=None):
        if active_init is not None:
            active = active_init.copy()
        else:
            active = np.ones(size, dtype=bool)
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            
            sub_compat = compat[np.ix_(active_indices, active_indices)]
            conflicts = (~sub_compat).sum(axis=1)
            
            if conflicts.max() == 0:
                break
            
            max_local = np.argmax(conflicts)
            max_idx = active_indices[max_local]
            active[max_idx] = False
        
        # Add back compatible elements
        for _ in range(20):
            added_any = False
            inactive = np.where(~active)[0]
            if len(inactive) == 0:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                active[inactive] = True
                break
            sub_compat = compat[np.ix_(inactive, active_indices)]
            can_add = sub_compat.all(axis=1)
            if can_add.any():
                active[inactive[can_add]] = True
                added_any = True
            if not added_any:
                break
        
        return active
    
    def local_search(compat, size, active, deadline, start_time):
        current_best_size = int(active.sum())
        
        for iteration in range(1000):
            elapsed = time.time() - start_time
            if elapsed > deadline - 3:
                break
            
            active_indices = np.where(active)[0]
            inactive_indices = np.where(~active)[0]
            
            if len(inactive_indices) == 0:
                break
            
            improved = False
            
            # Try to add any compatible inactive element
            if len(active_indices) > 0:
                sub_compat = compat[np.ix_(inactive_indices, active_indices)]
                can_add = sub_compat.all(axis=1)
                if can_add.any():
                    active[inactive_indices[can_add]] = True
                    current_best_size += int(can_add.sum())
                    improved = True
                    continue
            
            # Try swaps
            if len(active_indices) > 0:
                sub_compat = compat[np.ix_(inactive_indices, active_indices)]
                conflict_mask = ~sub_compat
                conflict_counts = conflict_mask.sum(axis=1)
                
                # Try inactive elements with fewest conflicts
                order = np.argsort(conflict_counts)
                
                for idx in order:
                    elapsed = time.time() - start_time
                    if elapsed > deadline - 3:
                        break
                    
                    r = inactive_indices[idx]
                    n_conflicts = conflict_counts[idx]
                    
                    if n_conflicts == 0:
                        active[r] = True
                        current_best_size += 1
                        improved = True
                        break
                    
                    if n_conflicts >= current_best_size:
                        continue
                    
                    conflicting = active_indices[conflict_mask[idx]]
                    
                    saved_active = active.copy()
                    for c in conflicting:
                        active[c] = False
                    
                    active[r] = True
                    temp_size = current_best_size - len(conflicting) + 1
                    
                    remaining_inactive = inactive_indices[(inactive_indices != r) & (~active)]
                    if len(remaining_inactive) > 0:
                        active_indices_now = np.where(active)[0]
                        if len(active_indices_now) > 0:
                            sub_compat2 = compat[np.ix_(remaining_inactive, active_indices_now)]
                            can_add2 = sub_compat2.all(axis=1)
                            if can_add2.any():
                                active[remaining_inactive[can_add2]] = True
                                temp_size += int(can_add2.sum())
                    
                    if temp_size > current_best_size:
                        current_best_size = temp_size
                        improved = True
                        break
                    else:
                        active = saved_active.copy()
            
            if not improved:
                break
        
        return active, current_best_size
    
    # Initial greedy
    active = greedy_remove(compat, size, deadline, start_time)
    active, current_best_size = local_search(compat, size, active, deadline, start_time)
    
    # Randomized restarts
    import random
    for restart in range(20):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        random.seed(restart * 7919)
        active2 = np.ones(size, dtype=bool)
        
        # Randomized greedy removal
        while True:
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            active_indices2 = np.where(active2)[0]
            if len(active_indices2) == 0:
                break
            
            sub_compat2 = compat[np.ix_(active_indices2, active_indices2)]
            conflicts2 = (~sub_compat2).sum(axis=1)
            
            if conflicts2.max() == 0:
                break
            
            max_conf = conflicts2.max()
            candidates = active_indices2[conflicts2 == max_conf]
            chosen = random.choice(candidates.tolist())
            active2[chosen] = False
        
        # Add back
        for _ in range(20):
            added_any = False
            inactive2 = np.where(~active2)[0]
            if len(inactive2) == 0:
                break
            active_indices2 = np.where(active2)[0]
            if len(active_indices2) == 0:
                active2[inactive2] = True
                break
            sub_compat2 = compat[np.ix_(inactive2, active_indices2)]
            can_add2 = sub_compat2.all(axis=1)
            if can_add2.any():
                active2[inactive2[can_add2]] = True
                added_any = True
            if not added_any:
                break
        
        active2, size2 = local_search(compat, size, active2, deadline, start_time)
        
        if size2 > current_best_size:
            active = active2
            current_best_size = size2
    
    result = sorted([int(i) + 1 for i in np.where(active)[0]])
    return result

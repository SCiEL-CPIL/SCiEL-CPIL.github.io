import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    n = 256
    # 1. Precompute conflict bitsets
    conflict = [0] * n
    
    for i in range(n):
        a = i + 1
        bits = 0
        for j in range(n):
            if i == j: continue
            b = j + 1
            s = a + b
            p = a * b
            if p % s == 0:
                bits |= (1 << j)
        conflict[i] = bits
        
    # Fast size computation using precomputed table for 16-bit chunks
    popcount_table = [0] * 65536
    for i in range(1, 65536):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)
    
    def get_size(m):
        return (popcount_table[m & 0xFFFF] + popcount_table[(m >> 16) & 0xFFFF] +
                popcount_table[(m >> 32) & 0xFFFF] + popcount_table[(m >> 48) & 0xFFFF])
    
    def greedy_add(mask):
        improved = True
        while improved:
            improved = False
            for v in range(n):
                if not (mask & (1 << v)):
                    if not (mask & conflict[v]):
                        mask |= (1 << v)
                        improved = True
                        break
        return mask
    
    def local_search(mask):
        current_size = get_size(mask)
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            
            # Step A: Greedy Add
            new_mask = greedy_add(mask)
            if new_mask != mask:
                mask = new_mask
                current_size = get_size(mask)
                improved = True
                continue
            
            # Step B: 1-Exchange
            found_swap = False
            for v in range(n):
                if mask & (1 << v):
                    continue
                confs = mask & conflict[v]
                if confs and (confs & (confs - 1)) == 0:
                    u_bit = confs
                    new_mask = (mask & ~u_bit) | (1 << v)
                    test_mask = greedy_add(new_mask)
                    if get_size(test_mask) > current_size:
                        mask = test_mask
                        current_size = get_size(mask)
                        improved = True
                        found_swap = True
                        break
            
            if found_swap:
                continue
            
            # Step C: 2-Exchange
            if (time.time() - start_time) < deadline - 5.0:
                found_2swap = False
                for v1 in range(n):
                    if mask & (1 << v1):
                        continue
                    confs1 = mask & conflict[v1]
                    if get_size(confs1) > 2:
                        continue
                    for v2 in range(v1 + 1, n):
                        if mask & (1 << v2):
                            continue
                        confs2 = mask & conflict[v2]
                        if get_size(confs2) > 2:
                            continue
                        
                        union_confs = confs1 | confs2
                        if get_size(union_confs) > 2:
                            continue
                        
                        new_mask = mask
                        new_mask &= ~union_confs
                        new_mask |= (1 << v1)
                        new_mask |= (1 << v2)
                        
                        test_mask = greedy_add(new_mask)
                        if get_size(test_mask) > current_size:
                            mask = test_mask
                            current_size = get_size(mask)
                            improved = True
                            found_2swap = True
                            break
                    if found_2swap:
                        break
            
            if found_2swap:
                continue
            
            # Step D: 3-Exchange (remove 3, add 3)
            if (time.time() - start_time) < deadline - 3.0:
                found_3swap = False
                # Only try if we have enough time
                candidates = [v for v in range(n) if not (mask & (1 << v))]
                for v1 in candidates:
                    confs1 = mask & conflict[v1]
                    if get_size(confs1) > 3:
                        continue
                    for v2 in candidates:
                        if v2 <= v1:
                            continue
                        confs2 = mask & conflict[v2]
                        if get_size(confs2) > 3:
                            continue
                        for v3 in candidates:
                            if v3 <= v2:
                                continue
                            confs3 = mask & conflict[v3]
                            if get_size(confs3) > 3:
                                continue
                            
                            union_confs = confs1 | confs2 | confs3
                            if get_size(union_confs) > 3:
                                continue
                            
                            new_mask = mask
                            new_mask &= ~union_confs
                            new_mask |= (1 << v1)
                            new_mask |= (1 << v2)
                            new_mask |= (1 << v3)
                            
                            test_mask = greedy_add(new_mask)
                            if get_size(test_mask) > current_size:
                                mask = test_mask
                                current_size = get_size(mask)
                                improved = True
                                found_3swap = True
                                break
                        if found_3swap:
                            break
                    if found_3swap:
                        break
            
            if found_3swap:
                continue
            
            # No improvement found in this pass - break to allow multi-start
            break
        
        return mask
    
    # Multi-start strategy with random perturbations
    best_mask = 0
    best_size = 0
    
    # Start 1: All odds
    mask = 0
    for i in range(0, n, 2):
        mask |= (1 << i)
    
    result = local_search(mask)
    best_mask = result
    best_size = get_size(result)
    
    # Additional starts with random perturbations
    max_starts = 20
    for s in range(1, max_starts):
        if (time.time() - start_time) > deadline - 5.0:
            break
        
        # Create perturbed version of best_mask
        perturbed_mask = best_mask
        indices = [i for i in range(n) if perturbed_mask & (1 << i)]
        if len(indices) > 3:
            random.shuffle(indices)
            num_remove = min(3, len(indices))
            for i in indices[:num_remove]:
                perturbed_mask &= ~(1 << i)
        
        result = local_search(perturbed_mask)
        sz = get_size(result)
        if sz > best_size:
            best_mask = result
            best_size = sz
    
    # Convert mask to list
    result_list = []
    for i in range(n):
        if best_mask & (1 << i):
            result_list.append(i + 1)
            
    return result_list

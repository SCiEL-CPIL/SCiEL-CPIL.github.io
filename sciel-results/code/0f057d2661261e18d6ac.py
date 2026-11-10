import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free set modulo 205.
    Uses bitmask-accelerated local search with multiple random starts.
    """
    m = 205
    deadline = time.time() + 580.0  # Safety margin for 600s limit
    
    # 1. Compute forbidden differences (non-zero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build conflict masks
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def get_conflicts(selected_mask, candidate):
        """
        Returns True if candidate conflicts with any node in selected_mask.
        """
        return (selected_mask & conflict_mask[candidate]) != 0

    def greedy_random(seed):
        """
        Generate a random greedy solution.
        """
        random.seed(seed)
        selected_mask = 0
        candidates = list(range(m))
        random.shuffle(candidates)
        
        for c in candidates:
            if not get_conflicts(selected_mask, c):
                selected_mask |= (1 << c)
                
        return selected_mask

    def local_search(start_mask):
        """
        Improve a solution using 1-add, 2-swap, 3-swap, and 1-swap moves.
        Uses hill-climbing with random restarts/shakes if stuck.
        """
        current_mask = start_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        
        no_improvement_count = 0
        max_iterations = 30000
        shake_count = 0
        
        while no_improvement_count < 80 and max_iterations > 0 and time.time() < deadline:
            max_iterations -= 1
            improved = False
            
            # 1. Try 1-Add: Add a node that doesn't conflict
            for c in range(m):
                if not (current_mask & (1 << c)):
                    if not get_conflicts(current_mask, c):
                        current_mask |= (1 << c)
                        current_size += 1
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        break
            
            if improved:
                no_improvement_count = 0
                continue
            
            # 2. Try 2-Swap: Remove one node i, add two nodes j, k (net +1)
            current_indices = [i for i in range(m) if current_mask & (1 << i)]
            
            found_2swap = False
            for i in current_indices:
                mask_without_i = current_mask & ~(1 << i)
                
                for j in range(m):
                    if current_mask & (1 << j):
                        continue
                    if mask_without_i & conflict_mask[j]:
                        continue
                    
                    for k in range(j + 1, m):
                        if current_mask & (1 << k):
                            continue
                        if mask_without_i & conflict_mask[k]:
                            continue
                        if (1 << j) & conflict_mask[k]:
                            continue
                        
                        new_mask = mask_without_i | (1 << j) | (1 << k)
                        current_mask = new_mask
                        current_size += 1
                        improved = True
                        found_2swap = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        break
                    if found_2swap:
                        break
                if found_2swap:
                    break
            
            if found_2swap:
                no_improvement_count = 0
                continue

            # 3. Try 3-Swap: Remove two nodes i, l, add three nodes j, k, n (net +1)
            found_3swap = False
            n_idx = len(current_indices)
            
            for ii in range(n_idx):
                i = current_indices[ii]
                for ll in range(ii + 1, n_idx):
                    l = current_indices[ll]
                    mask_without_il = current_mask & ~(1 << i) & ~(1 << l)
                    
                    # Find three candidates j, k, n that are compatible with mask_without_il and each other
                    for j in range(m):
                        if current_mask & (1 << j):
                            continue
                        if mask_without_il & conflict_mask[j]:
                            continue
                        
                        for k in range(j + 1, m):
                            if current_mask & (1 << k):
                                continue
                            if mask_without_il & conflict_mask[k]:
                                continue
                            if (1 << j) & conflict_mask[k]:
                                continue
                            
                            for n in range(k + 1, m):
                                if current_mask & (1 << n):
                                    continue
                                if mask_without_il & conflict_mask[n]:
                                    continue
                                if (1 << j) & conflict_mask[n]:
                                    continue
                                if (1 << k) & conflict_mask[n]:
                                    continue
                                
                                # Found valid 3-swap
                                new_mask = mask_without_il | (1 << j) | (1 << k) | (1 << n)
                                current_mask = new_mask
                                current_size += 1
                                improved = True
                                found_3swap = True
                                if current_size > best_size:
                                    best_size = current_size
                                    best_mask = current_mask
                                break
                            if found_3swap:
                                break
                        if found_3swap:
                            break
                    if found_3swap:
                        break
                if found_3swap:
                    break
            
            if found_3swap:
                no_improvement_count = 0
                continue

            # 4. Try 1-Swap: Remove one node i, add one node j (size stays same)
            # This helps escape local optima.
            if current_indices:
                i = random.choice(current_indices)
                mask_without_i = current_mask & ~(1 << i)
                
                valid_js = []
                for j in range(m):
                    if current_mask & (1 << j):
                        continue
                    if not (mask_without_i & conflict_mask[j]):
                        valid_js.append(j)
                
                if valid_js:
                    j = random.choice(valid_js)
                    current_mask = mask_without_i | (1 << j)
                    no_improvement_count = 0
                    continue
            
            # If no moves found, we are stuck.
            no_improvement_count += 1
            
            # If stuck for a long time, perform a perturbation
            if no_improvement_count > 40:
                # Perturb: remove 2-4 random nodes
                num_remove = random.randint(2, 4)
                if current_size > num_remove:
                    indices_to_remove = random.sample([i for i in range(m) if current_mask & (1 << i)], num_remove)
                    for idx in indices_to_remove:
                        current_mask &= ~(1 << idx)
                    current_size = bin(current_mask).count('1')
                    no_improvement_count = 0
                    shake_count += 1
                    if shake_count > 15:
                        break
                
        return best_mask, best_size

    # Main Search Loop
    best_global_mask = 0
    best_global_size = 0
    
    num_starts = 80
    start_time = time.time()
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        seed = i * 12345 + 67890
        init_mask = greedy_random(seed)
        
        improved_mask, improved_size = local_search(init_mask)
        
        if improved_size > best_global_size:
            best_global_size = improved_size
            best_global_mask = improved_mask

    # Convert best mask to array
    result = []
    for i in range(m):
        if best_global_mask & (1 << i):
            result.append(i)
            
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END

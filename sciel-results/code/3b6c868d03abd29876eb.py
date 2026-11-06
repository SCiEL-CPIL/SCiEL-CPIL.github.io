import time
import random
from math import lcm

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s, task says 60s.
    # We will stop searching at 50s to ensure return.
    deadline = start_time + 50.0
    
    nums = list(range(2, 257))
    n = len(nums)
    weights = [1.0 / x for x in nums]
    
    # Precompute conflict masks
    # mask[i] has bit j set if nums[i] conflicts with nums[j]
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(nums[i], nums[j]) <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    def get_set_from_mask(mask):
        return [nums[i] for i in range(n) if (mask >> i) & 1]
        
    def get_score(mask):
        return sum(weights[i] for i in range(n) if (mask >> i) & 1)
        
    def greedy_weight_order():
        indices = list(range(n))
        indices.sort(key=lambda i: -weights[i])
        mask = 0
        for i in indices:
            if (conflict_mask[i] & mask) == 0:
                mask |= (1 << i)
        return mask

    def greedy_random_order():
        indices = list(range(n))
        random.shuffle(indices)
        mask = 0
        for i in indices:
            if (conflict_mask[i] & mask) == 0:
                mask |= (1 << i)
        return mask

    def local_search_1swap(mask):
        improved = True
        # Limit iterations to avoid infinite loops or excessive time
        max_iters = 100
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            u_indices = [i for i in range(n) if (mask >> i) & 1]
            # Shuffle u_indices to avoid deterministic bias
            random.shuffle(u_indices)
            for u in u_indices:
                mask_without_u = mask ^ (1 << u)
                best_gain = 0
                best_v = -1
                for v in range(n):
                    if (mask >> v) & 1:
                        continue
                    if weights[v] <= weights[u]:
                        continue
                    if (conflict_mask[v] & mask_without_u) == 0:
                        gain = weights[v] - weights[u]
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                if best_v != -1:
                    mask = mask_without_u | (1 << best_v)
                    improved = True
                    break
        return mask

    def local_search_2swap(mask):
        improved = True
        max_iters = 50
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            u_indices = [i for i in range(n) if (mask >> i) & 1]
            random.shuffle(u_indices)
            # Iterate pairs
            for i_idx in range(len(u_indices)):
                u = u_indices[i_idx]
                for w in u_indices[i_idx+1:]:
                    mask_without_uw = mask ^ (1 << u) ^ (1 << w)
                    base_weight = weights[u] + weights[w]
                    
                    # Find candidates compatible with mask_without_uw
                    candidates = []
                    for v in range(n):
                        if (mask >> v) & 1:
                            continue
                        if (conflict_mask[v] & mask_without_uw) == 0:
                            candidates.append(v)
                            
                    if len(candidates) < 2:
                        continue
                    
                    # Find best pair v, z
                    # Sort candidates by weight descending to find good pairs faster
                    candidates.sort(key=lambda x: -weights[x])
                    
                    best_gain = 0
                    best_pair = None
                    # Check top candidates first
                    # If candidates list is long, we might miss optimal pair if we truncate, 
                    # but N=255 is small enough to check all pairs in candidates
                    # However, to be safe on time, we can limit checks if candidates is huge
                    # But usually candidates is small.
                    for k_idx, v in enumerate(candidates):
                        for z in candidates[k_idx+1:]:
                            if (conflict_mask[v] & (1 << z)) == 0:
                                gain = (weights[v] + weights[z]) - base_weight
                                if gain > best_gain:
                                    best_gain = gain
                                    best_pair = (v, z)
                    
                    if best_pair:
                        v, z = best_pair
                        mask = mask_without_uw | (1 << v) | (1 << z)
                        improved = True
                        break
                if improved:
                    break
        return mask

    best_mask = 0
    best_score = 0
    
    # Run multiple restarts
    # 20 weight-ordered, 80 random-ordered
    num_weight_restarts = 20
    num_random_restarts = 80
    
    for r in range(num_weight_restarts):
        if time.time() > deadline:
            break
        mask = greedy_weight_order()
        mask = local_search_1swap(mask)
        mask = local_search_2swap(mask)
        score = get_score(mask)
        if score > best_score:
            best_score = score
            best_mask = mask
            
    for r in range(num_random_restarts):
        if time.time() > deadline:
            break
        mask = greedy_random_order()
        mask = local_search_1swap(mask)
        mask = local_search_2swap(mask)
        score = get_score(mask)
        if score > best_score:
            best_score = score
            best_mask = mask
            
    # Final check: if best_mask is 0 (should not happen), return a single valid node
    if best_mask == 0:
        # Pick the smallest number 2
        best_mask = 1 << 0 # Index 0 is 2
        
    return get_set_from_mask(best_mask)

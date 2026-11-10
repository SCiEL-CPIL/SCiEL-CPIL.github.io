import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability matrix using NumPy for speed
    # comp[i, j] = 1 if mask i and mask j are comparable (one is subset of other), 0 otherwise
    # We only need upper triangle for efficiency, but full matrix for easy indexing
    masks = np.arange(num_masks, dtype=np.int32)
    # For each pair, check if (a & b) == a or (a & b) == b
    # Vectorized: for each i, compute (masks[i] & masks) and check conditions
    comp = np.zeros((num_masks, num_masks), dtype=np.int8)
    for i in range(num_masks):
        mi = masks[i]
        # (mi & masks) == mi means mi is subset of masks
        # (mi & masks) == masks means masks is subset of mi
        and_vals = mi & masks
        comp[i, :] = ((and_vals == mi) | (and_vals == masks)).astype(np.int8)
        comp[i, i] = 0  # Not comparable with self
    
    # Precompute for each mask, the list of comparable masks (for faster delta calc)
    # comp_list[i] = list of j where comp[i,j] == 1
    comp_list = [np.where(comp[i] == 1)[0].tolist() for i in range(num_masks)]
    
    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                a, b = masks[i], masks[j]
                if (a & b) == a or (a & b) == b:
                    s += 1
        return s
    
    # Build initial solution: use a down-set (ball) centered at a random mask
    random.seed(42)
    
    # Start with empty set and build up greedily
    current_masks = []
    used_masks = set()
    
    # Add empty set first (comparable with everything)
    current_masks.append(0)
    used_masks.add(0)
    
    # Greedily add masks that maximize comparability with existing set
    for _ in range(m - 1):
        best_mask = -1
        best_gain = -1
        for mask in range(1, num_masks):
            if mask in used_masks:
                continue
            gain = 0
            for cm in current_masks:
                if (mask & cm) == mask or (mask & cm) == cm:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        if best_mask != -1:
            current_masks.append(best_mask)
            used_masks.add(best_mask)
    
    best_score = calc_score(current_masks)
    best_solution = list(current_masks)
    
    # MCMC with Metropolis-Hastings
    # Temperature schedule
    T = 5.0
    cooling_rate = 0.99995
    min_T = 0.001
    
    restarts = 0
    max_restarts = 15
    
    while time.time() < deadline and restarts < max_restarts:
        current_masks = list(best_solution)
        used_masks = set(current_masks)
        current_score = best_score
        
        T = 5.0
        no_improve_steps = 0
        max_no_improve = 20000
        
        while time.time() < deadline and T > min_T:
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate new mask with rank-aware moves
            r = random.random()
            if r < 0.25:
                # Rank-preserving swap: replace with another mask of same cardinality
                popcount = bin(old_mask).count('1')
                candidates = [mask for mask in range(num_masks) 
                             if mask not in used_masks and bin(mask).count('1') == popcount]
                if candidates:
                    new_mask = random.choice(candidates)
                else:
                    new_mask = old_mask
            elif r < 0.5:
                # Rank-changing: move to a subset (remove 1-3 bits)
                bits = [i for i in range(n) if (old_mask >> i) & 1]
                if bits:
                    k = random.randint(1, min(3, len(bits)))
                    remove_bits = random.sample(bits, k)
                    new_mask = old_mask
                    for b in remove_bits:
                        new_mask ^= (1 << b)
                else:
                    new_mask = old_mask
            elif r < 0.75:
                # Rank-changing: move to a superset (add 1-3 bits)
                bits = [i for i in range(n) if not ((old_mask >> i) & 1)]
                if bits:
                    k = random.randint(1, min(3, len(bits)))
                    add_bits = random.sample(bits, k)
                    new_mask = old_mask
                    for b in add_bits:
                        new_mask |= (1 << b)
                else:
                    new_mask = old_mask
            else:
                # Random mask
                new_mask = random.randint(0, full_mask)
            
            if new_mask == old_mask or new_mask in used_masks:
                continue

            # Calculate delta using precomputed comparability
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                cm = current_masks[j]
                old_comp = 1 if ((old_mask & cm) == old_mask or (old_mask & cm) == cm) else 0
                new_comp = 1 if ((new_mask & cm) == new_mask or (new_mask & cm) == cm) else 0
                delta += new_comp - old_comp
            
            # Acceptance
            if delta > 0:
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                current_score += delta
                no_improve_steps = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
            else:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    used_masks.remove(old_mask)
                    used_masks.add(new_mask)
                    current_masks[idx] = new_mask
                    current_score += delta
                    no_improve_steps += 1
                else:
                    no_improve_steps += 1
            
            T *= cooling_rate
            
            if no_improve_steps > max_no_improve:
                break

        restarts += 1

    # Final polish: try all single-swap moves
    for _ in range(30):
        if time.time() > deadline - 3:
            break
        improved = False
        for i in range(m):
            if time.time() > deadline - 3:
                break
            old_mask = best_solution[i]
            best_new_mask = old_mask
            best_delta = 0
            for new_mask in range(num_masks):
                if new_mask == old_mask:
                    continue
                if new_mask in set(best_solution):
                    continue
                delta = 0
                for j in range(m):
                    if j == i:
                        continue
                    cm = best_solution[j]
                    old_comp = 1 if ((old_mask & cm) == old_mask or (old_mask & cm) == cm) else 0
                    new_comp = 1 if ((new_mask & cm) == new_mask or (new_mask & cm) == cm) else 0
                    delta += new_comp - old_comp
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            
            if best_delta > 0:
                best_solution[i] = best_new_mask
                best_score += best_delta
                improved = True
        
        if not improved:
            break

    # Convert to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result

import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    
    # Helper to check strict subset
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    # Precompute popcount for all masks
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Group masks by size for efficient candidate generation
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        masks_by_size[popcount[mask]].append(mask)
        
    # Helper to calculate score of a set of masks
    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if is_comparable(masks[i], masks[j]):
                    s += 1
        return s

    # Helper to calculate comparability count of a mask with a list of masks
    def get_comparability_with_others(mask, masks_list, exclude_idx):
        count = 0
        for j in range(len(masks_list)):
            if j == exclude_idx:
                continue
            if is_comparable(mask, masks_list[j]):
                count += 1
        return count

    # Initialization: Use a B_6 sub-lattice
    # Vary bits 0-5, fix bits 6-11 to 0
    base_masks = []
    for mask in range(1 << 6):
        base_masks.append(mask)
        
    # We need 40 masks. 
    # Let's pick the 40 masks from base_masks that form the most comparable set.
    # We can just start with a random subset of 40 from base_masks and refine.
    # Or, better, start with all 64 and remove the 24 with lowest "value".
    
    # Let's try a greedy initialization on the B_6 sub-lattice
    # Start with empty set of selected masks
    current_masks = []
    used_masks = set()
    
    # To maximize comparability, we should pick masks that are comparable with many others.
    # In B_6, masks of size 3 are comparable with many.
    # Let's just pick 40 random masks from the B_6 sub-lattice to start, then SA.
    random.seed(42)
    candidates = list(base_masks)
    random.shuffle(candidates)
    current_masks = candidates[:40]
    used_masks = set(current_masks)
    
    best_score = calc_score(current_masks)
    best_solution = list(current_masks)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.9995
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if time.time() > deadline - 2.0:
            break
            
        # Pick a random index to modify
        idx = random.randint(0, m - 1)
        old_mask = current_masks[idx]
        
        # Generate candidate
        # Bias towards masks from the B_6 sub-lattice or similar structures
        # 50% chance: pick from B_6 sub-lattice
        # 50% chance: pick from full space
        if random.random() < 0.5:
            new_mask = random.choice(base_masks)
        else:
            # Pick a size biased towards middle sizes (3, 4, 5, 6, 7)
            # Or just random
            new_mask = random.randint(0, full_mask)
            
        if new_mask == old_mask or new_mask in used_masks:
            continue

        # Calculate delta
        old_comp_count = get_comparability_with_others(old_mask, current_masks, idx)
        new_comp_count = get_comparability_with_others(new_mask, current_masks, idx)
        delta = new_comp_count - old_comp_count
        
        # Simulated annealing acceptance
        if delta >= 0:
            # Accept
            used_masks.remove(old_mask)
            used_masks.add(new_mask)
            current_masks[idx] = new_mask
            best_score += delta
            best_solution = list(current_masks)
            no_improve_count = 0
        else:
            # Accept with probability exp(delta / T)
            if random.random() < math.exp(delta / T):
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                best_score += delta
                best_solution = list(current_masks)
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        # If stuck, restart with higher temperature
        if no_improve_count > max_no_improve:
            T = 5.0
            no_improve_count = 0
            
    # Final polish: Hill Climbing
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for i in range(m):
            if time.time() > deadline - 1.0:
                break
            old_mask = current_masks[i]
            old_comp_count = get_comparability_with_others(old_mask, current_masks, i)
            
            # Try a few random replacements
            for _ in range(10):
                # Try from B_6
                if random.random() < 0.5:
                    new_mask = random.choice(base_masks)
                else:
                    new_mask = random.randint(0, full_mask)
                    
                if new_mask == old_mask or new_mask in used_masks:
                    continue
                    
                new_comp_count = get_comparability_with_others(new_mask, current_masks, i)
                delta = new_comp_count - old_comp_count
                
                if delta > 0:
                    used_masks.remove(old_mask)
                    used_masks.add(new_mask)
                    current_masks[i] = new_mask
                    best_score += delta
                    best_solution = list(current_masks)
                    improved = True
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

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    n_elements = 12
    universe_size = 1 << n_elements  # 4096
    family_size = 40
    
    all_masks = list(range(universe_size))
    
    def count_comparable(family):
        count = 0
        n = len(family)
        for i in range(n):
            a = family[i]
            for j in range(i + 1, n):
                b = family[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count

    def gain_of_adding(mask, family):
        """Calculate how many pairs involving 'mask' and existing 'family' are comparable."""
        gain = 0
        for existing in family:
            if (mask & existing) == mask or (mask & existing) == existing:
                gain += 1
        return gain

    # Phase 1: Greedy Construction
    random.seed(42)
    
    # Start with a random set to break symmetry
    current_family = [random.choice(all_masks)]
    
    available_masks = set(all_masks)
    available_masks.discard(current_family[0])
    
    for _ in range(family_size - 1):
        best_mask = -1
        best_gain = -1
        best_size = 100
        
        # Iterate through available masks to find the one with max gain
        for mask in available_masks:
            gain = gain_of_adding(mask, current_family)
            # Tie-breaker: prefer smaller size to keep chain potential high
            size = mask.bit_count()
            
            if gain > best_gain or (gain == best_gain and size < best_size):
                best_gain = gain
                best_mask = mask
                best_size = size
                
        if best_mask != -1:
            current_family.append(best_mask)
            available_masks.discard(best_mask)
            
    # Phase 2: Local Search (Hill Climbing) to improve the greedy result
    current_score = count_comparable(current_family)
    best_family = current_family.copy()
    best_score = current_score
    
    start_time = time.time()
    timeout = 50
    
    while time.time() - start_time < timeout:
        idx = random.randint(0, family_size - 1)
        old_set = current_family[idx]
        
        # Generate a candidate neighbor
        if random.random() < 0.5:
            new_set = random.choice(all_masks)
        else:
            bit_to_flip = random.randint(0, n_elements - 1)
            new_set = old_set ^ (1 << bit_to_flip)
            
        if new_set == old_set:
            continue
            
        is_dup = False
        for k in range(family_size):
            if k != idx and current_family[k] == new_set:
                is_dup = True
                break
        
        if is_dup:
            continue
            
        candidate_family = current_family.copy()
        candidate_family[idx] = new_set
        candidate_score = count_comparable(candidate_family)
        
        if candidate_score > current_score:
            current_family = candidate_family
            current_score = candidate_score
            
            if current_score > best_score:
                best_score = current_score
                best_family = current_family.copy()

    result = []
    for mask in best_family:
        subset = []
        for i in range(n_elements):
            if mask & (1 << i):
                subset.append(i)
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END

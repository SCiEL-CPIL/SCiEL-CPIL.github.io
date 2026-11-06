import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Generate all 4-element subsets of [1, 16] as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    
    best_family = []
    best_size = 0
    
    def is_valid_addition(family, new_mask):
        """
        Check if adding new_mask to family creates a sunflower.
        Condition: No A, B in family such that A & B == A & new_mask == B & new_mask.
        """
        if not family:
            return True
            
        # Group existing sets by their intersection with new_mask
        # If two sets A, B have the same intersection I with new_mask,
        # we must check if A & B == I.
        groups = {}
        
        for a in family:
            inter = a & new_mask
            if inter in groups:
                # Check if the new set 'a' and any existing set 'b' in this group
                # have intersection equal to 'inter'
                for b in groups[inter]:
                    if (a & b) == inter:
                        return False
                groups[inter].append(a)
            else:
                groups[inter] = [a]
                
        return True

    rng = random.Random(42)
    
    # Perform multiple greedy passes
    # The is_valid_addition is now O(N * K) where K is avg group size, which is small.
    # This allows many passes.
    
    for iteration in range(2000): # Increased iterations possible due to speed
        if time.time() > deadline - 5:
            break
        
        shuffled = list(range(n_total))
        rng.shuffle(shuffled)
        
        family = []
        for idx in shuffled:
            m = all_masks[idx]
            if is_valid_addition(family, m):
                family.append(m)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
        
        if best_size >= 250:
            break

    # Optional: Local search to try to improve best_family
    # If we have time, try to add any remaining mask to best_family
    if time.time() < deadline - 2 and best_family:
        family = list(best_family)
        family_set = set(family)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for idx in range(n_total):
                if time.time() > deadline - 1:
                    break
                m = all_masks[idx]
                if m in family_set:
                    continue
                if is_valid_addition(family, m):
                    family.append(m)
                    family_set.add(m)
                    improved = True
        
        if len(family) > best_size:
            best_family = family
            best_size = len(family)

    # Convert bitmasks back to sorted lists of 1-indexed elements
    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))
    
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))

import random
import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_masks.append(mask)
    
    # Helper to check if a new mask can be added to a family
    # family is a list of masks
    def can_add(new_mask, family):
        # If family is empty or size 1, always valid
        if len(family) < 2:
            return True
            
        # For each pair A, B in family, check if A, B, new_mask form a sunflower
        # Condition: (A & B) == (A & new_mask) == (B & new_mask)
        # To speed up, we can iterate and break early
        L = len(family)
        for i in range(L):
            a = family[i]
            inter_a_new = a & new_mask
            for j in range(i + 1, L):
                b = family[j]
                inter_b_new = b & new_mask
                if inter_a_new == inter_b_new:
                    if (a & b) == inter_a_new:
                        return False
        return True

    # Initial Greedy Construction
    random.shuffle(all_masks)
    family = []
    for m in all_masks:
        if can_add(m, family):
            family.append(m)
            if time.time() > deadline:
                break

    best_family_size = len(family)
    best_family = family[:]
    
    # Local Search (Hill Climbing with random restarts/swaps)
    # We will perform random moves: Add, Remove, Swap
    # We track the best size seen.
    
    # To make it faster, we might want to pre-compute something or use a different check.
    # But let's try this first.
    
    current_family = family[:]
    current_size = len(current_family)
    
    # Set of masks in current family for O(1) lookup
    current_set = set(current_family)
    
    # List of masks not in current family
    # This is expensive to maintain dynamically, so we'll just pick random from all_masks
    # and check if it's in current_set.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        move_type = random.random()
        
        if move_type < 0.7:
            # Try to add a random mask
            m = random.choice(all_masks)
            if m not in current_set:
                if can_add(m, current_family):
                    current_family.append(m)
                    current_set.add(m)
                    current_size += 1
                    if current_size > best_family_size:
                        best_family_size = current_size
                        best_family = current_family[:]
                else:
                    # Small probability to remove a random element to escape local optimum
                    if random.random() < 0.1 and current_size > 10:
                        idx = random.randint(0, current_size - 1)
                        removed = current_family.pop(idx)
                        current_set.discard(removed)
                        current_size -= 1
                        
        elif move_type < 0.9:
            # Try to remove a random mask
            if current_size > 10:
                idx = random.randint(0, current_size - 1)
                removed = current_family.pop(idx)
                current_set.discard(removed)
                current_size -= 1
                
        else:
            # Try a swap: remove one, add another
            if current_size > 10:
                idx = random.randint(0, current_size - 1)
                removed = current_family.pop(idx)
                current_set.discard(removed)
                current_size -= 1
                
                # Try to add a random new mask
                m = random.choice(all_masks)
                if m not in current_set and m != removed:
                    if can_add(m, current_family):
                        current_family.append(m)
                        current_set.add(m)
                        current_size += 1
                        if current_size > best_family_size:
                            best_family_size = current_size
                            best_family = current_family[:]
                    else:
                        # If can't add, put back the removed one to keep size stable?
                        # Or just keep it removed.
                        pass
                else:
                    # Put back
                    current_family.append(removed)
                    current_set.add(removed)
                    current_size += 1

    # Convert best_family masks back to sorted lists
    result = []
    for mask in best_family:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result

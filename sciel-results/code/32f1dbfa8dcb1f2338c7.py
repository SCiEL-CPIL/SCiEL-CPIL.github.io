import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 16
    k = 4
    all_sets = []
    for comb in combinations(range(n), k):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_sets.append(mask)
        
    best_size = 0
    best_family = []
    
    # We will run multiple random restarts
    
    while time.time() < deadline:
        random.shuffle(all_sets)
        family = []
        # constraints[i] = (core, mask_a, mask_b) for pair (family[i], family[j])?
        # It's easier to just keep a list of all active constraints.
        # When adding s at index idx, we add constraints for (s, family[j]) for all j < idx.
        constraints = []
        
        for s in all_sets:
            if time.time() > deadline:
                break
            
            # Check against all existing constraints
            valid = True
            for core, ma, mb in constraints:
                # Check if s forms sunflower with the pair that generated this constraint
                if (s & core) == core and (s & ma) == 0 and (s & mb) == 0:
                    valid = False
                    break
            
            if valid:
                # Add s to family
                # Generate new constraints for pairs (s, existing)
                for existing in family:
                    core = s & existing
                    # mask_a is s \ core, mask_b is existing \ core
                    # Wait, the condition is symmetric.
                    # For pair (s, existing):
                    # I = s & existing
                    # S_only = s & ~I
                    # E_only = existing & ~I
                    # New constraint: (I, S_only, E_only)
                    # Or (I, E_only, S_only)?
                    # The check is: (new & I) == I and (new & S_only) == 0 and (new & E_only) == 0.
                    # This is symmetric for the two non-core parts.
                    # So we can store (I, s & ~I, existing & ~I).
                    constraints.append((core, s & ~core, existing & ~core))
                
                family.append(s)
                
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        if best_size >= 250:
            break

    result = []
    for mask in best_family:
        elements = []
        for i in range(n):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(elements)
        
    return result

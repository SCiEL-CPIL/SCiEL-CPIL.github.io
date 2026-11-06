import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    # Elements are 1..16, so bits 1..16
    candidates = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        candidates.append(mask)
        
    N = len(candidates)  # 1820
    
    max_size = 0
    best_family = []
    
    # Precompute pairwise intersections for speed
    # pair_int[i][j] = candidates[i] & candidates[j]
    # This is a 1820x1820 matrix, which is about 3.3M entries
    # We can store it as a list of lists or use a dictionary
    # For speed, let's use a list of lists, but only for i<j and symmetric
    
    # Actually, let's precompute a conflict table.
    # conflict[i][j] is a bitmask or list of k such that i, j, k form a sunflower.
    # This is too big (1820^3).
    
    # Let's stick to the current approach but optimize the inner loop.
    
    # Run multiple randomized greedy + local search attempts
    # Use a better local search: simulated annealing or hill climbing with restarts
    
    for attempt in range(50):  # limited attempts to ensure time
        if time.time() > deadline:
            break
            
        random.shuffle(candidates)
        family = []
        family_set = set()
        
        # Phase 1: Greedy Construction with randomization
        # Instead of pure greedy, we can add some randomness
        # Or just greedy is fine as a baseline
        
        for mask in candidates:
            if is_valid_fast(family, mask):
                family.append(mask)
                family_set.add(mask)
        
        if len(family) > max_size:
            max_size = len(family)
            best_family = family.copy()
            
        # Phase 2: Local Search
        # Try to improve by swapping
        # We use a more aggressive local search
        
        steps_without_improvement = 0
        local_best_size = len(family)
        
        # Temperature for simulated annealing
        T = 1.0
        cooling = 0.999
        
        while time.time() < deadline and steps_without_improvement < 500:
            # Pick a random candidate from the complement
            idx = random.randint(0, N - 1)
            c = candidates[idx]
            
            if c in family_set:
                continue
            
            valid, blockers = is_valid_with_blockers(family, c)
            
            if valid:
                family.append(c)
                family_set.add(c)
                if len(family) > max_size:
                    max_size = len(family)
                    best_family = family.copy()
                steps_without_improvement = 0
                local_best_size = len(family)
            else:
                # Conflict found
                # Try to swap
                a, b = blockers
                
                # Randomly pick one blocker to remove
                if random.random() < 0.5:
                    victim = a
                else:
                    victim = b
                
                # Remove victim
                family.remove(victim)
                family_set.remove(victim)
                
                # Try to add c
                valid2 = is_valid_fast(family, c)
                
                if valid2:
                    family.append(c)
                    family_set.add(c)
                    # Size is same (removed 1, added 1)
                    # This is a swap. It doesn't increase size.
                    # We increment steps_without_improvement because size didn't increase.
                    steps_without_improvement += 1
                else:
                    # Swap failed. Restore victim.
                    family.append(victim)
                    family_set.add(victim)
                    steps_without_improvement += 1
                    
            # Cool down
            T *= cooling
            
    # Convert best_family masks back to sorted lists of integers
    result = []
    for m in best_family:
        s = [i for i in range(1, 17) if (m >> i) & 1]
        result.append(sorted(s))
        
    return result

def is_valid_fast(family, mask):
    """
    Check if mask can be added to family without forming a sunflower.
    Returns True if valid, False otherwise.
    """
    n = len(family)
    for i in range(n):
        a = family[i]
        for j in range(i + 1, n):
            b = family[j]
            core = a & b
            if (core & mask) == core:
                # Check disjointness from A\K and B\K
                # A\K = a ^ core
                # B\K = b ^ core
                if (mask & (a ^ core)) == 0 and (mask & (b ^ core)) == 0:
                    return False
    return True

def is_valid_with_blockers(family, mask):
    """
    Check if mask can be added to family.
    Returns (True, None) if valid.
    Returns (False, (a, b)) if invalid, where a, b are a blocking pair.
    """
    n = len(family)
    for i in range(n):
        a = family[i]
        for j in range(i + 1, n):
            b = family[j]
            core = a & b
            if (core & mask) == core:
                if (mask & (a ^ core)) == 0 and (mask & (b ^ core)) == 0:
                    return False, (a, b)
    return True, None

import time
import random
from itertools import combinations

# Precompute all 4-subsets of [1, 16]
ALL_SETS = [frozenset(c) for c in combinations(range(1, 17), 4)]
N_SETS = len(ALL_SETS)

# Convert to bitmasks for fast intersection
ALL_MASKS = [0] * N_SETS
for i, s in enumerate(ALL_SETS):
    m = 0
    for bit in s:
        m |= (1 << (bit - 1))
    ALL_MASKS[i] = m

# Precompute intersection matrix (N_SETS x N_SETS)
# inter[i][j] = ALL_MASKS[i] & ALL_MASKS[j]
# This is 1820 x 1820 = ~3.3M entries. Each is an int (16-bit).
# In numpy, this would be efficient. In pure Python, it's ~3.3M ints ~ 100MB.
# Let's use numpy for the intersection matrix to speed up checks.
import numpy as np

INTER = np.zeros((N_SETS, N_SETS), dtype=np.int32)
for i in range(N_SETS):
    mi = ALL_MASKS[i]
    for j in range(i, N_SETS):
        v = mi & ALL_MASKS[j]
        INTER[i, j] = v
        INTER[j, i] = v

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # We'll maintain a boolean array "in_family" of size N_SETS
    # and a list "family" of indices in the family.
    
    # Fast check: can we add idx_new to current family?
    # For each pair (a, b) in family, check if INTER[idx_new, a] == INTER[a, b] == INTER[idx_new, b]
    # Actually, the condition is: for any triple {a, b, c}, no two of the pairwise intersections are equal?
    # No, the condition is: no three distinct sets have all three pairwise intersections equal.
    # So for a new set x, and existing sets a, b:
    # We need: INTER[x, a] != INTER[a, b] OR INTER[x, a] != INTER[x, b] OR INTER[x, b] != INTER[a, b]
    # i.e., INTER[x,a], INTER[x,b], INTER[a,b] are not all equal.
    
    # So to check if x can be added to family F:
    # For all pairs (a, b) in F:
    #   if INTER[x, a] == INTER[a, b] == INTER[x, b]: return False
    # This is O(|F|^2) per addition.
    
    def can_add(idx_new, family):
        """Check if idx_new can be added to family (list of indices)."""
        if len(family) < 2:
            return True
        
        # Use numpy for vectorized operations if possible?
        # family is a list. Convert to numpy array.
        fam_arr = np.array(family, dtype=np.int32)
        
        # INTER[idx_new, fam_arr] gives intersections of idx_new with each family member
        # INTER[fam_arr[:, None], fam_arr[None, :]] gives pairwise intersections within family
        # This creates a |F| x |F| array. For |F|=200, that's 40K entries.
        
        # Actually, let's just do the loop. It's simpler and might be fast enough.
        for i in range(len(family)):
            a = family[i]
            x_a = INTER[idx_new, a]
            for j in range(i + 1, len(family)):
                b = family[j]
                x_b = INTER[idx_new, b]
                a_b = INTER[a, b]
                if x_a == a_b and x_b == a_b:
                    return False
        return True

    # Greedy construction
    def greedy(order):
        family = []
        for idx in order:
            if time.time() > deadline - 2.0:
                break
            if can_add(idx, family):
                family.append(idx)
        return family

    # Local search
    def local_search(family):
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            in_family = set(family)
            not_in_family = [i for i in range(N_SETS) if i not in in_family]
            
            # 1. Try to add
            random.shuffle(not_in_family)
            for idx_new in not_in_family:
                if time.time() > deadline - 5.0:
                    return family
                if can_add(idx_new, family):
                    family.append(idx_new)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try swap
            if family:
                # Pick random element to remove
                idx_remove = random.choice(family)
                not_in_family = [i for i in range(N_SETS) if i != idx_remove and i not in in_family]
                random.shuffle(not_in_family)
                
                for idx_add in not_in_family:
                    if time.time() > deadline - 5.0:
                        return family
                    temp_family = [x for x in family if x != idx_remove]
                    if can_add(idx_add, temp_family):
                        family = temp_family + [idx_add]
                        in_family = set(family)
                        improved = True
                        break
            
            if improved:
                continue
            
            # 3. Try 2-swap (remove 2, add 2)
            if len(family) >= 2:
                # Remove 2 random
                to_remove = random.sample(family, 2)
                in_remove = set(to_remove)
                temp_family = [x for x in family if x not in in_remove]
                not_in_family = [i for i in range(N_SETS) if i not in set(family)]
                random.shuffle(not_in_family)
                
                # Try to find 2 that can be added
                found = False
                for i in range(min(30, len(not_in_family))):
                    if time.time() > deadline - 5.0:
                        return family
                    a = not_in_family[i]
                    if not can_add(a, temp_family):
                        continue
                    for j in range(i + 1, min(30, len(not_in_family))):
                        if time.time() > deadline - 5.0:
                            return family
                        b = not_in_family[j]
                        if can_add(b, temp_family + [a]):
                            family = temp_family + [a, b]
                            found = True
                            improved = True
                            break
                    if found:
                        break
        
        return family

    # Main
    random.seed(42)
    
    best_family = []
    best_size = 0
    
    # Phase 1: Greedy restarts
    for restart in range(100):
        if time.time() > deadline - 15.0:
            break
        
        order = list(range(N_SETS))
        random.shuffle(order)
        family = greedy(order)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
        
        # Early local search on promising families
        if len(family) > best_size - 3 and time.time() < deadline - 10.0:
            improved = local_search(family)
            if len(improved) > best_size:
                best_size = len(improved)
                best_family = improved[:]
    
    # Phase 2: Local search on best family
    if best_size > 0 and time.time() < deadline - 5.0:
        current = best_family[:]
        for _ in range(20):
            if time.time() > deadline - 2.0:
                break
            improved = local_search(current)
            if len(improved) > best_size:
                best_size = len(improved)
                best_family = improved[:]
                current = improved[:]
            else:
                # Perturb
                if len(current) > 3:
                    to_remove = random.sample(current, 3)
                    current = [x for x in current if x not in set(to_remove)]
                else:
                    break
    
    return [sorted(list(ALL_SETS[idx])) for idx in best_family]

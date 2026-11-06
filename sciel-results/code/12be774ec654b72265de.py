import time
from itertools import combinations
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin before 90s kill
    
    # Represent each 4-set as a 16-bit bitmask (0-indexed elements 0..15)
    all_sets = []
    for c in combinations(range(16), 4):
        mask = 0
        for x in c:
            mask |= (1 << x)
        all_sets.append(mask)
    
    n_all = len(all_sets)  # 1820
    
    def is_sunflower(a, b, c):
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc
    
    def can_add(family, new_mask):
        """family is a list of masks. Check if adding new_mask creates a sunflower."""
        n = len(family)
        for i in range(n):
            a = family[i]
            xa = new_mask & a
            for j in range(i + 1, n):
                b = family[j]
                if (new_mask & b) == xa and (a & b) == xa:
                    return False
        return True
    
    def greedy_random(seed_val):
        rng = Random(seed_val)
        order = list(range(n_all))
        rng.shuffle(order)
        family = []
        for idx in order:
            if time.time() > deadline:
                break
            if can_add(family, all_sets[idx]):
                family.append(all_sets[idx])
        return family
    
    # Multiple greedy restarts
    best_family = []
    for seed_val in range(200):
        if time.time() > deadline - 10:
            break
        fam = greedy_random(seed_val)
        if len(fam) > len(best_family):
            best_family = fam
            if len(best_family) >= 250:
                break
    
    # Local search phase 1: try to add more sets
    if time.time() < deadline - 15:
        in_fam = set(best_family)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for idx in range(n_all):
                m = all_sets[idx]
                if m in in_fam:
                    continue
                if can_add(list(in_fam), m):
                    in_fam.add(m)
                    best_family = list(in_fam)
                    improved = True
                    break
    
    # Local search phase 2: try swaps (remove 1, add 2+)
    if time.time() < deadline - 10:
        in_fam = set(best_family)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for rm in list(in_fam):
                if time.time() > deadline - 5:
                    break
                cand = in_fam - {rm}
                added = 0
                for idx in range(n_all):
                    m = all_sets[idx]
                    if m in cand:
                        continue
                    if can_add(list(cand), m):
                        cand.add(m)
                        added += 1
                        if added >= 2:
                            break
                if added >= 2:
                    in_fam = cand
                    best_family = list(in_fam)
                    improved = True
                    break
    
    # Local search phase 3: try removing 1 and adding 3+
    if time.time() < deadline - 5:
        in_fam = set(best_family)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for rm in list(in_fam):
                if time.time() > deadline - 3:
                    break
                cand = in_fam - {rm}
                added = 0
                for idx in range(n_all):
                    m = all_sets[idx]
                    if m in cand:
                        continue
                    if can_add(list(cand), m):
                        cand.add(m)
                        added += 1
                        if added >= 3:
                            break
                if added >= 3:
                    in_fam = cand
                    best_family = list(in_fam)
                    improved = True
                    break
    
    # Convert back to 1-indexed sorted lists
    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if mask & (1 << i)]
        result.append(sorted(elements))
    return result

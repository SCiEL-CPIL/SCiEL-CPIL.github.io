import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Generate all 4-element subsets of [1,16] as bitmasks
    # Elements 1..16 correspond to bits 0..15
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    n_all = len(all_sets)  # 1820
    
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
    
    # Phase 1: Multiple greedy restarts with random orderings
    best_family = []
    best_size = 0
    
    for attempt in range(500):
        if time.time() > deadline - 20:
            break
        
        if attempt == 0:
            # Lexicographic order
            candidates = list(all_sets)
        else:
            # Random order
            candidates = list(all_sets)
            random.shuffle(candidates)
        
        family = []
        for s in candidates:
            if time.time() > deadline - 20:
                break
            if can_add(family, s):
                family.append(s)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        if best_size >= 250:
            break
    
    # Phase 2: Local search - try to add more sets
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
                    best_size = len(in_fam)
                    improved = True
                    break
    
    # Phase 3: Simulated annealing / swap-based local search
    if time.time() < deadline - 10:
        current_family = set(best_family)
        current_size = len(current_family)
        best_family_sa = list(current_family)
        best_size_sa = current_size
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        iterations = 50000
        
        for it in range(iterations):
            if time.time() > deadline - 5:
                break
            
            # Temperature annealing
            T = T_start * (T_end / T_start) ** (it / iterations)
            
            # Choose a random set to remove
            if not current_family:
                break
            rm = random.choice(list(current_family))
            cand = current_family - {rm}
            
            # Try to add as many sets as possible
            added = 0
            # Shuffle candidates for diversity
            cand_list = list(all_sets)
            random.shuffle(cand_list)
            
            for m in cand_list:
                if m in cand:
                    continue
                if can_add(list(cand), m):
                    cand.add(m)
                    added += 1
                    if added >= 2:  # Need net gain of at least 1
                        break
            
            new_size = len(cand)
            
            # Acceptance criterion
            if new_size > current_size:
                # Always accept improvement
                current_family = cand
                current_size = new_size
                if current_size > best_size_sa:
                    best_size_sa = current_size
                    best_family_sa = list(current_family)
            else:
                # Accept with probability based on temperature
                delta = new_size - current_size
                if delta < 0:
                    prob = min(1.0, T / abs(delta))
                    if random.random() < prob:
                        current_family = cand
                        current_size = new_size
    
    # Update best if SA found something better
    if best_size_sa > best_size:
        best_family = best_family_sa
        best_size = best_size_sa
    
    # Phase 4: Final greedy add pass
    if time.time() < deadline - 3:
        in_fam = set(best_family)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(n_all):
                m = all_sets[idx]
                if m in in_fam:
                    continue
                if can_add(list(in_fam), m):
                    in_fam.add(m)
                    best_family = list(in_fam)
                    best_size = len(in_fam)
                    improved = True
                    break
    
    # Convert back to 1-indexed sorted lists
    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if mask & (1 << i)]
        result.append(sorted(elements))
    return result

import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    n = 16
    k = 4
    
    # Generate all k-subsets as bitmasks
    all_sets = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    all_sets.append(mask)
    
    N = len(all_sets) # 1820
    
    # Precompute intersections for all pairs
    intersections = [[0]*N for _ in range(N)]
    for i in range(N):
        mi = all_sets[i]
        for j in range(i+1, N):
            mj = all_sets[j]
            inter = mi & mj
            intersections[i][j] = inter
            intersections[j][i] = inter
            
    # Use a fixed seed for reproducibility
    random.seed(42)

    def count_bad_triples_for_new(new_idx, family):
        count = 0
        # family is a list of indices
        L = len(family)
        for i in range(L):
            bi = family[i]
            inter_nb = intersections[new_idx][bi]
            for j in range(i+1, L):
                cj = family[j]
                if inter_nb == intersections[bi][cj] and inter_nb == intersections[new_idx][cj]:
                    count += 1
        return count

    def greedy_random():
        family = []
        family_set = set()
        indices = list(range(N))
        random.shuffle(indices)
        
        for idx in indices:
            if idx in family_set:
                continue
            new_bad = count_bad_triples_for_new(idx, family)
            if new_bad == 0:
                family.append(idx)
                family_set.add(idx)
        return family

    def local_search(family, time_limit):
        current_family = family[:]
        current_set = set(current_family)
        
        no_improve_count = 0
        last_improve_time = time.time()
        best_local_size = len(current_family)
        best_local_family = current_family[:]
        
        while time.time() < time_limit:
            # Try to add
            added = False
            for _ in range(5):
                b_idx = random.randint(0, N-1)
                if b_idx not in current_set:
                    new_bad = count_bad_triples_for_new(b_idx, current_family)
                    if new_bad == 0:
                        current_family.append(b_idx)
                        current_set.add(b_idx)
                        added = True
                        no_improve_count = 0
                        break
            
            if not added:
                # Try swaps
                swapped = False
                for _ in range(20):
                    if not current_family: break
                    a_idx = random.choice(current_family)
                    b_idx = random.randint(0, N-1)
                    if b_idx in current_set:
                        continue
                    
                    # Check swap
                    # Remove a_idx
                    temp_family = [x for x in current_family if x != a_idx]
                    new_bad = count_bad_triples_for_new(b_idx, temp_family)
                    if new_bad == 0:
                        # Accept
                        current_family = temp_family
                        current_family.append(b_idx)
                        current_set.remove(a_idx)
                        current_set.add(b_idx)
                        swapped = True
                        no_improve_count = 0
                        break
            
            if not added and not swapped:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
            
            if len(current_family) > best_local_size:
                best_local_size = len(current_family)
                best_local_family = current_family[:]
                last_improve_time = time.time()
                
            if time.time() - last_improve_time > 2.0:
                 break

        return best_local_family

    best_family_global = []
    best_size_global = 0
    
    while time.time() < deadline:
        fam = greedy_random()
        
        if len(fam) > best_size_global:
            best_family_global = fam
            best_size_global = len(fam)
            
        improved_fam = local_search(fam, deadline - 0.5)
        
        if len(improved_fam) > best_size_global:
            best_family_global = improved_fam
            best_size_global = len(improved_fam)
            
        if best_size_global >= 230:
             improved_fam = local_search(best_family_global, deadline - 0.5)
             if len(improved_fam) > best_size_global:
                 best_family_global = improved_fam
                 best_size_global = len(improved_fam)

    result = []
    for idx in best_family_global:
        mask = all_sets[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i+1)
        result.append(sorted(elems))
        
    return result

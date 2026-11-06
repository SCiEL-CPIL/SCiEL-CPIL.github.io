import time
import random
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 55.0
    
    # All 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        m = 0
        for x in combo:
            m |= (1 << x)
        all_sets.append(m)
    
    n_sets = len(all_sets)
    
    def mask_to_list(m):
        return sorted((i+1) for i in range(16) if (m >> i) & 1)
    
    # Precompute pairwise intersections
    pair_intersect = [[0]*n_sets for _ in range(n_sets)]
    for i in range(n_sets):
        for j in range(i+1, n_sets):
            v = all_sets[i] & all_sets[j]
            pair_intersect[i][j] = v
            pair_intersect[j][i] = v
    
    # Precompute for each set C, which pairs (i,j) would be blocked if C is added
    # blocked_pairs[C] = set of (i,j) where adding C creates a sunflower with i,j
    # Actually, we need: for a candidate C and current family F, C conflicts with (i,j) in F
    # if (F[i] & C) == (F[j] & C) == (F[i] & F[j])
    
    # For speed: precompute for each C, and each pair (i,j), whether C conflicts
    # This is too large. Instead, use a different approach.
    
    # For each set C, precompute its intersection with all other sets
    # inter[C][i] = all_sets[C] & all_sets[i]
    inter = [[0]*n_sets for _ in range(n_sets)]
    for i in range(n_sets):
        for j in range(n_sets):
            inter[i][j] = all_sets[i] & all_sets[j]
    
    def can_add_idx(C_idx, family_indices):
        """
        Checks if all_sets[C_idx] can be added to family (given as indices).
        Returns (True, None) if valid.
        Returns (False, (A_idx, B_idx)) if blocked by pair A, B.
        """
        C = all_sets[C_idx]
        n = len(family_indices)
        for i in range(n):
            fi = family_indices[i]
            fi_C = inter[fi][C_idx]
            for j in range(i+1, n):
                fj = family_indices[j]
                fj_C = inter[fj][C_idx]
                if fi_C == fj_C and fi_C == pair_intersect[fi][fj]:
                    return False, (fi, fj)
        return True, None
    
    def greedy_pass(order_indices):
        family = []
        for C_idx in order_indices:
            valid, _ = can_add_idx(C_idx, family)
            if valid:
                family.append(C_idx)
        return family
    
    best_family = []
    best_size = 0
    
    # Order 1: Natural (Lexicographic)
    fam1 = greedy_pass(list(range(n_sets)))
    if len(fam1) > best_size:
        best_family = fam1
        best_size = len(fam1)
        
    # Order 2: Reverse Lexicographic
    fam2 = greedy_pass(list(range(n_sets))[::-1])
    if len(fam2) > best_size:
        best_family = fam2
        best_size = len(fam2)
        
    # Order 3: Random
    for _ in range(20):
        if time.time() > DEADLINE: break
        order = list(range(n_sets))
        random.shuffle(order)
        fam3 = greedy_pass(order)
        if len(fam3) > best_size:
            best_family = fam3
            best_size = len(fam3)
            
    def local_search(family_indices, deadline):
        family = list(family_indices)
        in_family = set(family)
        candidates = [s for s in range(n_sets) if s not in in_family]
        
        random.shuffle(candidates)
        
        no_improve_count = 0
        max_no_improve = 800
        
        while time.time() < deadline:
            if not candidates:
                break
                
            # Pick a candidate
            idx = random.randint(0, len(candidates) - 1)
            C_idx = candidates[idx]
            
            valid, block_pair = can_add_idx(C_idx, family)
            
            if valid:
                # Add it
                family.append(C_idx)
                in_family.add(C_idx)
                candidates.pop(idx)
                no_improve_count = 0
                if len(candidates) % 100 == 0:
                    random.shuffle(candidates)
            else:
                A_idx, B_idx = block_pair
                
                improved = False
                
                def can_add_except(C_idx, family, excluded_idx):
                    C = all_sets[C_idx]
                    n = len(family)
                    for i in range(n):
                        fi = family[i]
                        if fi == excluded_idx: continue
                        fi_C = inter[fi][C_idx]
                        for j in range(i+1, n):
                            fj = family[j]
                            if fj == excluded_idx: continue
                            fj_C = inter[fj][C_idx]
                            if fi_C == fj_C and fi_C == pair_intersect[fi][fj]:
                                return False
                    return True

                # Try swap A
                if can_add_except(C_idx, family, A_idx):
                    family.remove(A_idx)
                    family.append(C_idx)
                    in_family.discard(A_idx)
                    in_family.add(C_idx)
                    candidates.pop(idx)
                    candidates.append(A_idx)
                    improved = True
                    no_improve_count = 0
                else:
                    # Try swap B
                    if can_add_except(C_idx, family, B_idx):
                        family.remove(B_idx)
                        family.append(C_idx)
                        in_family.discard(B_idx)
                        in_family.add(C_idx)
                        candidates.pop(idx)
                        candidates.append(B_idx)
                        improved = True
                        no_improve_count = 0
                    else:
                        # Try random swap with any member
                        if no_improve_count % 3 == 0 and family:
                            rand_idx = random.randint(0, len(family)-1)
                            R_idx = family[rand_idx]
                            if R_idx != A_idx and R_idx != B_idx:
                                if can_add_except(C_idx, family, R_idx):
                                    family.remove(R_idx)
                                    family.append(C_idx)
                                    in_family.discard(R_idx)
                                    in_family.add(C_idx)
                                    candidates.pop(idx)
                                    candidates.append(R_idx)
                                    improved = True
                                    no_improve_count = 0
                        if not improved:
                            no_improve_count += 1
                            if no_improve_count > max_no_improve:
                                break
                                
        return family

    # Run local search on the best family
    if best_family:
        improved_family = local_search(best_family, DEADLINE - 3.0)
        if len(improved_family) > len(best_family):
            best_family = improved_family

    # If we still have time, do another round of greedy + local search
    if time.time() < DEADLINE - 5.0:
        for _ in range(8):
            if time.time() > DEADLINE - 5.0: break
            order = list(range(n_sets))
            random.shuffle(order)
            fam = greedy_pass(order)
            if len(fam) > len(best_family):
                best_family = fam
            
            if time.time() < DEADLINE - 3.0:
                improved_fam = local_search(best_family, DEADLINE - 3.0)
                if len(improved_fam) > len(best_family):
                    best_family = improved_fam

    return [mask_to_list(all_sets[m]) for m in best_family]

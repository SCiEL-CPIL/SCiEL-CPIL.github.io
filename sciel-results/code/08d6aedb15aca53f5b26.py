import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    total_masks = len(all_masks)
    mask_to_idx = {m: i for i, m in enumerate(all_masks)}
    
    def get_submasks(mask):
        sub = mask
        while True:
            yield sub
            if sub == 0:
                break
            sub = (sub - 1) & mask
    
    def is_compatible(S, family):
        groups = {}
        for A in family:
            K = A & S
            groups.setdefault(K, []).append(A)
        
        for K, group in groups.items():
            if len(group) < 2:
                continue
            L = len(group)
            for i in range(L):
                Ai = group[i]
                Ai_out = Ai ^ K
                for j in range(i + 1, L):
                    Bj = group[j]
                    Bj_out = Bj ^ K
                    if (Ai_out & Bj_out) == 0:
                        return False
        return True
    
    def family_is_valid(family):
        # Verify no sunflower exists
        fam_list = list(family)
        L = len(fam_list)
        for i in range(L):
            for j in range(i + 1, L):
                K = fam_list[i] & fam_list[j]
                for l in range(j + 1, L):
                    if (fam_list[i] & fam_list[l]) == K and (fam_list[j] & fam_list[l]) == K:
                        return False
        return True
    
    best_family = []
    rng = random.Random(42)
    
    # Phase 1: Greedy with different orderings
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        shuffled = all_masks[:]
        rng.shuffle(shuffled)
        
        current_family = []
        for S in shuffled:
            if is_compatible(S, current_family):
                current_family.append(S)
        
        if len(current_family) > len(best_family):
            best_family = current_family[:]
            if len(best_family) >= 250:
                break
    
    # Phase 2: Local search - try to add sets that are compatible
    if time.time() < deadline - 5:
        # Try to add any set not in family
        in_family = set(best_family)
        for S in all_masks:
            if S in in_family:
                continue
            if is_compatible(S, best_family):
                best_family.append(S)
                in_family.add(S)
    
    # Phase 3: Local search - try swaps
    if time.time() < deadline - 3:
        in_family = set(best_family)
        not_in_family = [S for S in all_masks if S not in in_family]
        
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try removing one set and adding two
            for idx in range(len(best_family)):
                if time.time() > deadline - 3:
                    break
                removed = best_family[idx]
                candidate = best_family[:idx] + best_family[idx+1:]
                
                # Try to add up to 2 sets
                added = 0
                for S in not_in_family[:200]:  # Limit candidates
                    if S == removed:
                        continue
                    if is_compatible(S, candidate):
                        candidate.append(S)
                        added += 1
                        if added >= 2:
                            break
                
                if len(candidate) > len(best_family):
                    best_family = candidate
                    in_family = set(best_family)
                    not_in_family = [S for S in all_masks if S not in in_family]
                    improved = True
                    break
            
            if not improved:
                break
    
    # Convert best_family to list of sorted 1-indexed lists
    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(n) if (mask & (1 << i))]
        result.append(elements)
        
    return result

"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    n = 16
    k = 4

    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)

    m = len(all_masks)  # C(16,4) = 1820

    # Precompute pairwise intersections
    inter = [[0] * m for _ in range(m)]
    for i in range(m):
        mi = all_masks[i]
        for j in range(i + 1, m):
            v = mi & all_masks[j]
            inter[i][j] = v
            inter[j][i] = v

    # Precompute for each pair (i,j) the set of forbidden indices
    # forbidden[i][j] = set of c such that i,j,c form a sunflower
    # A sunflower: inter[i][j] == inter[i][c] == inter[j][c]
    
    # For speed, precompute forbidden as a list of lists
    # But storing all 1820*1820 sets is expensive. Instead, compute on the fly
    # but cache the forbidden lists for pairs we actually use.
    
    def get_forbidden(i, j):
        """Return list of indices c != i,j such that i,j,c form a sunflower."""
        ab = inter[i][j]
        result = []
        for c in range(m):
            if c == i or c == j:
                continue
            if inter[i][c] == ab and inter[j][c] == ab:
                result.append(c)
        return result

    def mask_to_list(mask):
        result = []
        for i in range(n):
            if mask & (1 << i):
                result.append(i + 1)
        return result

    def check_violation(family_indices, idx):
        """Check if adding idx to family creates a sunflower violation."""
        for ii in range(len(family_indices)):
            a = family_indices[ii]
            ia = inter[a][idx]
            for jj in range(ii + 1, len(family_indices)):
                b = family_indices[jj]
                ab = inter[a][b]
                jb = inter[b][idx]
                if ab == ia == jb:
                    return True
        return False

    def greedy_construct(order, family=None):
        """Greedy construction using conflict counts."""
        if family is None:
            family = []
        conf_count = [0] * m
        
        for idx in order:
            if time.time() > deadline:
                break
            if conf_count[idx] == 0:
                family.append(idx)
                for a_idx in family[:-1]:
                    forbids = get_forbidden(a_idx, idx)
                    for c_idx in forbids:
                        conf_count[c_idx] += 1
        return family

    def local_search_swap(family_indices):
        """Local search with swap operations to improve the family."""
        if time.time() > deadline - 3:
            return family_indices
        
        current = family_indices[:]
        in_family = set(current)
        improved = True
        
        while improved and time.time() < deadline - 2:
            improved = False
            
            # First, try to add any valid element
            for c_idx in range(m):
                if c_idx not in in_family and not check_violation(current, c_idx):
                    current.append(c_idx)
                    in_family.add(c_idx)
                    improved = True
                    break
            
            if not improved:
                # Try swaps: remove one, try to add two
                for i in range(len(current)):
                    if time.time() > deadline - 2:
                        break
                    removed = current[i]
                    # Temporarily remove
                    temp_family = current[:i] + current[i+1:]
                    temp_in = in_family - {removed}
                    
                    # Try to add two elements
                    added_count = 0
                    new_adds = []
                    for c_idx in range(m):
                        if c_idx not in temp_in and c_idx != removed:
                            if not check_violation(temp_family, c_idx):
                                temp_family.append(c_idx)
                                temp_in.add(c_idx)
                                new_adds.append(c_idx)
                                added_count += 1
                                if added_count >= 2:
                                    break
                    
                    if added_count >= 2:
                        current = temp_family
                        in_family = temp_in
                        improved = True
                        break
        
        return current

    best_size = 0
    best_family_idx = []

    # Strategy 1: Greedy in natural order
    family = greedy_construct(list(range(m)))
    if len(family) > best_size:
        best_size = len(family)
        best_family_idx = family[:]

    # Strategy 2: Multiple random shuffles with greedy
    random.seed(12345)
    max_attempts = 200

    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break

        order = list(range(m))
        random.shuffle(order)
        family = greedy_construct(order)

        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Strategy 3: Local search with swaps on best family
    if time.time() < deadline - 5:
        improved_family = local_search_swap(best_family_idx)
        if len(improved_family) > best_size:
            best_size = len(improved_family)
            best_family_idx = improved_family[:]

    # Strategy 4: More random restarts with local search
    seeds = [7, 42, 99, 2024, 555, 777, 1234, 4321, 8888, 13579,
             2468, 1357, 9999, 1111, 2222, 3333, 4444, 5555, 6666, 7777,
             1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    for seed_val in seeds:
        if time.time() > deadline - 3:
            break
        random.seed(seed_val)
        order = list(range(m))
        random.shuffle(order)
        family = greedy_construct(order)
        
        if time.time() < deadline - 2:
            family = local_search_swap(family)
            
        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Final: try to add more to the best family
    if time.time() < deadline - 1:
        in_family = set(best_family_idx)
        candidates = [idx for idx in range(m) if idx not in in_family]
        random.shuffle(candidates)
        for idx in candidates:
            if time.time() > deadline - 0.5:
                break
            if not check_violation(best_family_idx, idx):
                best_family_idx.append(idx)
                in_family.add(idx)

    # Cap at 250
    if len(best_family_idx) > 250:
        best_family_idx = best_family_idx[:250]

    result = [mask_to_list(all_masks[idx]) for idx in best_family_idx]
    return result
# EVOLVE-BLOCK-END

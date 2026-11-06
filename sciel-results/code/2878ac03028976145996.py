import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Generate all 4-subsets of [1,16]
    all_sets = []
    for combo in combinations(range(1, 17), 4):
        all_sets.append(combo)
    n_sets = len(all_sets)  # 1820
    
    # Convert to bitmasks for fast intersection
    masks = []
    for s in all_sets:
        m = 0
        for x in s:
            m |= (1 << (x-1))
        masks.append(m)
    
    # Precompute pairwise intersections as bitmasks
    # inter[i][j] = masks[i] & masks[j]
    # Store as list of lists of ints
    inter = [[0]*n_sets for _ in range(n_sets)]
    for i in range(n_sets):
        inter[i][i] = masks[i]
        for j in range(i+1, n_sets):
            ij = masks[i] & masks[j]
            inter[i][j] = ij
            inter[j][i] = ij
    
    def is_sunflower(i, j, k):
        return inter[i][j] == inter[i][k] == inter[j][k]
    
    def can_add(candidate_idx, family_set):
        """Check if adding candidate to family creates no sunflower"""
        fam_list = list(family_set)
        for a in range(len(fam_list)):
            ia = fam_list[a]
            for b in range(a+1, len(fam_list)):
                ib = fam_list[b]
                if is_sunflower(ia, ib, candidate_idx):
                    return False
        return True
    
    # Greedy initialization with random ordering
    def greedy_init(order):
        family = set()
        for idx in order:
            if can_add(idx, family):
                family.add(idx)
        return family
    
    # Local search: try to add one set
    def try_add_one(family):
        for idx in range(n_sets):
            if idx not in family:
                if can_add(idx, family):
                    family.add(idx)
                    return True
        return False
    
    # Local search: try to swap - remove one, add one
    def try_swap(family):
        fam_list = list(family)
        for remove_idx in fam_list:
            temp_fam = family - {remove_idx}
            for add_idx in range(n_sets):
                if add_idx not in temp_fam:
                    if can_add(add_idx, temp_fam):
                        family = temp_fam | {add_idx}
                        return True
        return False
    
    # Multi-start local search
    best_family = set()
    best_size = 0
    
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random ordering for greedy
        order = list(range(n_sets))
        random.shuffle(order)
        
        # Greedy init
        family = greedy_init(order)
        
        # Local search: repeatedly try to add
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = try_add_one(family)
        
        # Try some swaps
        for _ in range(5):
            if time.time() > deadline:
                break
            if not try_swap(family):
                break
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = set(family)
    
    # Convert back to sorted lists
    result = [sorted(all_sets[i]) for i in best_family]
    result.sort()
    return result[:250]

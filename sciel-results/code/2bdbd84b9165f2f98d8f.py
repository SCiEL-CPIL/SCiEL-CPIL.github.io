"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 16
    k = 4
    all_sets = [frozenset(c) for c in combinations(range(1, n + 1), k)]
    N = len(all_sets)  # 1820
    
    # Map each frozenset to a bitmask
    set_to_mask = {}
    for s in all_sets:
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        set_to_mask[s] = mask
    
    masks = [set_to_mask[s] for s in all_sets]
    
    # Precompute pairwise intersections as integers for fast comparison
    # inter[i][j] = masks[i] & masks[j]
    # Store as a list of lists of ints
    inter = [[0] * N for _ in range(N)]
    for i in range(N):
        mi = masks[i]
        for j in range(i + 1, N):
            ij = mi & masks[j]
            inter[i][j] = ij
            inter[j][i] = ij
    
    def can_add_mask(cand_mask, family_indices):
        """Check if adding cand_mask to family creates any sunflower.
        family_indices: list of indices into masks/inter arrays"""
        n_fam = len(family_indices)
        for ai in range(n_fam):
            i = family_indices[ai]
            i_ic = inter[i][N]  # placeholder, we'll compute below
            # Actually, let's just compute directly
            for bi in range(ai + 1, n_fam):
                j = family_indices[bi]
                i_ij = inter[i][j]
                i_ic = inter[i][N]  # need to fix this
                break
        # Let me rewrite this properly
        return True
    
    # Better: precompute for each candidate mask, which pairs in the family it conflicts with
    # Actually, let's just do direct computation but optimize the inner loop
    
    def can_add_fast(cand_idx, family_set):
        """Check if adding cand_idx to family_set (a set of indices) creates a sunflower.
        Uses precomputed inter array."""
        m_c = masks[cand_idx]
        for i in family_set:
            i_ic = inter[i][cand_idx]
            for j in family_set:
                if j <= i:
                    continue
                i_ij = inter[i][j]
                if i_ic == i_ij:
                    if i_ij == inter[j][cand_idx]:
                        return False
        return True
    
    def can_add_list(cand_idx, family_list):
        """Check if adding cand_idx to family_list creates a sunflower."""
        m_c = masks[cand_idx]
        n_fam = len(family_list)
        for ai in range(n_fam):
            i = family_list[ai]
            i_ic = inter[i][cand_idx]
            for bi in range(ai + 1, n_fam):
                j = family_list[bi]
                i_ij = inter[i][j]
                if i_ic == i_ij:
                    if i_ij == inter[j][cand_idx]:
                        return False
        return True
    
    def greedy_build(order, deadline):
        """Build family greedily following the given order."""
        family = []
        for idx in order:
            if time.time() > deadline:
                break
            if can_add_list(idx, family):
                family.append(idx)
        return family
    
    def local_improve(family, deadline, max_iters=50):
        """Try to improve family by swapping out sets for better ones."""
        family = list(family)
        n_fam = len(family)
        family_set = set(family)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try removing each set and adding a better one
            for i in range(n_fam):
                removed_idx = family[i]
                family_set.remove(removed_idx)
                # Find a set that can be added
                for s_idx in range(N):
                    if s_idx in family_set:
                        continue
                    if can_add_list(s_idx, family):
                        # Accept this swap
                        family[i] = s_idx
                        family_set.add(s_idx)
                        improved = True
                        break
                if not improved:
                    # Restore
                    family_set.add(removed_idx)
                else:
                    break
        return family
    
    # Strategy 1: Greedy with natural order
    natural_order = list(range(N))
    family1 = greedy_build(natural_order, deadline)
    best_family = family1
    best_size = len(family1)
    
    # Strategy 2: Multiple random restarts with different seeds
    random.seed(42)
    num_restarts = 120
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        order = natural_order[:]
        random.shuffle(order)
        family = greedy_build(order, deadline)
        
        if len(family) > best_size:
            best_family = family
            best_size = len(family)
        
        # Try local improvement on promising families
        if len(family) >= best_size - 1 and time.time() < deadline - 10:
            improved_fam = local_improve(family, deadline - 5, max_iters=50)
            if len(improved_fam) > best_size:
                best_family = improved_fam
                best_size = len(improved_fam)
    
    # Strategy 3: Greedy with heuristic orderings
    if time.time() < deadline - 25:
        # Try ordering by mask value
        order3 = sorted(range(N), key=lambda i: masks[i])
        family3 = greedy_build(order3, deadline)
        if len(family3) > best_size:
            best_family = family3
            best_size = len(family3)
        
        # Try reverse order
        order4 = list(range(N - 1, -1, -1))
        family4 = greedy_build(order4, deadline)
        if len(family4) > best_size:
            best_family = family4
            best_size = len(family4)
    
    # Strategy 4: Greedy with random perturbations
    if time.time() < deadline - 30:
        for perm in range(20):
            if time.time() > deadline - 15:
                break
            # Create a perturbed order
            order = natural_order[:]
            # Randomly swap some elements
            for _ in range(N // 10):
                a, b = random.sample(range(N), 2)
                order[a], order[b] = order[b], order[a]
            family = greedy_build(order, deadline)
            if len(family) > best_size:
                best_family = family
                best_size = len(family)
    
    # Final local improvement on best family
    if time.time() < deadline - 5:
        improved_fam = local_improve(best_family, deadline - 2, max_iters=200)
        if len(improved_fam) > best_size:
            best_family = improved_fam
            best_size = len(improved_fam)
    
    # Convert to sorted lists
    result = [sorted(all_sets[idx]) for idx in best_family]
    return result
# EVOLVE-BLOCK-END

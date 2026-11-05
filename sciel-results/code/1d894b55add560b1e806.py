import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 16
    k = 4
    
    # Generate all 4-element subsets of [1, 16]
    all_subsets = list(combinations(range(1, n + 1), k))
    total_sets = len(all_subsets)  # 1820
    
    # Precompute bitmasks for fast intersection
    masks = []
    for s in all_subsets:
        m = 0
        for bit in s:
            m |= (1 << (bit - 1))
        masks.append(m)
    
    # Precompute pairwise intersections (as bitmasks)
    # inter[i][j] = masks[i] & masks[j]
    # For memory efficiency, we can compute on the fly or precompute
    # 1820 * 1820 = ~3.3 million entries, which is manageable
    
    # Actually, let's think about the sunflower condition more carefully.
    # Three sets A, B, C form a sunflower if A∩B = A∩C = B∩C.
    # Let this common intersection be I.
    # So for any three sets in our family, no three can have all pairwise intersections equal.
    
    # Key insight: For a fixed set S, if we have two other sets A, B such that
    # A∩S = B∩S = A∩B = I, then {A, B, S} forms a sunflower.
    # 
    # So when trying to add S to a family, we need to check:
    # For every pair (A, B) already in the family, if A∩S = B∩S = A∩B, conflict.
    #
    # Equivalently, group family members by their intersection with S.
    # For each group with common intersection I, check if any pair A, B in that group
    # has A∩B = I.
    
    # Let's use a smarter approach:
    # For each pair of sets in the family, compute their intersection.
    # When adding S, for each existing set A, compute I = A∩S.
    # Then we need to check: is there another set B in the family with B∩S = I AND A∩B = I?
    #
    # We can maintain for each possible intersection pattern with S, the sets that have it.
    # But this is still O(|family|^2) in the worst case per addition.
    
    # Alternative: Use a different data structure.
    # For each pair of sets, store their intersection.
    # When adding S, for each existing set A, let I = A∩S.
    # We need to know: among all B in family with B∩S = I, is there one with A∩B = I?
    #
    # This is hard to precompute easily.
    
    # Let's try a local search / simulated annealing approach instead.
    # Start with a random valid family, then try to add sets or swap sets.
    
    # First, let's implement a fast validity check for a family.
    # For a given family, check if it's sunflower-free.
    
    def is_valid_family(family_indices):
        """Check if the family (list of indices into all_subsets) is sunflower-free."""
        m = len(family_indices)
        if m < 3:
            return True
        
        # For each triple, check sunflower condition
        # This is O(m^3) which is too slow for large m
        # Instead, use the pairwise intersection approach
        
        # Build a dictionary: for each pair (i, j), store the intersection mask
        # Then for each triple (i, j, k), check if all three pairwise intersections are equal
        
        # Better: For each pair (i, j), compute inter_ij.
        # For a triple to be a sunflower, inter_ij = inter_ik = inter_jk.
        # So group pairs by their intersection value.
        # If any intersection value appears in 3 pairs that form a triangle, it's a sunflower.
        
        # Actually, let's just do it directly but efficiently.
        # For each set i in the family, for each set j > i, compute inter_ij.
        # Then for each set k > j, check if inter_ik == inter_jk == inter_ij.
        
        # This is still O(m^3) but with small constant.
        # For m up to ~200, that's 200^3/6 ≈ 1.3 million operations, which is fine.
        
        for i in range(m):
            idx_i = family_indices[i]
            mask_i = masks[idx_i]
            for j in range(i + 1, m):
                idx_j = family_indices[j]
                mask_j = masks[idx_j]
                inter_ij = mask_i & mask_j
                for k in range(j + 1, m):
                    idx_k = family_indices[k]
                    mask_k = masks[idx_k]
                    inter_ik = mask_i & mask_k
                    if inter_ik != inter_ij:
                        continue
                    inter_jk = mask_j & mask_k
                    if inter_jk == inter_ij:
                        return False
        return True
    
    # Greedy construction with multiple random restarts
    best_family = []
    best_size = 0
    
    random.seed(42)  # For reproducibility
    
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Random order of all sets
        order = list(range(total_sets))
        random.shuffle(order)
        
        family = []
        
        for idx in order:
            if time.time() > deadline - 5.0:
                break
            
            # Try to add this set
            candidate = family + [idx]
            
            # Check validity
            if is_valid_family(candidate):
                family.append(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
        
        # If we've found a large family, we can break early
        if best_size >= 100:
            break
    
    # Now try to improve the best family with local search
    if best_family:
        family = best_family[:]
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try to add each set not in the family
            in_family = set(family)
            for idx in range(total_sets):
                if idx in in_family:
                    continue
                candidate = family + [idx]
                if is_valid_family(candidate):
                    family.append(idx)
                    in_family.add(idx)
                    improved = True
                    break  # Restart the loop since family changed
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    return [list(all_subsets[i]) for i in best_family]

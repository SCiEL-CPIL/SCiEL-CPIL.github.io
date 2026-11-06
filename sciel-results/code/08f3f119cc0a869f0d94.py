"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Generate all 4-element subsets of [1,16]
    all_sets = [frozenset(c) for c in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Precompute intersection masks for fast conflict checking
    # Represent each set as a bitmask of 16 bits
    def to_mask(fs):
        m = 0
        for x in fs:
            m |= (1 << (x - 1))
        return m
    
    masks = [to_mask(s) for s in all_sets]
    
    # Precompute pairwise intersections
    # inter[i][j] = mask of intersection of set i and set j
    # We'll use a dictionary approach: for each pair, store intersection mask
    
    # For the min-degree greedy: when we have a current family F,
    # the "degree" of candidate set S is the number of pairs (A,B) in F
    # such that S would form a sunflower with A and B.
    # S forms a sunflower with A,B iff A∩B = A∩S = B∩S.
    
    # Let's precompute for each pair of sets their intersection mask
    # Then for a new set S, we check: for each pair (A,B) in F,
    #   inter[A][B] == inter[A][S] == inter[B][S]
    
    # To speed this up, let's precompute inter[i][j] for all i,j
    # There are C(16,4) = 1820 sets, so 1820^2/2 ≈ 1.66M pairs
    # Each intersection is a bitwise AND, very fast
    
    # Actually, let's use a different approach for min-degree:
    # For each candidate S, count how many pairs (A,B) in F would conflict.
    # This is O(|F|^2) per candidate, and we have O(n_sets) candidates.
    # Total per iteration: O(n_sets * |F|^2), and |F| grows to ~250.
    # That's O(1820 * 250^2) ≈ 114M operations per full pass. Too slow for many iterations.
    
    # Better approach: maintain a conflict count incrementally.
    # When we add a new set S to F, for each existing set A in F,
    # we can determine which other existing sets B would conflict with S and A.
    # S conflicts with pair (A,B) iff inter[A][B] == inter[A][S] == inter[B][S].
    # For fixed S and A, the condition is: inter[A][B] == inter[A][S] and inter[B][S] == inter[A][S].
    # So B must satisfy: inter[A][B] == inter[A][S] AND inter[B][S] == inter[A][S].
    
    # Let's precompute: for each pair (i,j), store inter_mask = masks[i] & masks[j]
    # And for each i, group j's by inter_mask value.
    
    # Actually, let me think differently. Let's just do the greedy with a good heuristic.
    # Instead of min-degree (which is expensive to compute), let's use:
    # 1. Start with a random seed set
    # 2. At each step, pick the candidate that has the fewest conflicts with existing pairs
    #    But compute this efficiently using precomputed structures.
    
    # Precompute intersection masks for all pairs
    # inter_mask[i][j] = masks[i] & masks[j]
    # Store as a 2D list or dict
    
    # For efficiency, let's use numpy
    import numpy as np
    
    # Create a matrix of masks
    mask_array = np.array(masks, dtype=np.uint32)
    
    # Compute all pairwise intersections
    # inter_matrix[i,j] = mask_array[i] & mask_array[j]
    # This is a 1820x1820 matrix
    inter_matrix = np.zeros((n_sets, n_sets), dtype=np.uint32)
    for i in range(n_sets):
        inter_matrix[i, i:] = mask_array[i] & mask_array[i:]
        inter_matrix[i:, i] = inter_matrix[i, i:]
    
    # Now for the greedy:
    # Maintain a boolean array 'in_family' of size n_sets
    # For each candidate S not in family, compute conflict count:
    #   For each pair (A,B) both in family, check if inter[A,B] == inter[A,S] == inter[B,S]
    # 
    # To speed up: for fixed S, group family members by their intersection with S.
    # Let G_m = {A in family : inter[A,S] = m}
    # Then S conflicts with pair (A,B) iff A,B in same G_m AND inter[A,B] = m.
    # So for each m, we need to count pairs (A,B) in G_m with inter[A,B] = m.
    # This is still O(|family|^2) in the worst case.
    
    # Alternative: just do the straightforward O(|F|^2) check per candidate,
    # but only for a subset of candidates (e.g., top-K by some heuristic).
    
    # Let's try a different strategy:
    # Use the simple greedy but with multiple random restarts and a better ordering.
    # The key insight: sort candidates by how "rare" their elements are,
    # or use a local improvement step after greedy.
    
    best_family = []
    
    def _forms_sunflower_fast(new_idx, family_indices):
        """Check if new_idx forms a sunflower with any two sets in family_indices."""
        new_mask = mask_array[new_idx]
        for i in range(len(family_indices)):
            a_idx = family_indices[i]
            a_mask = mask_array[a_idx]
            ab_mask = inter_matrix[a_idx, new_idx]
            for j in range(i + 1, len(family_indices)):
                b_idx = family_indices[j]
                # Check: inter[a,b] == inter[a,new] == inter[b,new]
                if inter_matrix[a_idx, b_idx] == ab_mask and inter_matrix[b_idx, new_idx] == ab_mask:
                    return True
        return False
    
    def greedy_with_order(order):
        """Greedy construction with given order of candidate indices."""
        family = []
        for idx in order:
            if time.time() > deadline:
                break
            if not _forms_sunflower_fast(idx, family):
                family.append(idx)
        return family
    
    # Try multiple strategies
    all_indices = list(range(n_sets))
    
    # Strategy 1: Lexicographic order
    if time.time() < deadline:
        family = greedy_with_order(all_indices)
        if len(family) > len(best_family):
            best_family = family[:]
    
    # Strategy 2: Multiple random orders
    for attempt in range(20):
        if time.time() > deadline:
            break
        random.shuffle(all_indices)
        family = greedy_with_order(all_indices)
        if len(family) > len(best_family):
            best_family = family[:]
    
    # Strategy 3: Local search improvement
    # Try to add more sets or swap sets to improve
    if time.time() < deadline and len(best_family) > 0:
        family_set = set(best_family)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any set that doesn't conflict
            for idx in all_indices:
                if idx in family_set:
                    continue
                if not _forms_sunflower_fast(idx, best_family):
                    best_family.append(idx)
                    family_set.add(idx)
                    improved = True
                    break
    
    # Strategy 4: Try removing one set and adding two
    if time.time() < deadline and len(best_family) > 1:
        for remove_idx in best_family[:]:
            if time.time() > deadline:
                break
            # Temporarily remove
            best_family.remove(remove_idx)
            family_set = set(best_family)
            # Try to add two sets
            added = []
            for idx in all_indices:
                if idx in family_set or idx == remove_idx:
                    continue
                if not _forms_sunflower_fast(idx, best_family):
                    best_family.append(idx)
                    family_set.add(idx)
                    added.append(idx)
                    if len(added) == 2:
                        break
            if len(added) < 2:
                # Revert
                for idx in added:
                    best_family.remove(idx)
    
    return [sorted(all_sets[idx]) for idx in best_family]
# EVOLVE-BLOCK-END

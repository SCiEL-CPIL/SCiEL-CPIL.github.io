"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # 3s safety margin before 90s kill
    
    # Precompute all 4-element subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n_total = len(all_sets)
    
    # Convert to frozensets for faster intersection
    all_frozensets = [frozenset(s) for s in all_sets]
    
    # Optimize: precompute all pairwise intersections between all sets
    # This allows O(1) lookup instead of computing intersections on the fly
    # intersection_cache[i][j] = frozenset(all_frozensets[i] & all_frozensets[j])
    # But this is 1820^2 = ~3.3M entries, which might be too much memory
    # Instead, let's use a smarter approach
    
    # For the sunflower check when adding set S:
    # We need to check if there exist A, B in chosen such that A∩B = A∩S = B∩S
    # Key insight: A∩S is determined by which elements of S are in A
    # Since S has only 4 elements, there are at most 2^4 = 16 possible intersections
    
    # Better approach: For each candidate S, for each existing A, compute I = A∩S
    # Group by I, then for each group, check if any pair has intersection I
    # To speed up the pair check, precompute for each possible intersection value I,
    # maintain a data structure that helps check if any two sets in the group have intersection I
    
    # Even better: since sets are 4-element subsets of [1,16], we can encode them as integers
    # and use bit operations for fast intersection
    
    def encode_set(s):
        """Encode a frozenset of elements from 1-16 as an integer bitmask"""
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        return mask
    
    all_masks = [encode_set(s) for s in all_frozensets]
    
    def intersect_masks(m1, m2):
        """Return the intersection of two bitmask-encoded sets as a bitmask"""
        return m1 & m2
    
    def masks_equal(m1, m2):
        """Check if two bitmask-encoded sets are equal"""
        return m1 == m2
    
    best_family = []
    best_size = 0
    
    # Try multiple random orderings
    max_trials = 50
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = list(range(n_total))
        else:
            order = list(range(n_total))
            random.seed(trial * 7919 + 104729)
            random.shuffle(order)
        
        chosen_masks = []
        chosen_indices = []
        
        # For efficient sunflower checking, maintain:
        # - For each possible intersection pattern with a new set S, track which chosen sets have that pattern
        # - But this depends on S, so we compute on the fly
        
        for idx in order:
            if time.time() > deadline:
                break
            
            s_mask = all_masks[idx]
            
            # Check if s can be added
            can_add = True
            m = len(chosen_masks)
            
            if m >= 2:
                # Group chosen sets by their intersection with s
                groups = {}
                for i in range(m):
                    inter = intersect_masks(chosen_masks[i], s_mask)
                    if inter not in groups:
                        groups[inter] = [i]
                    else:
                        groups[inter].append(i)
                
                # For each group, check if any pair has intersection equal to the group key
                for inter, indices in groups.items():
                    if len(indices) >= 2:
                        # Check all pairs in this group
                        n_idx = len(indices)
                        found = False
                        for j in range(n_idx):
                            if found:
                                break
                            for k in range(j + 1, n_idx):
                                a_mask = chosen_masks[indices[j]]
                                b_mask = chosen_masks[indices[k]]
                                if masks_equal(intersect_masks(a_mask, b_mask), inter):
                                    can_add = False
                                    found = True
                                    break
                    if not can_add:
                        break
            
            if can_add:
                chosen_masks.append(s_mask)
                chosen_indices.append(idx)
        
        if len(chosen_masks) > best_size:
            best_size = len(chosen_masks)
            best_family = list(chosen_indices)
    
    # Convert indices back to sorted lists
    result = [sorted(all_sets[i]) for i in best_family]
    
    # Quick verification using bitmask approach (much faster than frozenset intersection)
    def is_valid_bitmask(family_indices):
        n = len(family_indices)
        masks = [all_masks[i] for i in family_indices]
        for i in range(n):
            for j in range(i + 1, n):
                ab = masks[i] & masks[j]
                for k in range(j + 1, n):
                    ac = masks[i] & masks[k]
                    bc = masks[j] & masks[k]
                    if ab == ac == bc:
                        return False
        return True
    
    if not is_valid_bitmask(best_family):
        # Fallback: rebuild with a simple greedy
        chosen = []
        chosen_masks = []
        for i in range(n_total):
            s_mask = all_masks[i]
            can_add = True
            m = len(chosen_masks)
            if m >= 2:
                groups = {}
                for j in range(m):
                    inter = chosen_masks[j] & s_mask
                    if inter not in groups:
                        groups[inter] = [j]
                    else:
                        groups[inter].append(j)
                for inter, indices in groups.items():
                    if len(indices) >= 2:
                        n_idx = len(indices)
                        for j in range(n_idx):
                            for k in range(j + 1, n_idx):
                                if (chosen_masks[indices[j]] & chosen_masks[indices[k]]) == inter:
                                    can_add = False
                                    break
                            if not can_add:
                                break
                    if not can_add:
                        break
            if can_add:
                chosen.append(i)
                chosen_masks.append(s_mask)
        result = [sorted(all_sets[i]) for i in chosen]
    
    return result
# EVOLVE-BLOCK-END

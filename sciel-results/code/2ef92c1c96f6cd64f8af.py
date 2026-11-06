"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Precompute all 4-element subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n_total = len(all_sets)
    
    # Convert to frozensets for faster intersection
    all_frozensets = [frozenset(s) for s in all_sets]
    
    # Greedy construction with optimized sunflower check
    # A sunflower of 3 sets A, B, C exists if:
    # A∩B = A∩C = B∩C (all three pairwise intersections equal)
    
    # For speed, we maintain a list of chosen sets and for each new candidate,
    # check against all pairs of existing sets. To optimize, we can use the fact that
    # if we're adding set S, we need to check for any pair (A, B) in chosen such that
    # A∩B = A∩S = B∩S.
    
    # Alternative approach: For a given S, compute A∩S and B∩S for all A, B in chosen.
    # This is still O(m^2) per candidate.
    
    # Let's try a different optimization: for each candidate S, and for each existing A,
    # compute I_A = A ∩ S. Then for the pair (A, B) to form a sunflower with S,
    # we need B ∩ S = I_A AND A ∩ B = I_A.
    # So we can group existing sets by their intersection with S.
    # For each group G (sets with same intersection I with S), check if any pair (A, B) in G
    # has A ∩ B = I.
    
    chosen = []
    chosen_fs = []  # frozensets for speed
    
    # Shuffle to get different results on different runs? No, keep deterministic for reproducibility.
    # Actually, let's try a few different orderings and keep the best.
    
    best_family = []
    
    for trial in range(10):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = list(range(n_total))
        else:
            order = list(range(n_total))
            random.seed(trial * 12345 + 67890)
            random.shuffle(order)
        
        chosen = []
        chosen_fs = []
        
        for idx in order:
            if time.time() > deadline:
                break
            
            s = all_frozensets[idx]
            
            # Check if s can be added
            can_add = True
            
            if len(chosen_fs) >= 2:
                # Group existing sets by their intersection with s
                groups = {}
                for i, a in enumerate(chosen_fs):
                    inter = a & s
                    if inter not in groups:
                        groups[inter] = [i]
                    else:
                        groups[inter].append(i)
                
                # For each group, check if any pair has intersection equal to the group key
                for inter, indices in groups.items():
                    if len(indices) >= 2:
                        # Check pairs in this group
                        for j in range(len(indices)):
                            for k in range(j + 1, len(indices)):
                                a = chosen_fs[indices[j]]
                                b = chosen_fs[indices[k]]
                                if a & b == inter:
                                    can_add = False
                                    break
                            if not can_add:
                                break
                    if not can_add:
                        break
            
            if can_add:
                chosen.append(s)
                chosen_fs.append(s)
        
        if len(chosen) > len(best_family):
            best_family = list(chosen)
    
    # If we still have time, try to improve by local search or different strategies
    # For now, return the best family found
    
    # Convert frozensets back to sorted lists
    result = [sorted(s) for s in best_family]
    
    # Verify the result is valid (no sunflowers)
    # This is a sanity check; should be fast for reasonable family sizes
    def is_valid(family):
        n = len(family)
        # Convert to sets for intersection operations if they are lists
        fs_family = [frozenset(s) for s in family]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = fs_family[i], fs_family[j], fs_family[k]
                    ab = a & b
                    ac = a & c
                    bc = b & c
                    if ab == ac == bc:
                        return False
        return True
    
    if not is_valid(result):
        # Fallback: use a simple valid family
        # Just return the first valid greedy result
        chosen = []
        for s in all_frozensets:
            can_add = True
            for a in chosen:
                for b in chosen:
                    if a is b:
                        continue
                    ab = a & b
                    ac = a & s
                    bc = b & s
                    if ab == ac == bc:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                chosen.append(s)
        result = [sorted(s) for s in chosen]
    
    return result
# EVOLVE-BLOCK-END

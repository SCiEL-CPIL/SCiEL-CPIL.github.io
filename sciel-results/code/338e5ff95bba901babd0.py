"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Generate all 4-element subsets of [1,16]
    all_sets = [frozenset(c) for c in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Precompute intersection masks
    def to_mask(fs):
        m = 0
        for x in fs:
            m |= (1 << (x - 1))
        return m
    
    masks = [to_mask(s) for s in all_sets]
    mask_array = np.array(masks, dtype=np.uint32)
    
    # Compute all pairwise intersections
    inter_matrix = np.zeros((n_sets, n_sets), dtype=np.uint32)
    for i in range(n_sets):
        inter_matrix[i, i:] = mask_array[i] & mask_array[i:]
        inter_matrix[i:, i] = inter_matrix[i, i:]
    
    def _forms_sunflower_fast(new_idx, family_indices):
        """Check if new_idx forms a sunflower with any two sets in family_indices."""
        for i in range(len(family_indices)):
            a_idx = family_indices[i]
            ab_mask = inter_matrix[a_idx, new_idx]
            for j in range(i + 1, len(family_indices)):
                b_idx = family_indices[j]
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
    
    # Fast greedy using incremental conflict counts
    def greedy_fast(order):
        """
        Greedy with O(N) per insertion using incremental conflict counting.
        
        conflict_count[s] = number of pairs (a,b) in family such that s would
        form a sunflower with a and b.
        
        When we add t to family, for each candidate s not in family:
          new_conflicts += count of a in family (excluding t) such that
          inter[a][t] == inter[a][s] == inter[t][s]
          i.e., inter[a][t] == inter[a][s] AND inter[t][s] == inter[a][t]
        
        For fixed s and t: inter[t][s] is fixed. Let m = inter[t][s].
        We need a in family (a != t) such that inter[a][t] == m AND inter[a][s] == m.
        So: a must have inter[a][t] == m AND inter[a][s] == m.
        
        We can precompute for each pair (t,s): the set of a's where both conditions hold.
        But that's expensive. Instead, for each t being added, iterate over all s
        and count how many existing a's satisfy the condition.
        
        Actually, let's do it more simply:
        When adding t, for each candidate s (not in family, s != t):
          delta = 0
          for a in family (a != t):
            if inter[a][t] == inter[a][s] == inter[t][s]:
              delta += 1
          conflict_count[s] += delta
          Also, t itself gets conflicts with pairs involving s... wait no.
          Actually we also need to update conflict_count for t? No, t is being added.
          But we need to check if t has 0 conflicts before adding.
        
        Let me restructure:
        - Maintain conflict_count[s] for all s.
        - At each step, pick the s with conflict_count[s] == 0 (and not in family).
        - When adding s, update conflict_count for all other candidates.
        """
        in_family = [False] * n_sets
        family = []
        # conflict_count[s] = number of pairs in family that would form sunflower with s
        conflict_count = [0] * n_sets
        
        for idx in order:
            if time.time() > deadline:
                break
            if in_family[idx]:
                continue
            if conflict_count[idx] == 0:
                # Add idx to family
                in_family[idx] = True
                family.append(idx)
                # Update conflict counts for all other candidates
                # For each candidate s not in family (s != idx):
                #   new pairs formed: (idx, a) for each a in family (a != idx)
                #   s conflicts with pair (idx, a) iff inter[idx][a] == inter[idx][s] == inter[a][s]
                #   i.e., inter[idx][a] == inter[idx][s] AND inter[a][s] == inter[idx][s]
                for s in range(n_sets):
                    if in_family[s] or s == idx:
                        continue
                    is_val = inter_matrix[idx, s]
                    for a in family:
                        if a == idx:
                            continue
                        if inter_matrix[idx, a] == is_val and inter_matrix[a, s] == is_val:
                            conflict_count[s] += 1
        return family
    
    best_family = []
    all_indices = list(range(n_sets))
    
    # Strategy 1: Fast greedy with lexicographic order
    if time.time() < deadline:
        family = greedy_fast(all_indices)
        if len(family) > len(best_family):
            best_family = family[:]
    
    # Strategy 2: Multiple random orders with fast greedy
    for attempt in range(30):
        if time.time() > deadline:
            break
        random.shuffle(all_indices)
        family = greedy_fast(all_indices)
        if len(family) > len(best_family):
            best_family = family[:]
    
    # Strategy 3: Local search - try to add any set that doesn't conflict
    if time.time() < deadline and len(best_family) > 0:
        family_set = set(best_family)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in all_indices:
                if idx in family_set:
                    continue
                if not _forms_sunflower_fast(idx, best_family):
                    best_family.append(idx)
                    family_set.add(idx)
                    improved = True
                    break
    
    # Strategy 4: Swap-based local search
    if time.time() < deadline and len(best_family) > 1:
        family_set = set(best_family)
        swapped = True
        while swapped and time.time() < deadline:
            swapped = False
            for ri in range(len(best_family)):
                if time.time() > deadline:
                    break
                remove_idx = best_family[ri]
                # Try removing remove_idx and see if we can add 2+ new sets
                temp_family = best_family[:ri] + best_family[ri+1:]
                temp_set = set(temp_family)
                added = []
                for idx in all_indices:
                    if idx in temp_set or idx == remove_idx:
                        continue
                    if not _forms_sunflower_fast(idx, temp_family):
                        temp_family.append(idx)
                        temp_set.add(idx)
                        added.append(idx)
                        if len(added) >= 2:
                            break
                if len(added) >= 2:
                    best_family = temp_family
                    family_set = temp_set
                    swapped = True
                else:
                    # Revert (nothing changed since we used temp)
                    pass
    
    return [sorted(all_sets[idx]) for idx in best_family]
# EVOLVE-BLOCK-END

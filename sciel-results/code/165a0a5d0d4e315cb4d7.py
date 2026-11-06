# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin
    
    # Generate all 4-element subsets of [1,16] as bitmasks
    all_sets = []
    mask_to_idx = {}
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        idx = len(all_sets)
        all_sets.append(mask)
        mask_to_idx[mask] = idx
    
    n_all = len(all_sets)  # 1820
    
    # Precompute: for each pair (i,j), check if there exists k such that 
    # all_sets[i], all_sets[j], all_sets[k] form a sunflower
    # A sunflower: A∩B = A∩C = B∩C
    # For pair (i,j), the intersection I = all_sets[i] & all_sets[j]
    # A set k forms a sunflower with i,j iff:
    #   all_sets[k] & all_sets[i] == I AND all_sets[k] & all_sets[j] == I
    # which means all_sets[k] contains I and all_sets[k] ∩ (all_sets[i] \ I) == 0 
    # and all_sets[k] ∩ (all_sets[j] \ I) == 0
    
    # For efficiency: for each pair (i,j), compute I = sets[i] & sets[j]
    # Then sets[k] conflicts iff (sets[k] & I) == I and (sets[k] & (sets[i]^I)) == 0 and (sets[k] & (sets[j]^I)) == 0
    # i.e., sets[k] ⊇ I and sets[k] ∩ (sets[i] ∪ sets[j] \ I) == 0
    # i.e., sets[k] ⊇ I and sets[k] ∩ (sets[i] | sets[j]) == I
    
    # Let's precompute for each set k, the list of pairs (i,j) that it would conflict with
    # Actually, let's precompute conflict_pairs[k] = list of (i,j) pairs such that {i,j,k} is a sunflower
    
    # Better approach: for each k, find all pairs (i,j) with i<j, i≠k, j≠k
    # such that sets[i]&sets[j] == sets[i]&sets[k] == sets[j]&sets[k]
    
    # Even better: for each k, and for each possible intersection I that k participates in,
    # find other sets that share exactly I with k
    
    # Let me think differently. For a fixed k, a pair (i,j) conflicts with k iff:
    #   I_ij = sets[i] & sets[j]
    #   sets[i] & sets[k] == I_ij
    #   sets[j] & sets[k] == I_ij
    # This means I_ij ⊆ sets[k], and sets[i] ∩ (sets[k]\I_ij) = 0, sets[j] ∩ (sets[k]\I_ij) = 0
    
    # Precompute: for each k, for each subset I of sets[k] (there are at most 2^4=16 subsets),
    # find all sets i such that sets[i] & sets[k] == I
    # Then for each pair of such sets (i,j) where sets[i] & sets[j] == I, we have a conflict
    
    # This is getting complex. Let me just precompute all sunflower triples directly.
    # There are C(1820,3) ≈ 1 billion triples, too many.
    
    # Alternative: for each pair (i,j), compute I = sets[i] & sets[j]
    # Find all k such that sets[k] & sets[i] == I and sets[k] & sets[j] == I
    # Store: conflicts[i][j] = set of k values, and conflicts[k].add((i,j))
    
    # But storing all this might use too much memory. Let me use a different approach.
    
    # For the greedy, when adding set k to family F, I need to check if there exist i,j in F
    # such that {i,j,k} is a sunflower.
    
    # Key optimization: maintain a set of "active pairs" - pairs (i,j) in F.
    # When adding k, check if any active pair (i,j) has k in its conflict set.
    # But I need to store conflict sets.
    
    # Let me precompute: for each k, the set of pairs (i,j) (with i<j, i≠k, j≠k) 
    # such that {i,j,k} is a sunflower. Store as a set of frozenset({i,j}).
    
    # Memory: 1820 sets, each with potentially many pairs. Let's estimate.
    # For a random k, how many pairs (i,j) conflict? Probably not too many.
    
    # Actually, let me just precompute it. 1820 * 1820 / 2 ≈ 1.6M pairs, 
    # for each pair compute I and find conflicts. This is O(n^3) ≈ 6 billion, too slow.
    
    # Let me use the subset approach:
    # For each k, enumerate all 16 subsets I of sets[k].
    # For each I, find all sets i (i≠k) such that sets[i] & sets[k] == I.
    # Call this group G_I.
    # For each pair (i,j) in G_I with i<j, check if sets[i] & sets[j] == I.
    # If yes, then (i,j) conflicts with k.
    
    # This is O(n * 16 * |G_I|^2) in the worst case, but |G_I| is small on average.
    
    # Let me try this approach.
    
    # Precompute groups: for each k and each I ⊆ sets[k], the list of i with sets[i]&sets[k]==I
    # Actually, let me just precompute conflict_pairs[k] directly.
    
    conflict_pairs = [set() for _ in range(n_all)]
    
    # For each k, for each proper subset I of sets[k] (including empty):
    #   Find all i≠k with sets[i] & sets[k] == I
    #   For each pair (i,j) in that group, if sets[i] & sets[j] == I, add (i,j) to conflict_pairs[k]
    
    # To find sets[i] & sets[k] == I efficiently, note that I must be a subset of sets[k].
    # sets[k] has exactly 4 bits set. So there are 16 possible I values.
    
    # For each k, build a dict: intersection_value -> list of i
    # Then for each I, check pairs within the list.
    
    # This is O(n * 16 + n * sum_I |G_I|^2). The sum over I of |G_I| = n-1, 
    # but sum of |G_I|^2 could be larger. In practice, most G_I are small.
    
    # Let me just do it.
    
    if time.time() > deadline - 5:
        # Not enough time for precomputation, fall back to simple greedy
        pass
    else:
        # Precompute conflict_pairs
        # For each k, group other sets by their intersection with k
        for k in range(n_all):
            sk = all_sets[k]
            # Group: intersection_mask -> list of indices
            groups = {}
            for i in range(n_all):
                if i == k:
                    continue
                inter = sk & all_sets[i]
                if inter not in groups:
                    groups[inter] = []
                groups[inter].append(i)
            
            # For each group I, check pairs
            for I, idx_list in groups.items():
                if len(idx_list) < 2:
                    continue
                for a_pos in range(len(idx_list)):
                    i = idx_list[a_pos]
                    for b_pos in range(a_pos + 1, len(idx_list)):
                        j = idx_list[b_pos]
                        if all_sets[i] & all_sets[j] == I:
                            # (i,j) conflicts with k
                            pair = (min(i,j), max(i,j))
                            conflict_pairs[k].add(pair)
    
    # Now do greedy construction with fast conflict checking
    best_family = []
    best_size = 0
    
    for attempt in range(5000):
        if time.time() > deadline - 5:
            break
        
        if attempt == 0:
            candidates = list(range(n_all))
        elif attempt % 3 == 1:
            candidates = list(range(n_all))
            random.shuffle(candidates)
        else:
            # Sort by some heuristic - maybe by number of conflicts (fewer conflicts first)
            candidates = sorted(range(n_all), key=lambda x: len(conflict_pairs[x]))
            random.shuffle(candidates)  # add some randomness
        
        family = []
        family_set = set()
        active_pairs = set()  # pairs (i,j) in family
        
        for k in candidates:
            if time.time() > deadline - 3:
                break
            
            # Check if k conflicts with any active pair
            # k conflicts if there's a pair (i,j) in active_pairs such that (i,j) in conflict_pairs[k]
            # i.e., if active_pairs & conflict_pairs[k] is non-empty
            if active_pairs & conflict_pairs[k]:
                continue
            
            # Add k to family
            # Update active_pairs: for each existing member i in family, add (i,k)
            for i in family:
                pair = (min(i,k), max(i,k))
                active_pairs.add(pair)
            
            family.append(k)
            family_set.add(k)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
            if best_size >= 250:
                break
    
    # Local search: try to add more sets or swap
    if time.time() < deadline - 3 and best_family:
        fam_set = set(best_family)
        active_pairs = set()
        for i in range(len(best_family)):
            for j in range(i+1, len(best_family)):
                active_pairs.add((min(best_family[i], best_family[j]), max(best_family[i], best_family[j])))
        
        # Try adding sets not in family
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for k in range(n_all):
                if k in fam_set:
                    continue
                if not (active_pairs & conflict_pairs[k]):
                    # Can add k
                    for i in best_family:
                        active_pairs.add((min(i,k), max(i,k)))
                    best_family.append(k)
                    fam_set.add(k)
                    improved = True
                    break
        
        # Try swaps: remove one, add two
        if time.time() < deadline - 1:
            improved = True
            while improved and time.time() < deadline - 0.5:
                improved = False
                for rm in best_family[:]:
                    if time.time() > deadline - 0.5:
                        break
                    # Try removing rm and adding two new sets
                    # This is expensive, so limit attempts
                    test_fam = [x for x in best_family if x != rm]
                    test_set = set(test_fam)
                    test_pairs = set()
                    for i in range(len(test_fam)):
                        for j in range(i+1, len(test_fam)):
                            test_pairs.add((min(test_fam[i], test_fam[j]), max(test_fam[i], test_fam[j])))
                    
                    added = 0
                    for k in range(n_all):
                        if k in test_set or k == rm:
                            continue
                        if not (test_pairs & conflict_pairs[k]):
                            # Can add k
                            for i in test_fam:
                                test_pairs.add((min(i,k), max(i,k)))
                            test_fam.append(k)
                            test_set.add(k)
                            added += 1
                            if added >= 2:
                                break
                    
                    if added >= 2:
                        best_family = test_fam
                        improved = True
                        break
    
    # Convert best_family indices back to lists of sorted integers
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = []
        for bit in range(16):
            if mask & (1 << bit):
                elems.append(bit + 1)
        result.append(elems)
        
    return result
# EVOLVE-BLOCK-END

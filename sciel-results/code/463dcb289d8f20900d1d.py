import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    
    n = 16
    k = 4
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for comb in combinations(range(n), k):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_sets.append(mask)
    
    num_sets = len(all_sets)
    
    # Precompute pairwise intersection matrix
    # inter[i][j] = all_sets[i] & all_sets[j]
    # We need to check: for any three distinct i, j, k, 
    # NOT (inter[i][j] == inter[i][k] == inter[j][k] AND (all_sets[i] & all_sets[j]) == inter[i][j])
    # Actually the condition is: A∩B = A∩C = B∩C
    # Which means inter[i][j] == inter[i][k] == inter[j][k]
    
    # For efficient checking, we can precompute for each pair (i,j) the intersection value.
    # When adding a new set s, we need to check against all pairs (a,b) in family:
    #   inter[s][a] == inter[s][b] == inter[a][b]
    
    # Let's use a different approach: for each candidate set s, check against all pairs in family.
    # But that's O(|F|^2) per candidate.
    
    # Better approach: For each set a in family, compute inter[s][a]. 
    # Then for each pair of sets (a,b) in family, check if inter[s][a] == inter[s][b] == inter[a][b].
    # This is still O(|F|^2) per candidate.
    
    # Alternative: For each set a in family, group other sets by their intersection with a.
    # For a new set s, for each a in family, look up inter[s][a] in the groups of a.
    # If there's already a set b in family with inter[a][b] == inter[s][a], then we need to check
    # if inter[s][b] == inter[s][a] as well.
    
    # Let me think about this more carefully.
    # The sunflower condition for three sets A, B, C is: A∩B = A∩C = B∩C.
    # Let x = A∩B = A∩C = B∩C.
    # So for a new set S, and two existing sets A, B in family:
    #   S∩A = S∩B = A∩B
    # This means S∩A = A∩B, so S∩A is exactly the intersection of A and B.
    # Equivalently: (S∩A) == (A∩B) AND (S∩B) == (A∩B)
    # Which is: S∩A == A∩B AND S∩B == A∩B
    # Note that S∩A == A∩B implies S∩A ⊆ B (since A∩B ⊆ B) and S∩A ⊆ A (trivially).
    # Actually S∩A == A∩B means the elements of S that are in A are exactly the elements of A that are in B.
    # So S∩A = A∩B. This is a strong condition.
    
    # For efficient checking: for each existing set A in family, compute S∩A.
    # Then for each other set B in family, check if S∩A == A∩B and S∩B == A∩B.
    # But S∩B == A∩B is automatically true if S∩A == A∩B? No, not necessarily.
    # We need both conditions.
    
    # Let's precompute inter[i][j] for all pairs.
    # Then for a new set s (index in all_sets), and family members (list of indices):
    #   For each pair (a, b) in family:
    #     if inter[s][a] == inter[a][b] and inter[s][b] == inter[a][b]:
    #       conflict!
    
    # Precompute the full intersection matrix
    inter = np.zeros((num_sets, num_sets), dtype=np.int32)
    for i in range(num_sets):
        for j in range(i+1, num_sets):
            val = all_sets[i] & all_sets[j]
            inter[i][j] = val
            inter[j][i] = val
    
    # Now use greedy with local search
    random.shuffle(all_sets)
    
    family = []
    family_set = set()
    
    # Greedy initialization
    for s_idx in range(num_sets):
        s = all_sets[s_idx]
        valid = True
        for i in range(len(family)):
            a = family[i]
            sa = s & a
            for j in range(i+1, len(family)):
                b = family[j]
                sb = s & b
                ab = a & b
                if sa == ab and sb == ab:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            family.append(s)
            family_set.add(s)
    
    best_family = list(family)
    best_size = len(family)
    
    # Local search with swaps
    # To make this faster, we can precompute for each set in family, 
    # and use the inter matrix for quick lookups.
    
    # Map from mask to index in all_sets for quick lookup
    mask_to_idx = {m: i for i, m in enumerate(all_sets)}
    
    while time.time() < deadline:
        if len(family) < 2:
            break
        
        # Pick random index in family
        idx_a = random.randint(0, len(family) - 1)
        a = family[idx_a]
        a_idx = mask_to_idx[a]
        
        # Pick random candidate not in family
        b = None
        for _ in range(10):
            cand = random.choice(all_sets)
            if cand not in family_set:
                b = cand
                break
        if b is None:
            continue
        
        b_idx = mask_to_idx[b]
        
        # Check if swap is valid: remove a, add b
        # Need to check that b doesn't form a sunflower with any two other sets in family (excluding a)
        valid = True
        for i in range(len(family)):
            if i == idx_a:
                continue
            ai = family[i]
            ai_idx = mask_to_idx[ai]
            ba = b & ai
            for j in range(i+1, len(family)):
                if j == idx_a:
                    continue
                aj = family[j]
                aj_idx = mask_to_idx[aj]
                bj = b & aj
                iaj = ai & aj
                if ba == iaj and bj == iaj:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            family[idx_a] = b
            family_set.remove(a)
            family_set.add(b)
            
            if len(family) > best_size:
                best_size = len(family)
                best_family = list(family)
    
    # Convert bitmasks back to sorted lists of 1-based integers
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result

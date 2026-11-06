import time
import random
from itertools import combinations
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85  # safety margin (90s limit)
    
    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    masks_arr = np.array(all_masks, dtype=np.int64)
    
    # Precompute pairwise intersections matrix (1820x1820) - might be too big
    # 1820*1820 = 3.3M entries, each 8 bytes = 26MB. OK.
    # But we don't need the full matrix always.
    
    def is_valid_addition(C, family_list):
        """Check if C can be added to family (list of masks)."""
        n = len(family_list)
        if n < 2:
            return True
        fam = np.array(family_list, dtype=np.int64)
        # For each pair (i,j), check if (A&C) == (A&B) and (B&C) == (A&B)
        # A&B is fam[i] & fam[j]
        # We need: for all i<j: (fam[i]&C) == (fam[i]&fam[j]) AND (fam[j]&C) == (fam[i]&fam[j])
        # This is O(n^2) but vectorized per pair... still slow for large n.
        
        # Alternative: iterate
        for i in range(n):
            A = family_list[i]
            for j in range(i+1, n):
                B = family_list[j]
                I = A & B
                if (A & C) == I and (B & C) == I:
                    return False
        return True
    
    # Better: precompute for the current family, the set of forbidden masks.
    # When we add a new set, we update forbidden set by adding all C that form sunflower with any pair involving the new set.
    
    # Let's use a different approach: maintain forbidden set as a Python set of masks.
    # Initially empty. When family has sets F = {f1, f2, ..., fk}, forbidden contains all C such that exists i<j with fi∩fj = fi∩C = fj∩C.
    
    # To update: when adding new set X to family, new forbidden = old forbidden ∪ {C : exists Y in old_family with Y∩X = Y∩C = X∩C}
    
    # For a fixed pair (Y, X), the forbidden C's are: C = (Y∩X) ∪ T where T ⊆ [16]\(Y∪X), |T| = 4 - |Y∩X|
    
    def get_forbidden_for_pair(A, B):
        """Return all C such that A∩B = A∩C = B∩C, with |C|=4."""
        S = A & B
        s_size = bin(S).count('1')
        remaining = 4 - s_size
        if remaining < 0 or remaining > 16:
            return []
        # C must contain S, and C\S is a subset of [16]\(A|B) of size 'remaining'
        complement = ~(A | B) & 0xFFFF  # 16 bits
        # Find all subsets of complement of size 'remaining'
        # Get list of bits in complement
        bits = [i for i in range(16) if complement & (1 << i)]
        result = []
        for combo in combinations(bits, remaining):
            T = 0
            for b in combo:
                T |= (1 << b)
            C = S | T
            result.append(C)
        return result
    
    # Strategy: greedy with random restarts, maintaining forbidden set incrementally.
    
    best_family = []
    best_size = 0
    
    for trial in range(50):
        if time.time() > deadline - 15:
            break
        
        random.shuffle(all_masks)
        family = []
        forbidden = set()  # masks that cannot be added
        
        for mask in all_masks:
            if time.time() > deadline - 15:
                break
            if mask in forbidden:
                continue
            # Check: is mask valid? It's valid if not in forbidden.
            # But forbidden was built from pairs in current family.
            # Adding mask: new forbidden = forbidden ∪ (forbidden from pairs (mask, f) for f in family)
            family.append(mask)
            # Update forbidden
            for f in family[:-1]:
                for C in get_forbidden_for_pair(mask, f):
                    if C not in family:  # only add if not already in family (but we track separately)
                        forbidden.add(C)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Local search: try swaps
    family = best_family[:]
    current_size = len(family)
    
    # Rebuild forbidden set for current family
    forbidden = set()
    for i in range(len(family)):
        for j in range(i+1, len(family)):
            for C in get_forbidden_for_pair(family[i], family[j]):
                forbidden.add(C)
    
    # Try to add any mask not in family and not forbidden
    added = True
    while added and time.time() < deadline - 10:
        added = False
        for mask in all_masks:
            if mask not in family and mask not in forbidden:
                family.append(mask)
                # Update forbidden
                for f in family[:-1]:
                    for C in get_forbidden_for_pair(mask, f):
                        forbidden.add(C)
                added = True
                current_size = len(family)
    
    # Swap-based local search with simulated annealing
    if time.time() < deadline - 5:
        T = 1.0
        cooling = 0.9995
        family_set = set(family)
        
        while time.time() < deadline - 2 and T > 0.01:
            # Pick a random mask to remove
            idx_out = random.randint(0, len(family)-1)
            mask_out = family[idx_out]
            
            # Pick a random mask to add (not in family)
            mask_in = random.choice(all_masks)
            attempts = 0
            while mask_in in family_set and attempts < 50:
                mask_in = random.choice(all_masks)
                attempts += 1
            if mask_in in family_set:
                continue
            
            # Try the swap: remove mask_out, add mask_in
            # Check if mask_in is valid with family minus mask_out
            # Build new forbidden from family without mask_out, then check mask_in
            fam_temp = family[:idx_out] + family[idx_out+1:]
            
            # Check if mask_in can be added to fam_temp
            valid = True
            for i in range(len(fam_temp)):
                A = fam_temp[i]
                for j in range(i+1, len(fam_temp)):
                    B = fam_temp[j]
                    I = A & B
                    if (A & mask_in) == I and (B & mask_in) == I:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Accept the swap
                family = fam_temp + [mask_in]
                family_set.remove(mask_out)
                family_set.add(mask_in)
                current_size = len(family)
                
                # Rebuild forbidden (expensive but necessary for correctness)
                forbidden = set()
                for i in range(len(family)):
                    for j in range(i+1, len(family)):
                        for C in get_forbidden_for_pair(family[i], family[j]):
                            forbidden.add(C)
            
            T *= cooling
        
        # Final attempt to add
        added = True
        while added and time.time() < deadline - 1:
            added = False
            for mask in all_masks:
                if mask not in family_set and mask not in forbidden:
                    family.append(mask)
                    family_set.add(mask)
                    for f in family[:-1]:
                        for C in get_forbidden_for_pair(mask, f):
                            forbidden.add(C)
                    added = True
                    current_size = len(family)
    
    # Convert back to sorted lists
    result = []
    for mask in family:
        elements = [i+1 for i in range(16) if mask & (1 << i)]
        result.append(elements)
    
    return result

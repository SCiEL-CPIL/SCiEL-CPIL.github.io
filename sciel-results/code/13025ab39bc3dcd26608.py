import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    total_masks = len(all_masks)
    
    # Precompute conflict graph: two sets conflict if they can form a sunflower with some third set
    # A sunflower of 3 sets A, B, C has all pairwise intersections equal.
    # So A and B conflict if there exists C such that A∩B = A∩C = B∩C.
    # For given A, B, the common intersection must be I = A∩B, and C must contain I and
    # C∩A = I, C∩B = I, meaning C contains I but no other elements from A\I or B\I.
    # So C must be a 4-set containing I with C ⊆ I ∪ ([n] \ (A∪B)).
    
    # Build conflict adjacency as bitsets for speed
    conflict = [0] * total_masks  # bitmask of conflicting indices
    
    # For each pair (A, B), check if they conflict
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            A = all_masks[i]
            B = all_masks[j]
            I = A & B  # required common core
            
            # C must be a 4-set with I ⊆ C, C∩A = I, C∩B = I
            # So C contains all of I, and the remaining (4 - popcount(I)) elements
            # must come from [n] \ (A ∪ B)
            i_count = bin(I).count('1')
            if i_count > 4:
                continue
            
            remaining_slots = 4 - i_count
            if remaining_slots < 0:
                continue
            
            # Available elements for C's non-core part
            forbidden = A | B  # elements in A or B that are not in I
            available_mask = ((1 << n) - 1) & ~forbidden
            # But I elements are already in C, so we need remaining_slots from available_mask
            # Wait, I ⊆ A∪B, so I elements are in forbidden. Let me reconsider.
            
            # C must contain I. The other (4 - i_count) elements of C must not be in A\I or B\I.
            # A\I = A & ~I, B\I = B & ~I
            # So forbidden for additional elements = (A & ~I) | (B & ~I)
            # Available = [n] \ ((A & ~I) | (B & ~I))
            # But C already includes I, so from available we pick (4 - i_count) elements.
            
            forbidden_extra = (A & ~I) | (B & ~I)
            available_extra = ((1 << n) - 1) & ~forbidden_extra
            # Remove I from available since those are already in C
            available_extra &= ~I
            
            # Check if there are at least remaining_slots available elements
            if bin(available_extra).count('1') < remaining_slots:
                continue
            
            # There exists at least one valid C, so A and B conflict
            conflict[i] |= (1 << j)
            conflict[j] |= (1 << i)
    
    # Greedy maximal independent set with randomized ordering
    best_family = []
    best_size = 0
    
    def greedy_mis(order):
        """Greedy maximal independent set: add node if no neighbor already in set."""
        selected = []
        selected_set = set()
        for idx in order:
            # Check if any neighbor is already selected
            c = conflict[idx]
            # Check if c has any bit set that corresponds to selected nodes
            # Use bitset intersection
            sel_bits = 0
            for s in selected_set:
                sel_bits |= (1 << s)
            if (c & sel_bits) == 0:
                selected.append(idx)
                selected_set.add(idx)
        return selected
    
    # Phase 1: Multiple greedy MIS with different orderings
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        if attempt % 4 == 0:
            order = list(range(total_masks))
            random.shuffle(order)
        elif attempt % 4 == 1:
            # Prefer low-degree nodes first (fewer conflicts)
            order = sorted(range(total_masks), key=lambda i: bin(conflict[i]).count('1') + random.random())
        elif attempt % 4 == 2:
            # Prefer high-degree nodes first
            order = sorted(range(total_masks), key=lambda i: -bin(conflict[i]).count('1') + random.random())
        else:
            # Random weighted by inverse degree
            weights = [1.0 / (1 + bin(conflict[i]).count('1')) for i in range(total_masks)]
            order = list(range(total_masks))
            # Sample without replacement weighted
            for i in range(total_masks - 1, 0, -1):
                total_w = sum(weights[j] for j in range(i + 1))
                r = random.random() * total_w
                cumulative = 0
                for j in range(i + 1):
                    cumulative += weights[j]
                    if cumulative >= r:
                        order[i], order[j] = order[j], order[i]
                        break
        
        family = greedy_mis(order)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            if best_size >= 250:
                break
    
    # Phase 2: Local search - try swaps to improve
    if time.time() < deadline - 5 and best_size > 0:
        family_set = set(best_family)
        
        for ls_round in range(30):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try removing each element and finding a better replacement
            for i in range(len(best_family)):
                if time.time() > deadline - 5:
                    break
                
                removed = best_family[i]
                temp_set = family_set - {removed}
                temp_list = [x for x in best_family if x != removed]
                
                # Try a few random candidates not in temp_set
                candidates = [m for m in range(total_masks) if m not in temp_set]
                random.shuffle(candidates)
                
                found = False
                for C in candidates[:200]:
                    # Check if C conflicts with any in temp_set
                    c_conf = conflict[C]
                    sel_bits = 0
                    for s in temp_set:
                        sel_bits |= (1 << s)
                    if (c_conf & sel_bits) == 0:
                        new_fam = temp_list + [C]
                        if len(new_fam) > len(best_family):
                            best_family = new_fam
                            best_size = len(best_family)
                            family_set = set(best_family)
                            improved = True
                        found = True
                        break
                
                if found and improved:
                    break
            
            if not improved:
                break
    
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
    
    return result

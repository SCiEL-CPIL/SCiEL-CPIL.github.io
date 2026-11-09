"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sunflower_triple(a, b, c):
    """Check if three 4-sets form a sunflower (all pairwise intersections equal)."""
    # For a sunflower, a∩b = a∩c = b∩c
    # This means the intersection of any two is the same set
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _is_sunflower_triple_fast(mask_a, mask_b, mask_c):
    """Fast sunflower check using bitmasks.
    Three sets form a sunflower iff a∩b = a∩c = b∩c.
    This is equivalent to: the symmetric difference pattern has a specific structure.
    Specifically, if we let x = a∩b∩c (core), then:
    - a = x ∪ A, b = x ∪ B, c = x ∪ C where A, B, C are disjoint non-empty (for 4-sets)
    - Actually for k=4, the core can be size 0, 1, 2, or 3.
    - If core size is 3: a,b,c share 3 elements, each has 1 unique. Then a∩b = core (size 3), a∩c = core, b∩c = core. Sunflower.
    - If core size is 2: a,b,c share 2 elements. Each has 2 more. For a∩b=a∩c=b∩c, the non-core parts must be such that... 
      Let a = core ∪ {p,q}, b = core ∪ {r,s}, c = core ∪ {t,u}. For a∩b to equal core, {p,q}∩{r,s}=∅. Similarly all non-core pairs disjoint.
      But we only have 16-2=14 other elements, and we need 6 distinct non-core elements. This is possible.
      Actually, a∩b = core ∪ ({p,q}∩{r,s}). For this to equal core, we need {p,q}∩{r,s}=∅.
      Similarly for other pairs. So all six non-core elements must be distinct.
    - If core size is 1: similar, need 9 distinct non-core elements.
    - If core size is 0: need 12 distinct non-core elements, and all pairwise intersections empty.
    
    Using bitmasks: a∩b = a∩c = b∩c iff:
    (a & b) == (a & c) == (b & c)
    """
    ab = mask_a & mask_b
    ac = mask_a & mask_c
    bc = mask_b & mask_c
    return ab == ac == bc

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # Generate all 4-subsets as bitmasks
    all_sets = []
    all_masks = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(combo)
        all_masks.append(mask)
    
    n_total = len(all_sets)  # 2184
    
    # Precompute: for each pair of sets, check if they can form a sunflower with any third set
    # Actually, let's build the conflict graph differently.
    # Two sets A, B are "conflicting" if there exists a third set C such that {A,B,C} is a sunflower.
    # But that's complex. Let's use a different approach.
    
    # Alternative: Greedy with random ordering, checking against already-chosen sets.
    # For efficiency, we maintain the chosen set and check each new candidate against all pairs of chosen sets.
    
    # Better: Use the fact that for a fixed pair (A,B), the sets C that form a sunflower with A and B are constrained.
    # If A∩B = S (the core), then C must contain S, and C\S must be disjoint from (A\S) and (B\S).
    # So C = S ∪ T where T is a (4-|S|)-subset of [16]\(A∪B).
    
    # Let's precompute for each pair (i,j) the "forbidden" masks for the third set.
    # Actually, let's just do greedy with random restarts.
    
    best_family = []
    best_size = 0
    
    # Multiple random restarts
    max_restarts = 50
    attempts = 0
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        random.seed(restart * 7919)  # Deterministic for reproducibility
        order = list(range(n_total))
        random.shuffle(order)
        
        chosen = []
        chosen_masks = []
        
        for idx in order:
            if time.time() > deadline - 2:
                break
            
            mask = all_masks[idx]
            is_valid = True
            
            # Check against all pairs of already chosen sets
            for i in range(len(chosen)):
                mask_i = chosen_masks[i]
                # Check if {mask, mask_i, mask_j} forms a sunflower for any j < i
                # Actually we need to check all pairs (i,j) in chosen
                for j in range(i):
                    mask_j = chosen_masks[j]
                    if _is_sunflower_triple_fast(mask, mask_i, mask_j):
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                chosen.append(idx)
                chosen_masks.append(mask)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            if best_size >= 250:
                break
        
        attempts += 1
    
    # Return the best family found
    result = [sorted([all_sets[i][k]+1 for k in range(4)]) for i in best_family]
    return result
# EVOLVE-BLOCK-END

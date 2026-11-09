import random
from itertools import combinations
from collections import defaultdict
import time

def construct():
    # All 4-subsets of [1..16] as bitmasks
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    n_total = len(all_sets)  # 1820
    
    # Precompute pairwise intersections for all pairs
    # This is O(n^2) = ~1.6M entries, manageable
    # But storing all is heavy; instead we'll compute on the fly but use a smarter approach
    
    best_family = []
    best_size = 0
    
    # Strategy: Use randomized greedy with lookahead, plus local search improvements
    # Key insight: The sunflower condition for k=4 sets means that for any three sets A,B,C:
    # |A∩B| = |B∩C| = |A∩C| and the common intersection structure must be consistent.
    # For 4-element sets, the intersections can have size 0,1,2,3,4.
    # Three sets form a "sunflower" if there exists a core C such that A-C, B-C, C'-C are pairwise disjoint
    # and A∩B∩C' = C.
    # 
    # Simpler characterization for k=4: Three distinct 4-sets A,B,C violate the condition if
    # there's a set T such that A∩B∩C = T and (A-T), (B-T), (C-T) are pairwise disjoint.
    # This is equivalent to: the three pairwise intersections are all equal to some set K,
    # AND the triple intersection equals K.
    
    def violates_sunflower(A, B, C):
        """Check if three 4-element sets form a sunflower (including empty core)."""
        AB = A & B
        AC = A & C
        BC = B & C
        # For a sunflower, all three pairwise intersections must be equal
        if AB != AC or AB != BC:
            return False
        # Also need the triple intersection to equal this common intersection
        ABC = A & B & C
        if ABC != AB:
            return False
        return True
    
    # For efficiency, precompute intersection relationships
    # We'll use a different approach: maintain for each possible intersection mask K,
    # the set of chosen elements that have intersection K with a new element.
    
    # Better approach: For adding S to chosen family, check all pairs (A,B) in chosen
    # where A&B = A&S = B&S. This is the sunflower condition.
    # 
    # Even better: Group chosen elements by their intersection with S.
    # For each group with intersection K, if there are two elements A,B in that group
    # such that A&B == K, then (A,B,S) forms a sunflower.
    
    def try_add(S, chosen, groups_by_inter):
        """Try to add S to the family. groups_by_inter maps intersection_mask -> list of indices."""
        # Compute intersections of S with all chosen
        temp_groups = defaultdict(list)
        for i, A in enumerate(chosen):
            K = A & S
            temp_groups[K].append(i)
        
        # Check for conflicts within each group
        for K, idxs in temp_groups.items():
            if len(idxs) > 1:
                for i in range(len(idxs)):
                    for j in range(i+1, len(idxs)):
                        if (chosen[idxs[i]] & chosen[idxs[j]]) == K:
                            return False
        return True
    
    # Local search: try to improve a family by removing one set and adding another
    def local_search(family, max_iters=50):
        """Try to improve family via 1-swap moves."""
        family_set = set(family)
        remaining = [s for s in all_sets if s not in family_set]
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            random.shuffle(remaining)
            for S in remaining:
                if S in family_set:
                    continue
                # Try removing each member and adding S
                for i in range(len(family)):
                    removed = family[i]
                    new_family = family[:i] + family[i+1:]
                    # Check if S can be added to new_family
                    if try_add(S, new_family, None):
                        new_family.append(S)
                        family = new_family
                        family_set.discard(removed)
                        family_set.add(S)
                        remaining = [s for s in all_sets if s not in family_set]
                        improved = True
                        break
                if improved:
                    break
        return family
    
    # Run multiple restarts
    start_time = time.time()
    timeout = 55  # Leave 5s buffer
    
    for restart in range(100):
        if time.time() - start_time > timeout:
            break
        
        random.shuffle(all_sets)
        chosen = []
        chosen_set = set()
        
        for S in all_sets:
            if try_add(S, chosen, None):
                chosen.append(S)
                chosen_set.add(S)
        
        # Local search
        chosen = local_search(chosen, max_iters=30)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = list(chosen)
    
    # Convert bitmasks back to sorted lists
    result = []
    for mask in best_family:
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result

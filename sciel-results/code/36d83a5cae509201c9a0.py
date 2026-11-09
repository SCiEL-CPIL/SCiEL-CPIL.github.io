"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Precompute all 4-element subsets
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Precompute subsets of each set for fast conflict checking
    # For a set S, we need to check all I subset of S
    # We can just iterate 16 masks or precompute list of subsets
    # Since k=4, 2^4=16 subsets.
    
    def get_subsets(s):
        # s is a set of 4 elements
        # Return list of all subsets of s
        elems = list(s)
        subs = []
        for mask in range(16):
            sub = set()
            for i in range(4):
                if mask & (1 << i):
                    sub.add(elems[i])
            subs.append(frozenset(sub))
        return subs

    # Precompute subsets for all sets to avoid repeated calculation
    all_subsets = [get_subsets(s) for s in all_sets]
    
    best_family = []
    best_size = 0
    
    def _greedy_construct(order, deadline):
        """
        Greedy construction with optimized conflict checking using pair buckets.
        """
        family = []
        family_indices = [] # indices into all_sets
        # pair_buckets: dict mapping frozenset(intersection) to list of (idx_i, idx_j)
        pair_buckets = {}
        
        for idx in order:
            if time.time() > deadline:
                break
            
            s = all_sets[idx]
            conflict = False
            
            # Check against all existing pairs
            # Only need to check pairs (i, j) such that intersection I is a subset of s
            for sub in all_subsets[idx]:
                if sub in pair_buckets:
                    for i, j in pair_buckets[sub]:
                        # Check if A & S == sub AND B & S == sub
                        # Note: We know A & B == sub
                        if (all_sets[i] & s) == sub and (all_sets[j] & s) == sub:
                            conflict = True
                            break
                    if conflict:
                        break
            
            if not conflict:
                # Add to family
                new_idx = len(family)
                family.append(s)
                family_indices.append(idx)
                
                # Update pair_buckets
                for i in range(new_idx):
                    inter = family[i] & s
                    key = frozenset(inter)
                    if key not in pair_buckets:
                        pair_buckets[key] = []
                    pair_buckets[key].append((i, new_idx))
                
                if len(family) > best_size:
                    # Update global best? Or return and compare?
                    # Returning and comparing is safer for SA later, 
                    # but here we just want the best greedy result.
                    pass
        
        return family

    # Phase 1: Multiple Greedy Restarts
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Random order
        order = list(range(n_sets))
        rng.shuffle(order)
        
        family = _greedy_construct(order, deadline)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        # If we hit a high threshold, we might stop greedy and move to SA
        if best_size >= 200: # Arbitrary threshold to trigger SA
             break

    # Phase 2: Simulated Annealing on the best family
    if time.time() < deadline and best_family:
        # Initialize SA state
        # We need to be able to add/remove sets.
        # We'll work with a set of indices.
        current_indices = set()
        for i, s in enumerate(best_family):
             # Find index in all_sets
             # This is slow. Better to keep indices from greedy.
             # Let's refactor _greedy_construct to return indices.
             pass
        
        # Let's restart Phase 1 to capture indices.
        # Or just reconstruct indices from sets.
        # Map set to index
        set_to_idx = {frozenset(s): i for i, s in enumerate(all_sets)}
        current_indices = {set_to_idx[frozenset(s)] for s in best_family}
        
        # Rebuild pair_buckets for current family
        pair_buckets = {}
        idx_list = list(current_indices)
        for i in range(len(idx_list)):
            for j in range(i+1, len(idx_list)):
                s_i = all_sets[idx_list[i]]
                s_j = all_sets[idx_list[j]]
                inter = frozenset(s_i & s_j)
                if inter not in pair_buckets:
                    pair_buckets[inter] = []
                pair_buckets[inter].append((i, j)) # Note: these are indices in idx_list, not global
                # Wait, the conflict check logic above used indices into 'family' list.
                # We need to be consistent.
                # Let's use a helper class or just careful bookkeeping.
                
        # To keep it simple and robust, let's just do "Local Improvement" 
        # by trying to add all remaining sets in random order.
        # And "Swapping": remove one, try to add two.
        
        # Simple Local Search:
        # 1. Try to add any set not in family.
        # 2. If stuck, try removing one set and adding two others.
        
        # We will implement a simple SA loop that tries to add sets.
        # State: list of sets in family.
        # Move: Pick a random set not in family, try to add.
        #       If fails, pick a random set in family, remove it, try to add the candidate.
        #       If that fails, maybe remove two and add three?
        
        # Given time constraints, a simple "Add if possible" loop with random shuffles of the complement
        # is often effective if the greedy was good.
        
        # Let's do a few rounds of "Try to add random unselected sets".
        selected_set = set(frozenset(s) for s in best_family)
        unselected = [s for s in all_sets if frozenset(s) not in selected_set]
        
        # Rebuild buckets for best_family for fast checking
        fam_list = best_family[:]
        buckets = {}
        for i in range(len(fam_list)):
            for j in range(i+1, len(fam_list)):
                inter = frozenset(fam_list[i] & fam_list[j])
                if inter not in buckets:
                    buckets[inter] = []
                buckets[inter].append((i, j))
        
        def can_add(s, fam_list, buckets):
            for sub in all_subsets[set_to_idx[frozenset(s)]]:
                if sub in buckets:
                    for i, j in buckets[sub]:
                        if (fam_list[i] & s) == sub and (fam_list[j] & s) == sub:
                            return False
            return True

        def add_set(s, fam_list, buckets):
            new_idx = len(fam_list)
            fam_list.append(s)
            for i in range(new_idx):
                inter = frozenset(fam_list[i] & s)
                if inter not in buckets:
                    buckets[inter] = []
                buckets[inter].append((i, new_idx))

        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle unselected for randomness
            rng.shuffle(unselected)
            for s in unselected:
                if time.time() > deadline:
                    break
                if frozenset(s) in selected_set:
                    continue
                if can_add(s, fam_list, buckets):
                    add_set(s, fam_list, buckets)
                    selected_set.add(frozenset(s))
                    improved = True
            
            if improved:
                # Rebuild unselected list
                selected_set = set(frozenset(s) for s in fam_list)
                unselected = [s for s in all_sets if frozenset(s) not in selected_set]
                
        best_family = fam_list

    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit

    n = 16
    k = 4

    # Generate all 4-element subsets of [1,16] as bitmasks (0-indexed internally)
    all_sets = list(combinations(range(n), k))
    total = len(all_sets)
    
    # Map each combination to a bitmask
    masks = []
    for combo in all_sets:
        mask = 0
        for x in combo:
            mask |= (1 << x)
        masks.append(mask)

    # Precompute all pairwise intersection masks
    # inter[i][j] = masks[i] & masks[j]
    # Using a flat list or 2D list. 1820x1820 is ~3.3M entries.
    # Each entry is a small int. This is fast to compute and access.
    inter = [[0] * total for _ in range(total)]
    for i in range(total):
        mi = masks[i]
        for j in range(i + 1, total):
            val = mi & masks[j]
            inter[i][j] = val
            inter[j][i] = val

    def creates_sunflower(family_indices, t_idx):
        """
        Check if adding set t_idx to the family (list of indices) creates a sunflower.
        A sunflower exists if there are A, B in family such that:
        A & B == A & T == B & T
        """
        t_mask = masks[t_idx]
        f_len = len(family_indices)
        # Iterate over pairs in the family
        for i in range(f_len):
            a_idx = family_indices[i]
            a_mask = masks[a_idx]
            ac = a_mask & t_mask
            
            # Optimization: if ac is 0, then for a sunflower, ab and bc must also be 0.
            # If ac is not 0, it must equal ab and bc.
            
            for j in range(i + 1, f_len):
                b_idx = family_indices[j]
                b_mask = masks[b_idx]
                ab = a_mask & b_mask
                bc = b_mask & t_mask
                if ab == ac == bc:
                    return True
        return False

    def run_greedy(order_indices):
        family = []
        # Use a set for fast lookup if needed, but list is fine for iteration
        for idx in order_indices:
            if time.time() > deadline - 2.0:
                break
            if not creates_sunflower(family, idx):
                family.append(idx)
        return family

    def local_search(family, max_rounds=50):
        """
        Try to improve the family by swapping elements or adding if possible.
        Since greedy is maximal (no single add works), we focus on swaps.
        """
        family = list(family)
        in_family = [False] * total
        for idx in family:
            in_family[idx] = True
            
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5.0:
            improved = False
            rounds += 1
            
            # Try to find a swap that increases size? No, swaps keep size same.
            # But maybe a swap enables a subsequent add?
            # Or maybe we can find a "better" configuration?
            # For now, let's just try random swaps to escape local minima if we were doing SA.
            # But since we just want max size, and greedy is maximal, swaps don't increase size directly.
            # However, a swap might allow an ADD later.
            
            # Let's try: remove one, try to add two?
            # That's O(N^2) per removal.
            
            # Simpler: Just try to add any available set. (Should fail if greedy was maximal)
            # But maybe the greedy order left some set addable if we removed a conflicting one?
            
            # Strategy:
            # 1. Pick a random member to remove.
            # 2. Try to add any compatible set.
            # 3. If added, keep the swap.
            # 4. If not, restore.
            
            # To make it more effective, we can try multiple removals per round.
            
            candidates_to_remove = list(range(len(family)))
            random.shuffle(candidates_to_remove)
            
            for i in candidates_to_remove:
                if time.time() > deadline - 5.0:
                    break
                
                removed_idx = family[i]
                # Temporarily remove
                new_family = [x for j, x in enumerate(family) if j != i]
                in_family[removed_idx] = False
                
                # Try to add any set not in family
                added = False
                # Shuffle candidates for randomness
                all_candidates = [idx for idx in range(total) if not in_family[idx]]
                random.shuffle(all_candidates)
                
                for cand in all_candidates:
                    if time.time() > deadline - 5.0:
                        break
                    if not creates_sunflower(new_family, cand):
                        # Success! Swap
                        family = new_family
                        family.append(cand)
                        in_family[cand] = True
                        improved = True
                        added = True
                        break
                
                if not added:
                    # Restore
                    family = list(family) # Wait, family was modified in place? No, new_family was a copy.
                    # Actually, if not added, we didn't change 'family' list structure, just 'in_family' flag.
                    # We need to be careful with 'family' list consistency.
                    # Let's just rebuild 'family' from 'in_family' if we get confused, or stick to indices.
                    # Easier:
                    # If not added, revert in_family
                    in_family[removed_idx] = True
                    # 'family' list was not changed if we didn't append. 
                    # But 'new_family' was a copy.
                    # So 'family' is still the original list.
                    pass

        return family

    best_family = []

    # 1. Lexicographic Greedy
    order_lex = list(range(total))
    fam = run_greedy(order_lex)
    if len(fam) > len(best_family):
        best_family = fam[:]

    # 2. Reverse Lexicographic Greedy
    if time.time() < deadline - 10.0:
        order_rev = list(reversed(range(total)))
        fam = run_greedy(order_rev)
        if len(fam) > len(best_family):
            best_family = fam[:]

    # 3. Random Permutation Greedies
    # We generate random permutations of the universe [0..15]
    # and sort the sets based on the permuted values.
    
    num_perm_runs = 20
    for run in range(num_perm_runs):
        if time.time() > deadline - 10.0:
            break
        
        random.seed(12345 + run)
        perm = list(range(n))
        random.shuffle(perm)
        
        # Create keys for sorting
        # For each set (mask), get its elements, map them through perm, sort, and use as key.
        keys = []
        for idx in range(total):
            mask = masks[idx]
            elems = []
            for i in range(n):
                if mask & (1 << i):
                    elems.append(perm[i])
            # The relative order of these elements in the permuted space defines the "lexicographic" position
            # We want to sort the sets by this tuple.
            keys.append((tuple(sorted(elems)), idx))
        
        keys.sort()
        order_perm = [idx for _, idx in keys]
        
        fam = run_greedy(order_perm)
        if len(fam) > len(best_family):
            best_family = fam[:]
        
        # Try local search on the best result so far? Or on this one?
        # Local search is expensive. Let's apply it to the best one found so far.
        # Actually, let's apply it to the current best_family at the end of the loop or periodically.
        # To save time, let's only run local search on the final best_family.
        # Or run it on a few promising ones.
        
        # Let's try local search on the current best_family if we have time
        if time.time() < deadline - 20.0 and run % 5 == 4:
            # Run local search on a copy of the current best
            improved_fam = local_search(best_family, max_rounds=10)
            if len(improved_fam) > len(best_family):
                best_family = improved_fam

    # Final local search on the best family found
    if time.time() < deadline - 10.0 and best_family:
        # Run local search with more rounds if time permits
        improved_fam = local_search(best_family, max_rounds=20)
        if len(improved_fam) > len(best_family):
            best_family = improved_fam

    # Final attempt to add any remaining sets (in case local search broke maximality? unlikely but safe)
    if best_family:
        in_best = [False] * total
        for idx in best_family:
            in_best[idx] = True
        
        # Try to add any set
        for idx in range(total):
            if time.time() > deadline - 2.0:
                break
            if not in_best[idx]:
                if not creates_sunflower(best_family, idx):
                    best_family.append(idx)
                    in_best[idx] = True

    # Convert back to sorted lists of 1-based integers
    result = []
    for idx in best_family:
        # Reconstruct from mask or original combo
        # original combo is all_sets[idx]
        result.append(sorted(all_sets[idx]))
    
    return result

# EVOLVE-BLOCK-END

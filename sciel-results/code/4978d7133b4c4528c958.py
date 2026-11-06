"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    MASK = (1 << 201) - 1
    TARGET_BIT = 1 << 200
    TARGET = 200

    def greedy(order):
        """Greedy: include numbers in given order if it doesn't create sum 200.
        Uses bitset for speed."""
        reachable = 1  # bit 0 set
        selected = []
        for x in order:
            new_bits = (reachable << x) & MASK
            if new_bits & TARGET_BIT:
                continue
            reachable |= new_bits
            selected.append(x)
        return selected

    def compute_reachable(lst):
        """Compute reachable bitset for a list."""
        r = 1
        for x in lst:
            r |= (r << x) & MASK
        return r

    def find_conflict_subset(reachable, x, current_set):
        """
        If adding x creates a conflict, find a small subset of current_set 
        that sums to TARGET - x.
        Returns the subset or None.
        """
        target_remainder = TARGET - x
        if target_remainder < 0:
            return None
        
        # We need a subset of current_set summing to target_remainder.
        # We can backtrack from the reachable bitset, but that's hard.
        # Instead, we can just check if target_remainder is reachable.
        # If so, we need to find *which* elements.
        # For local search, we just need *one* element to swap out.
        # Let's find any element y in current_set such that (reachable_wo_y) has bit (target_remainder - y) set?
        # Actually, simpler: just return None and let the caller try all swaps.
        # This function is a placeholder for future optimization.
        return None

    def local_improve(selected, excluded):
        """
        Improved local search:
        1. Try to add excluded elements.
        2. If blocked, try to swap with an element from the current set.
        3. If still blocked, try 1-for-2 swaps.
        Uses incremental updates where possible.
        """
        best = list(selected)
        excluded_list = list(excluded)
        best_set = set(best)
        reachable = compute_reachable(best)
        
        # Sort excluded to try larger ones first? Or smaller?
        # Larger numbers are "rarer" in sums, so maybe easier to fit?
        # Let's try sorting excluded descending.
        excluded_list.sort(reverse=True)

        improved = True
        stagnation = 0
        
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Phase 1: Add
            for i, x in enumerate(excluded_list):
                new_bits = (reachable << x) & MASK
                if not (new_bits & TARGET_BIT):
                    best.append(x)
                    best_set.add(x)
                    excluded_list.pop(i)
                    reachable |= new_bits
                    improved = True
                    stagnation = 0
                    break
            if improved:
                continue

            # Phase 2: Swap 1-for-1
            # We want to replace y in best with x in excluded.
            # This is valid if (best - {y} + {x}) has no sum 200.
            # Equivalent to: (reachable_wo_y << x) does not have TARGET_BIT.
            # And reachable_wo_y is computed by removing y from best.
            
            # Optimization: Only consider swaps if x > y? Not necessarily, but often beneficial for size?
            # No, size is constant in 1-for-1.
            
            n_best = len(best)
            n_excl = len(excluded_list)
            
            # To speed up, we can precompute reachable_wo_y for all y? 
            # That's O(N^2) bit ops. N~60. 3600 ops. Fast.
            # But we only do it if we are in this phase.
            
            swapped = False
            # Try to find a valid swap
            # We iterate y in best, x in excluded.
            # To avoid O(N*M) full recomputes, we can be smart.
            
            # Let's just do the standard loop but break early.
            for i, y in enumerate(best):
                # Compute reachable without y
                # This is the expensive part.
                reachable_no_y = compute_reachable([z for z in best if z != y])
                
                for j, x in enumerate(excluded_list):
                    if x == y:
                        continue
                    new_bits = (reachable_no_y << x) & MASK
                    if not (new_bits & TARGET_BIT):
                        # Swap successful
                        best[i] = x
                        best_set.discard(y)
                        best_set.add(x)
                        
                        # Update excluded list
                        # Remove x, add y
                        excluded_list[j] = y
                        # Re-sort? Maybe not needed immediately.
                        
                        reachable = compute_reachable(best)
                        improved = True
                        swapped = True
                        break
                if swapped:
                    break
            if improved:
                continue

            # Phase 3: Swap 1-for-2 (Increase size by 1)
            # Remove y from best, add x1, x2 from excluded.
            # Size +1.
            found = False
            for i, y in enumerate(best):
                reachable_no_y = compute_reachable([z for z in best if z != y])
                
                # Try pairs (x1, x2) from excluded
                n_excl = len(excluded_list)
                for a in range(n_excl):
                    x1 = excluded_list[a]
                    b1 = (reachable_no_y << x1) & MASK
                    if b1 & TARGET_BIT:
                        continue
                    for b in range(n_excl):
                        if b == a:
                            continue
                        x2 = excluded_list[b]
                        b2 = (b1 << x2) & MASK
                        if not (b2 & TARGET_BIT):
                            # Swap y -> x1, x2
                            best[i] = x1
                            best.append(x2)
                            best_set.discard(y)
                            best_set.add(x1)
                            best_set.add(x2)
                            
                            # Update excluded
                            # Remove x1, x2, add y
                            new_excluded = []
                            for k, ex in enumerate(excluded_list):
                                if k != a and k != b:
                                    new_excluded.append(ex)
                            new_excluded.append(y)
                            excluded_list = new_excluded
                            excluded_list.sort(reverse=True)
                            
                            reachable = compute_reachable(best)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if improved:
                continue

            # Phase 4: Random Perturbation (Simulated Annealing style)
            # If stuck, try a random swap that might decrease size temporarily? 
            # Or just randomize order of excluded?
            # Let's try a random 1-for-1 swap even if it doesn't increase size?
            # No, we only care about max size.
            
            # If we are stuck, break to save time for other restarts?
            # Or try 2-for-3?
            # 2-for-3 is very expensive.
            
            # Let's try a "kick": randomly swap two elements in best?
            # No, that might break the constraint.
            
            # If no improvement, break local search to allow more greedy restarts.
            break

        return best

    # Try multiple orderings
    all_numbers = list(range(1, 101))
    best_result = []

    # Descending order
    r = greedy(all_numbers[::-1])
    if len(r) > len(best_result):
        best_result = r

    # Ascending order
    r = greedy(all_numbers)
    if len(r) > len(best_result):
        best_result = r

    # Modular orderings
    for mod in [2, 3, 4, 5, 7, 11, 13]:
        groups = [[] for _ in range(mod)]
        for x in all_numbers:
            groups[x % mod].append(x)
        order = []
        max_len = max(len(g) for g in groups)
        for i in range(max_len):
            for g in groups:
                if i < len(g):
                    order.append(g[i])
        r = greedy(order)
        if len(r) > len(best_result):
            best_result = r

    # Weighted ordering
    r = greedy(sorted(all_numbers, key=lambda x: abs(x - 50)))
    if len(r) > len(best_result):
        best_result = r

    # More aggressive random search with multiple seeds
    rng = random.Random(42)
    attempts = 0
    random_budget = deadline - 15
    while time.time() < random_budget:
        order = all_numbers[:]
        rng.shuffle(order)
        r = greedy(order)
        if len(r) > len(best_result):
            best_result = r
            attempts = 0
        else:
            attempts += 1
        if attempts > 300:
            break

    # Local improvement on best result
    selected_set = set(best_result)
    excluded = [x for x in all_numbers if x not in selected_set]
    best_result = local_improve(best_result, excluded)

    # Try local improvement from other promising greedy results
    # We keep the best few greedy results to improve them
    candidates = []
    # Add the initial best
    candidates.append(best_result)
    
    # Add some other good greedy results
    for order in [all_numbers[::-1], all_numbers, sorted(all_numbers, key=lambda x: x % 3)]:
        r = greedy(order)
        if len(r) >= len(best_result) - 5:
            candidates.append(r)

    # Improve candidates
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        sel_set = set(cand)
        exc = [x for x in all_numbers if x not in sel_set]
        r_improved = local_improve(cand, exc)
        if len(r_improved) > len(best_result):
            best_result = r_improved

    if not best_result:
        return [100]

    return best_result
# EVOLVE-BLOCK-END

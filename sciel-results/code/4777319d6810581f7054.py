import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.monotonic() + 620.0  # Safety margin for 630s limit

    # 1. Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        f = (x * x) % m
        if f != 0:
            forbidden.add(f)
    forbidden.discard(0)

    # 2. Precompute conflict masks
    # conflict_mask[i] has bit j set if i and j conflict
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def is_feasible(subset_mask, candidate):
        """Check if candidate can be added to subset_mask."""
        if subset_mask & (1 << candidate):
            return False
        return (conflict_mask[candidate] & subset_mask) == 0

    def popcount(mask):
        return bin(mask).count('1')

    def mask_to_list(mask):
        return [i for i in range(m) if (mask >> i) & 1]

    def list_to_mask(lst):
        mask = 0
        for x in lst:
            mask |= (1 << x)
        return mask

    # 3. Greedy Construction
    def greedy_construct(order):
        subset = 0
        for cand in order:
            if is_feasible(subset, cand):
                subset |= (1 << cand)
        return subset

    # 4. Branch and Bound for Maximum Independent Set
    # Using bitset operations for efficiency
    def branch_and_bound(available, current_set, current_size, best_size_ref):
        """
        available: bitmask of vertices still available
        current_set: bitmask of vertices in current independent set
        current_size: number of vertices in current_set
        best_size_ref: list containing [best_size, best_mask]
        """
        if time.monotonic() > deadline:
            return
        
        # Upper bound: current_size + popcount(available)
        # If this can't beat best, prune
        if current_size + popcount(available) <= best_size_ref[0]:
            return
        
        if available == 0:
            if current_size > best_size_ref[0]:
                best_size_ref[0] = current_size
                best_size_ref[1] = current_set
            return
        
        # Choose vertex with minimum degree in available subgraph
        # (or maximum degree for better pruning - actually min degree for MIS)
        # For maximum independent set, we want to branch on vertices
        # Let's use a heuristic: pick vertex with highest degree in available
        
        # Compute degrees within available
        best_v = -1
        best_deg = -1
        avail_list = [v for v in range(m) if (available >> v) & 1]
        
        for v in avail_list:
            deg = popcount(conflict_mask[v] & available)
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        new_available = available & ~(1 << best_v) & ~conflict_mask[best_v]
        new_current = current_set | (1 << best_v)
        new_size = current_size + 1
        
        if new_size + popcount(new_available) > best_size_ref[0]:
            branch_and_bound(new_available, new_current, new_size, best_size_ref)
        
        # Branch 2: Exclude best_v
        new_available2 = available & ~(1 << best_v)
        
        if current_size + popcount(new_available2) > best_size_ref[0]:
            branch_and_bound(new_available2, current_set, current_size, best_size_ref)

    # 5. Main Search
    best_mask = 0
    best_size = 0
    rng = random.Random(42)
    
    # Phase 1: Multiple Greedy Starts to get good initial lower bound
    orderings = []
    # Random
    for _ in range(50):
        order = list(range(m))
        rng.shuffle(order)
        orderings.append(order)
    # Degree based
    degrees = [bin(conflict_mask[i]).count('1') for i in range(m)]
    order_low_deg = sorted(range(m), key=lambda x: degrees[x])
    order_high_deg = sorted(range(m), key=lambda x: -degrees[x])
    orderings.append(order_low_deg)
    order_high_deg
    
    initial_solutions = []
    for order in orderings:
        sol = greedy_construct(order)
        initial_solutions.append(sol)
        if popcount(sol) > best_size:
            best_mask = sol
            best_size = popcount(sol)
            
    # Phase 2: Branch and Bound with the best greedy solution as initial bound
    # Use bitset-based B&B
    all_mask = (1 << m) - 1
    best_bb = [best_size, best_mask]
    
    # Run branch and bound
    # This might be slow for 205 vertices, so we need to be careful
    # Let's try it with the time limit
    branch_and_bound(all_mask, 0, 0, best_bb)
    
    if best_bb[1] != 0:
        best_mask = best_bb[1]
        best_size = best_bb[0]
    
    # Phase 3: If B&B didn't finish or found same, try local search improvements
    def local_search(initial_mask, rng, max_iters=1000):
        current = initial_mask
        cur_size = popcount(current)
        best = current
        best_size = cur_size
        
        all_elements = list(range(m))
        
        for _ in range(max_iters):
            if time.monotonic() > deadline:
                break
            
            improved = False
            
            # Phase A: Try to add any element
            for cand in all_elements:
                if not (current & (1 << cand)):
                    if is_feasible(current, cand):
                        current |= (1 << cand)
                        cur_size += 1
                        improved = True
                        if cur_size > best_size:
                            best = current
                            best_size = cur_size
                        break
            
            if improved:
                continue
            
            # Phase B: Try to swap
            in_current = [x for x in all_elements if (current & (1 << x))]
            not_in_current = [x for x in all_elements if not (current & (1 << x))]
            
            rng.shuffle(in_current)
            rng.shuffle(not_in_current)
            
            found_swap = False
            for rem in in_current:
                if time.monotonic() > deadline:
                    break
                temp_mask = current & ~(1 << rem)
                
                for add in not_in_current:
                    if add == rem:
                        continue
                    if is_feasible(temp_mask, add):
                        current = temp_mask | (1 << add)
                        found_swap = True
                        break
                if found_swap:
                    break
            
            if found_swap:
                continue
            
            # Phase C: Kick - remove 2, add 2+
            if len(in_current) >= 2:
                rem1, rem2 = rng.sample(in_current, 2)
                temp_mask = current & ~(1 << rem1) & ~(1 << rem2)
                
                candidates = [x for x in not_in_current if x != rem1 and x != rem2]
                rng.shuffle(candidates)
                
                found_adds = []
                for cand in candidates:
                    if len(found_adds) == 2:
                        break
                    if is_feasible(temp_mask, cand):
                        conflict_with_prev = False
                        for prev in found_adds:
                            if (conflict_mask[cand] >> prev) & 1:
                                conflict_with_prev = True
                                break
                        if not conflict_with_prev:
                            found_adds.append(cand)
                
                if len(found_adds) >= 2:
                    new_mask = temp_mask
                    for a in found_adds:
                        new_mask |= (1 << a)
                    current = new_mask
                    cur_size = popcount(current)
                    if cur_size > best_size:
                        best = current
                        best_size = cur_size
            else:
                break

        return best, best_size

    # Run local search from best B&B result
    if time.monotonic() < deadline - 10:
        improved_mask, improved_size = local_search(best_mask, rng, max_iters=500)
        if improved_size > best_size:
            best_mask = improved_mask
            best_size = improved_size
    
    # Final check
    if best_mask == 0:
        best_mask = 1  # element 0

    result_set = mask_to_list(best_mask)
    result = np.array(sorted(result_set), dtype=np.int64)
    return result

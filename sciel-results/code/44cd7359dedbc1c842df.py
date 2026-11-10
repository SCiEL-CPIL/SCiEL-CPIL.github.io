# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem modulo 205.
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    
    Strategy:
    1. Compute forbidden differences F = {x^2 mod 205 : x in 1..204} \ {0}.
    2. Use a sophisticated local search / simulated annealing to maximize the set size
       while maintaining the constraint that no forbidden difference appears.
    3. Include a deadline guard to ensure we finish well before the 600s limit.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # Safety margin of 20s
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = (x * x) % m
        if val != 0:
            forbidden.add(val)
    
    # Build adjacency: adj[i] = set of j such that (j-i) mod m is forbidden
    # For efficiency, we'll track for each element which others are forbidden
    forbidden_set = forbidden  # for fast lookup
    
    # Greedy initialization: try multiple orderings and take the best
    def greedy_init(order):
        selected = []
        selected_set = set()
        for cand in order:
            ok = True
            for s in selected:
                d1 = (cand - s) % m
                d2 = (s - cand) % m
                if d1 in forbidden_set or d2 in forbidden_set:
                    ok = False
                    break
            if ok:
                selected.append(cand)
                selected_set.add(cand)
        return selected, selected_set
    
    # Try several deterministic orderings
    best_init = None
    best_init_size = -1
    
    orders = [
        list(range(m)),
        list(range(m-1, -1, -1)),
        # Random-ish but deterministic orderings
        list(range(0, m, 3)),
        list(range(1, m, 3)),
        list(range(2, m, 3)),
        list(range(0, m, 7)),
        list(range(1, m, 7)),
        list(range(2, m, 7)),
        list(range(0, m, 13)),
        list(range(1, m, 13)),
        list(range(2, m, 13)),
        list(range(0, m, 17)),
        list(range(1, m, 17)),
        list(range(2, m, 17)),
        list(range(0, m, 19)),
        list(range(1, m, 19)),
        list(range(2, m, 19)),
    ]
    
    # Also try interleaved orders
    rng = random.Random(42)
    for seed in range(20):
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        orders.append(order)
    
    for order in orders:
        selected, selected_set = greedy_init(order)
        if len(selected) > best_init_size:
            best_init_size = len(selected)
            best_init = (list(selected), set(selected))
    
    if best_init is None:
        best_init = ([0], {0})
        best_init_size = 1
    
    # Start with the best greedy solution
    current_set, current_set_size = best_init[0], best_init_size
    current_set_set = best_init[1]
    
    # Local search: try to add elements, and if stuck, try swapping
    def can_add(cand, sel_set):
        for s in sel_set:
            d1 = (cand - s) % m
            d2 = (s - cand) % m
            if d1 in forbidden_set or d2 in forbidden_set:
                return False
        return True
    
    # Phase 1: Greedy fill - repeatedly try to add the first available element
    improved = True
    while improved:
        improved = False
        for cand in range(m):
            if cand not in current_set_set:
                if can_add(cand, current_set_set):
                    current_set.append(cand)
                    current_set_set.add(cand)
                    current_set_size += 1
                    improved = True
                    break
    
    # Phase 2: Local search with swaps
    # For each pair (in_set_elem, out_set_elem), check if swapping improves feasibility
    # Actually, we want to keep size the same or increase it. Let's do:
    # Try to remove one and add two, or just try random restarts with perturbations
    
    def count_conflicts(elem, sel_set):
        """Count how many elements in sel_set conflict with elem."""
        cnt = 0
        for s in sel_set:
            d1 = (elem - s) % m
            d2 = (s - elem) % m
            if d1 in forbidden_set or d2 in forbidden_set:
                cnt += 1
        return cnt
    
    # Simulated annealing-like approach
    best_set = list(current_set)
    best_set_set = set(current_set_set)
    best_size = current_set_size
    
    rng2 = random.Random(12345)
    temp = 1.0
    cooling = 0.999
    
    iteration = 0
    while time.time() < deadline - 5:
        iteration += 1
        
        if iteration % 10000 == 0 and time.time() > deadline - 10:
            break
        
        # Generate a move:
        # 1. Try to add a random element not in set
        # 2. Try to swap: remove one, add one
        # 3. Try to remove one (to allow better future additions)
        move_type = rng2.random()
        
        if move_type < 0.5:
            # Try to add
            candidates_to_add = [c for c in range(m) if c not in current_set_set]
            if not candidates_to_add:
                continue
            cand = rng2.choice(candidates_to_add)
            if can_add(cand, current_set_set):
                current_set.append(cand)
                current_set_set.add(cand)
                current_set_size += 1
                if current_set_size > best_size:
                    best_size = current_set_size
                    best_set = list(current_set)
                    best_set_set = set(current_set_set)
            # If not addable, skip (no penalty in this simple version)
        
        elif move_type < 0.85:
            # Swap: remove one from set, add one from outside
            if not current_set:
                continue
            to_remove = rng2.choice(current_set)
            candidates_to_add = [c for c in range(m) if c not in current_set_set]
            if not candidates_to_add:
                continue
            to_add = rng2.choice(candidates_to_add)
            
            # Temporarily remove
            current_set_set.discard(to_remove)
            new_conflicts = 0
            for s in current_set_set:
                d1 = (to_add - s) % m
                d2 = (s - to_add) % m
                if d1 in forbidden_set or d2 in forbidden_set:
                    new_conflicts += 1
            
            if new_conflicts == 0:
                current_set.remove(to_remove)
                current_set.append(to_add)
                current_set_set.add(to_add)
                # Size stays the same
                if current_set_size > best_size:
                    best_size = current_set_size
                    best_set = list(current_set)
                    best_set_set = set(current_set_set)
            else:
                # Revert
                current_set_set.add(to_remove)
        
        else:
            # Remove one (to break out of local optima)
            if not current_set:
                continue
            to_remove = rng2.choice(current_set)
            current_set.remove(to_remove)
            current_set_set.discard(to_remove)
            current_set_size -= 1
        
        # Cool down temperature (not really used in this simple version, but for structure)
        temp *= cooling
    
    # Final greedy fill on the best set found
    final_set = list(best_set)
    final_set_set = set(best_set_set)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for cand in range(m):
            if cand not in final_set_set:
                if can_add(cand, final_set_set):
                    final_set.append(cand)
                    final_set_set.add(cand)
                    improved = True
                    break
    
    # Verify the final set
    for i in range(len(final_set)):
        for j in range(i+1, len(final_set)):
            a, b = final_set[i], final_set[j]
            d1 = (a - b) % m
            d2 = (b - a) % m
            if d1 in forbidden_set or d2 in forbidden_set:
                # Should not happen, but if it does, rebuild
                final_set = [0]
                final_set_set = {0}
                break
    
    return np.array(final_set, dtype=np.int64)

# EVOLVE-BLOCK-END

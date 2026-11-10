# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square modulo 205.
    
    Strategy:
    1. Compute forbidden differences (nonzero quadratic residues mod 205).
    2. Build conflict bitmask: conflict[i] has bit j set if i and j cannot both be in set.
    3. Use bitmask-accelerated local search with multiple random restarts.
       - Start with a random greedy construction.
       - Refine using 1-add and 1-swap moves, all using O(1) bitmask conflict checks.
    4. Return the best solution found within time budget.
    """
    m = 205
    deadline = time.time() + 620  # 10s safety margin before 630s kill
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Precompute: for each i, conflict[i] is a bitmask of all j that conflict with i
    conflict = [0] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i] |= (1 << j)
    
    def can_add(cand, selected_mask):
        """Check if cand can be added to the set represented by selected_mask."""
        return (conflict[cand] & selected_mask) == 0
    
    def greedy_random(seed):
        """Random greedy construction using bitmask conflict checks."""
        rng = random.Random(seed)
        candidates = list(range(m))
        rng.shuffle(candidates)
        selected_mask = 0
        selected_list = []
        for cand in candidates:
            if can_add(cand, selected_mask):
                selected_mask |= (1 << cand)
                selected_list.append(cand)
        return selected_mask, selected_list
    
    def local_search(selected_mask, selected_list, rng, max_iters=2000):
        """1-add and 1-swap local search using bitmask conflict checks."""
        best_mask = selected_mask
        best_list = list(selected_list)
        best_size = len(best_list)
        
        no_improve = 0
        iters = 0
        while iters < max_iters and no_improve < 150:
            iters += 1
            improved = False
            
            # Try 1-add: find a candidate that can be added
            candidates_to_add = [c for c in range(m) if not (best_mask >> c) & 1]
            rng.shuffle(candidates_to_add)
            for cand in candidates_to_add:
                if can_add(cand, best_mask):
                    best_mask |= (1 << cand)
                    best_list.append(cand)
                    best_size += 1
                    improved = True
                    no_improve = 0
                    break
                else:
                    no_improve += 1
            
            if improved:
                continue
            
            # Try 1-swap: remove one element, add another
            # For each element to remove, compute the mask without it,
            # then find a candidate that can be added to the reduced set
            swap_order = list(range(len(best_list)))
            rng.shuffle(swap_order)
            for idx in swap_order:
                rem = best_list[idx]
                temp_mask = best_mask & ~(1 << rem)
                
                # Find a candidate not in temp_mask that can be added
                # We want a candidate that conflicts with rem (otherwise it would have been added already)
                # but doesn't conflict with the rest
                candidates_to_add = [c for c in range(m) if not (temp_mask >> c) & 1]
                rng.shuffle(candidates_to_add)
                for cand in candidates_to_add:
                    if can_add(cand, temp_mask):
                        # Perform the swap
                        best_list[idx] = cand
                        best_mask = temp_mask | (1 << cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
        
        return best_mask, best_list, best_size
    
    # Main search loop
    best_mask = 0
    best_list = []
    best_size = 0
    
    # Start with deterministic greedy (in order)
    sel = []
    sel_mask = 0
    for cand in range(m):
        if can_add(cand, sel_mask):
            sel.append(cand)
            sel_mask |= (1 << cand)
    best_list = sel
    best_size = len(sel)
    best_mask = sel_mask
    
    # Try multiple random restarts
    seed = 1
    while time.time() < deadline:
        # Random greedy start
        start_mask, start_list = greedy_random(seed)
        
        # Local search
        res_mask, res_list, res_size = local_search(start_mask, start_list, random.Random(seed))
        
        if res_size > best_size:
            best_mask = res_mask
            best_list = res_list
            best_size = res_size
        
        seed += 1
        
        # Also try local search on the current best every few iterations
        if seed % 3 == 0:
            res_mask, res_list, res_size = local_search(best_mask, best_list, random.Random(seed + 10000))
            if res_size > best_size:
                best_mask = res_mask
                best_list = res_list
                best_size = res_size
    
    return np.array(best_list, dtype=np.int64)

# EVOLVE-BLOCK-END

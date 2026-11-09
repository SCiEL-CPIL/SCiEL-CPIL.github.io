"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    total = len(all_masks)  # 1820
    
    # Precompute pairwise intersections
    # intersect[i][j] is a bitmask
    intersect = [[0] * total for _ in range(total)]
    for i in range(total):
        mi = all_masks[i]
        for j in range(i + 1, total):
            v = mi & all_masks[j]
            intersect[i][j] = v
            intersect[j][i] = v

    def is_sunflower(a, b, c):
        """Check if three distinct indices form a sunflower."""
        # Check if all pairwise intersections are equal
        # We assume a, b, c are distinct
        return intersect[a][b] == intersect[a][c] == intersect[b][c]

    def find_conflicts_with_new(family, new_idx):
        """
        Returns a list of pairs (i, j) of indices in family 
        such that family[i], family[j], new_idx form a sunflower.
        Also returns True if any conflict exists.
        """
        conflicts = []
        n_fam = len(family)
        new_mask = all_masks[new_idx] # Not strictly needed if using intersect array, but good for clarity
        
        for i in range(n_fam):
            ai = family[i]
            # Optimization: If new_idx and ai don't share enough to form a core with any other set, skip?
            # Hard to know without checking j.
            
            for j in range(i + 1, n_fam):
                aj = family[j]
                if is_sunflower(ai, aj, new_idx):
                    conflicts.append((i, j))
                    
        return conflicts

    def can_add(family, new_idx):
        """Check if new_idx can be added without forming a sunflower."""
        n_fam = len(family)
        for i in range(n_fam):
            ai = family[i]
            for j in range(i + 1, n_fam):
                aj = family[j]
                if is_sunflower(ai, aj, new_idx):
                    return False
        return True

    def get_initial_greedy(rng):
        """Get a valid family using greedy method."""
        order = list(range(total))
        rng.shuffle(order)
        chosen = []
        for idx in order:
            if can_add(chosen, idx):
                chosen.append(idx)
        return chosen

    def local_search(initial_family, rng, time_limit):
        """
        Perform local search on the independent set.
        Moves: 
        1. Add a node if possible.
        2. Swap a node if it doesn't decrease size and breaks potential blocks.
        """
        current_family = initial_family[:]
        current_size = len(current_family)
        best_family = current_family[:]
        best_size = current_size
        
        # To speed up "can_add" and "swap", we might keep a set of indices in family
        # But family is small (~200), so list iteration is fine.
        
        # Create a lookup for fast membership check?
        # family_set = set(current_family)
        
        steps = 0
        while time.time() < time_limit:
            steps += 1
            if steps % 100 == 0 and time.time() > time_limit - 2:
                break
            
            # Try to add a random node
            # Pick a random node not in family
            # To avoid picking from 1820 every time, we can just pick random indices and check
            # Or maintain a list of candidates.
            # For simplicity, pick random indices.
            
            added = False
            swapped = False
            
            # Attempt 1: Add a node
            for _ in range(10): # Try 10 random candidates for addition
                if time.time() > time_limit - 1:
                    return best_family, best_size
                
                u = rng.randint(0, total - 1)
                if u in current_family:
                    continue
                
                if can_add(current_family, u):
                    current_family.append(u)
                    current_size += 1
                    added = True
                    if current_size > best_size:
                        best_size = current_size
                        best_family = current_family[:]
                    break
            
            if added:
                continue
            
            # Attempt 2: Swap a node
            # Pick a node v in family and u not in family
            # Check if (Family - {v}) + {u} is valid
            # This is equivalent to: Does u form a sunflower with any pair in Family - {v}?
            # i.e., All sunflower pairs involving u must include v.
            
            for _ in range(10): # Try 10 random swaps
                if time.time() > time_limit - 1:
                    return best_family, best_size
                
                u = rng.randint(0, total - 1)
                if u in current_family:
                    continue
                
                # Find a v in current_family to remove
                v_idx_in_list = rng.randint(0, current_size - 1)
                v = current_family[v_idx_in_list]
                
                # Check validity of swap
                # We need to ensure no sunflower is formed by u and two sets in current_family \ {v}
                # Iterate through pairs in current_family
                # If a pair (a, b) forms a sunflower with u:
                #   If a == v or b == v, it's okay (v is removed)
                #   Else, swap is invalid
                
                valid_swap = True
                n_fam = current_size
                for i in range(n_fam):
                    ai = current_family[i]
                    if ai == v:
                        continue
                    for j in range(i + 1, n_fam):
                        aj = current_family[j]
                        if aj == v:
                            continue
                        if is_sunflower(ai, aj, u):
                            # Conflict found not involving v
                            valid_swap = False
                            break
                    if not valid_swap:
                        break
                
                if valid_swap:
                    # Perform swap
                    current_family[v_idx_in_list] = u
                    swapped = True
                    # Size remains same, but structure changed
                    # We don't update best_size here as size didn't change
                    # But we might have unblocked future additions
                    break
            
            # If we stuck for too long, maybe restart from best?
            # For now, we just continue. The random swaps act as simulated annealing at T=0 (only neutral or improving moves).
            # Actually, this is just random walk on the level set of the size.
            # It might get stuck.
            # If no swap found after many tries, we could try removing a node to reduce size (annealing), but let's stick to size-preserving/swapping first.
            
            # Heuristic: If we haven't added or swapped in a long while, do a "kick" by removing a random node
            # (This allows escaping local minima where no swap is valid but a different larger set exists).
            # However, reducing size is risky. We'll only do it if we've stalled.
            # We can track `stall_count`.
            
    # Main Loop
    rng = random.Random(42)
    
    best_family_overall = []
    best_size_overall = 0
    
    # Phase 1: Get a good initial solution with a few restarts
    for i in range(5):
        if time.time() > deadline - 20:
            break
        fam = get_initial_greedy(rng)
        if len(fam) > best_size_overall:
            best_size_overall = len(fam)
            best_family_overall = fam[:]
            
    # Phase 2: Local Search on the best initial solution
    # We run local search on the best family found so far
    if best_family_overall:
        ls_family, ls_size = local_search(best_family_overall, rng, deadline - 5)
        if ls_size > best_size_overall:
            best_size_overall = ls_size
            best_family_overall = ls_family[:]

    # Phase 3: More restarts with Local Search
    restarts = 0
    while time.time() < deadline - 2:
        restarts += 1
        fam = get_initial_greedy(rng)
        
        # If this greedy result is already bad, skip LS?
        # Or run LS on it if it's close to best?
        # Let's run LS on it if it's within 5 of best, or just always run it if time permits.
        
        # To save time, we can be selective.
        if len(fam) > best_size_overall - 10:
            ls_family, ls_size = local_search(fam, rng, deadline - 2)
            if ls_size > best_size_overall:
                best_size_overall = ls_size
                best_family_overall = ls_family[:]
                if best_size_overall >= 250:
                    break
        else:
            # Just check if greedy alone beats it
            if len(fam) > best_size_overall:
                best_size_overall = len(fam)
                best_family_overall = fam[:]
                if best_size_overall >= 250:
                    break

    # Convert bitmasks back to sorted lists of 1-based integers
    result = []
    for idx in best_family_overall:
        mask = all_masks[idx]
        elems = [i + 1 for i in range(n) if mask & (1 << i)]
        result.append(elems)
    
    return result
# EVOLVE-BLOCK-END

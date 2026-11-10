import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 580  # Safety margin
    
    # 1. Determine allowed differences
    # d is allowed if d is NOT in F.
    # F = {x^2 mod 205 : x = 1..204} \ {0}
    # d in F iff (d mod 5 in {1, 4}) AND (d mod 41 in QR_41_nonzero)
    
    # Precompute QRs mod 41
    qr_41 = set()
    for x in range(1, 41):
        qr_41.add((x * x) % 41)
        
    allowed_diffs = np.zeros(m, dtype=bool)
    for d in range(1, m):
        d5 = d % 5
        d41 = d % 41
        is_forbidden = (d5 in (1, 4)) and (d41 in qr_41)
        if not is_forbidden:
            allowed_diffs[d] = True
            
    # Note: d=0 is not in F, but we don't check d=0 for distinct elements.
    # The condition is for pairs a,b. d = (a-b)%205. If a!=b, d!=0.
    
    # Precompute compatibility matrix
    # compat[u][v] = True if (u-v) % 205 is an allowed difference
    # This is symmetric.
    compat = np.zeros((m, m), dtype=bool)
    for u in range(m):
        for v in range(u + 1, m):
            d = (u - v) % m
            if allowed_diffs[d]:
                compat[u, v] = True
                compat[v, u] = True
                
    # State
    in_set = np.zeros(m, dtype=bool)
    # conflicts[x] = number of elements y in in_set such that NOT compat[x][y]
    conflicts = np.zeros(m, dtype=int)
    
    def add_element(x):
        in_set[x] = True
        for y in range(m):
            if in_set[y] and y != x:
                if not compat[x, y]:
                    conflicts[x] += 1
                    conflicts[y] += 1
            # Optimization: only update conflicts for y that are in_set
            # Actually, we need to update conflicts for ALL x?
            # No, conflicts[x] counts how many elements in the set conflict with x.
            # When we add y, we must increment conflicts[x] for all x that are NOT compatible with y.
            # And we must increment conflicts[y] for all x in the set that are NOT compatible with y.
            
    # The above add_element is O(N^2) if not careful.
    # Let's use a simpler approach:
    # Maintain `in_set` and `conflicts`.
    # To add x:
    #   1. Check if conflicts[x] == 0.
    #   2. If yes, set in_set[x] = True.
    #   3. For all y in 0..m-1:
    #      If in_set[y] and y != x:
    #         If not compat[x, y]:
    #             conflicts[x] += 1
    #             conflicts[y] += 1
    #      Note: conflicts[x] should remain 0 after addition? No, conflicts[x] counts conflicts with OTHERS.
    #      If we add x, conflicts[x] becomes the number of y in set (excluding x) that are not compatible with x.
    #      Since we only add if conflicts[x] was 0, it remains 0?
    #      Wait. conflicts[x] is the count of y in in_set such that !compat[x,y].
    #      If we add x, we are checking if it's compatible with all CURRENT members.
    #      So if conflicts[x] == 0, it is compatible.
    #      After adding x, for any existing y, if !compat[x,y], then conflicts[y] must be incremented (because x is a new conflict for y).
    #      And conflicts[x] itself? It counts conflicts with existing y. Since there were 0, it stays 0.
    #      But what about future additions?
    #      This logic holds.
    
    def try_add(x):
        if in_set[x]:
            return False
        if conflicts[x] > 0:
            return False
        in_set[x] = True
        for y in range(m):
            if in_set[y] and y != x:
                if not compat[x, y]:
                    conflicts[y] += 1
                    # conflicts[x] remains 0 because we checked it was 0 before adding?
                    # Yes, if it was 0, it means no existing y had !compat[x,y].
                    # So conflicts[x] stays 0.
        return True

    def try_remove(x):
        if not in_set[x]:
            return False
        in_set[x] = False
        for y in range(m):
            if in_set[y] and y != x:
                if not compat[x, y]:
                    conflicts[y] -= 1
                    # conflicts[x] doesn't matter anymore as it's removed
        return True

    def get_size():
        return np.sum(in_set)

    def get_set_list():
        return np.where(in_set)[0]

    best_set = np.zeros(0, dtype=np.int64)
    best_size = 0
    best_conflicts = None
    best_in_set = None

    def update_best():
        nonlocal best_set, best_size
        current_size = get_size()
        if current_size > best_size:
            best_size = current_size
            best_set = get_set_list()
            # Save state to restore if needed? 
            # For now, we just track the max found.
            # We need to be able to backtrack or restart.
            # We'll use a copy of in_set and conflicts.
            # But copying 205 ints is cheap.

    # Strategy 1: Start with multiples of 5
    # Reset state
    in_set[:] = False
    conflicts[:] = 0
    
    # Initialize with multiples of 5
    init_candidates = [i for i in range(m) if i % 5 == 0]
    random.shuffle(init_candidates)
    for x in init_candidates:
        try_add(x)
        
    update_best()
    
    # Strategy 2: Greedy Growth from current state
    # Try to add any element with conflicts[x] == 0
    changed = True
    while changed:
        changed = False
        for x in range(m):
            if not in_set[x] and conflicts[x] == 0:
                try_add(x)
                changed = True
                update_best()
                
    # Strategy 3: Local Search (Swaps)
    # Try to remove one element and add another to potentially unlock more additions
    # Or just try to add elements after removing a "blocking" element.
    
    # We will run a loop until deadline
    # Use random restarts if stuck
    
    current_time = time.time()
    max_iterations = 1000000 # Cap iterations to avoid infinite loops if logic fails
    
    # We need a way to "restart" or "perturb"
    # Let's keep a history of best states?
    # Simpler: Random Restart with Greedy
    
    for iter_num in range(max_iterations):
        if time.time() > deadline:
            break
            
        # If we haven't found a large set, try a random restart
        # Or just keep searching locally.
        
        # Local Search Step:
        # Pick a random element in the set to remove?
        # Or pick a random element NOT in the set and see if we can force it in by removing some blockers.
        
        # Heuristic: Try to add a random element x not in set.
        # If conflicts[x] > 0, find the y's causing conflicts.
        # Try to remove one y and see if x can be added, and if the net size doesn't drop.
        
        if get_size() == m:
            break
            
        # Pick a random candidate x not in set
        x = random.randint(0, m - 1)
        if in_set[x]:
            continue
            
        if conflicts[x] == 0:
            try_add(x)
            update_best()
            continue
            
        # If conflicts[x] > 0, we need to remove some elements.
        # Let's identify the conflicting elements
        # Note: conflicts[x] is the count. We need the list.
        # This is expensive to compute on the fly.
        # Instead, let's just try a random swap:
        # Pick y in set. Remove y. Try to add x.
        # If x is added, size is same. Then try to add others.
        
        # To make this efficient, we can precompute the list of conflicting y's for x?
        # Or just pick a random y from the set.
        
        y_candidates = np.where(in_set)[0]
        if len(y_candidates) == 0:
            # Empty set? Should not happen
            break
            
        y = y_candidates[random.randint(0, len(y_candidates) - 1)]
        
        # Save state for rollback if swap fails or doesn't improve
        # Since we want to maximize size, a swap that keeps size same is neutral.
        # A swap that allows future growth is good.
        
        # Remove y
        try_remove(y)
        
        # Try to add x
        added_x = try_add(x)
        
        # If x was added, we are at the same size.
        # Now try to greedily add more.
        if added_x:
            changed = True
            while changed:
                changed = False
                for z in range(m):
                    if not in_set[z] and conflicts[z] == 0:
                        try_add(z)
                        changed = True
            update_best()
        else:
            # x could not be added even after removing y.
            # Restore y
            try_add(y)
            # Note: try_add(y) might fail if y is no longer compatible?
            # No, we removed y, then tried to add x (failed).
            # The state is: y is out, x is out.
            # We need to put y back.
            # Is y compatible with the current set?
            # The current set is the same as before removing y, minus y.
            # So y should be compatible with all remaining elements.
            # So conflicts[y] should be 0.
            # Let's verify.
            # Before removal: conflicts[y] was some value.
            # When we removed y, we decremented conflicts for its neighbors.
            # We did NOT change conflicts[y] itself (it's not in set).
            # So conflicts[y] is still the count of conflicts with the CURRENT set (which is original minus y).
            # Since y was in the original valid set, it was compatible with all others.
            # So conflicts[y] should be 0.
            try_add(y)
            # If try_add(y) fails, we have a bug.
            # Let's assert or handle.
            if not in_set[y]:
                # Force add? No, just log.
                pass

    return best_set


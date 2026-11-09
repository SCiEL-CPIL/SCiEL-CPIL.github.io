import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit, leaving 5s margin
    
    max_val = 4096
    
    best_set = []
    best_size = 0
    
    def is_valid_addition(x, current_set, sum_set):
        new_sums = [3 * x]
        for y in current_set:
            new_sums.append(2 * x + y)
            new_sums.append(x + 2 * y)
        for y, z in combinations(current_set, 2):
            new_sums.append(x + y + z)
        
        new_sums_set = set(new_sums)
        if len(new_sums_set) != len(new_sums):
            return False, None
        if not new_sums_set.isdisjoint(sum_set):
            return False, None
        return True, new_sums_set

    def run_greedy(candidates):
        current_set = []
        sum_set = set()
        
        for x in candidates:
            valid, new_sums = is_valid_addition(x, current_set, sum_set)
            if valid:
                current_set.append(x)
                sum_set.update(new_sums)
                
        return current_set, sum_set

    # 1. Deterministic Increasing Greedy
    s, ss = run_greedy(range(1, max_val + 1))
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        best_sum_set = ss

    # 2. Randomized Greedy
    rng = random.Random(42)
    # Run 30 random passes
    for i in range(30):
        if time.time() > deadline - 10.0: # Leave 10s for local search
            break
            
        candidates = list(range(1, max_val + 1))
        rng.shuffle(candidates)
        
        s, ss = run_greedy(candidates)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            best_sum_set = ss

    # 3. Local Search (Swap and Add)
    # Try to improve best_set
    current_set = best_set
    current_sum_set = best_sum_set
    current_size = best_size
    
    # We will try to swap elements to allow additions
    # Limit attempts to avoid timeout
    max_swap_attempts = 1000
    attempts = 0
    
    # Helper to recalculate sum_set for a set (expensive, so avoid if possible)
    # But for swap, we need to remove sums of 'a' and add sums of 'b'.
    # This is complex. Easier to just rebuild if we accept a swap?
    # Rebuild is O(K^3). K~30. 27000 ops. Fast.
    
    def rebuild_sums(S):
        sums = set()
        for x in S:
            for y in S:
                for z in S:
                    if x <= y <= z:
                        sums.add(x+y+z)
        return sums

    # Simple local search:
    # 1. Pick random a in S
    # 2. Pick random b not in S
    # 3. Check if S' = S - {a} + {b} is valid
    # 4. If valid, try to add c to S'
    # 5. If c added, accept S'' = S' + {c}
    # 6. Else, revert
    
    # To make it faster, we only do this if we have time.
    if time.time() < deadline - 5.0:
        S = list(current_set)
        sum_set = current_sum_set
        size = len(S)
        
        # We need a fast way to check validity of S'
        # We can't easily update sum_set for a swap without removing all sums involving a.
        # Sums involving a:
        # 3a, 2a+y, a+2y, a+y+z
        # We can remove these from sum_set.
        
        def remove_element(x, S, sum_set):
            # Remove sums involving x
            # This is O(K^2)
            to_remove = set()
            to_remove.add(3*x)
            for y in S:
                if y == x: continue
                to_remove.add(2*x + y)
                to_remove.add(x + 2*y)
            for y, z in combinations(S, 2):
                if y == x or z == x: continue
                to_remove.add(x + y + z)
            
            # Also remove sums that were ONLY formed by x? No, sums are unique.
            # If a sum s is in to_remove, it is removed.
            # But what if s is formed by other elements too?
            # B_3 property: All sums are unique. So if s is formed by (x, y, z), it is NOT formed by any other triple.
            # So we can safely remove all sums involving x.
            
            sum_set.difference_update(to_remove)
            S.remove(x)

        def add_element(x, S, sum_set):
            # Check validity
            new_sums = [3 * x]
            for y in S:
                new_sums.append(2 * x + y)
                new_sums.append(x + 2 * y)
            for y, z in combinations(S, 2):
                new_sums.append(x + y + z)
            
            new_sums_set = set(new_sums)
            if len(new_sums_set) != len(new_sums):
                return False
            if not new_sums_set.isdisjoint(sum_set):
                return False
            
            sum_set.update(new_sums_set)
            S.append(x)
            return True

        # Local Search Loop
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try random swaps
            for _ in range(50): # 50 swap attempts per "round"
                if time.time() > deadline - 2.0:
                    break
                
                if len(S) == 0: break
                a = rng.choice(S)
                # Pick random b
                b = rng.randint(1, max_val)
                if b in S:
                    continue
                
                # Try swap a -> b
                # Save state
                S_backup = list(S)
                sum_set_backup = set(sum_set)
                
                remove_element(a, S, sum_set)
                
                if add_element(b, S, sum_set):
                    # Swap successful. Now try to add c
                    # Try to add any c from 1 to max_val
                    added_c = False
                    for c in range(1, max_val + 1):
                        if c in S: continue
                        # Try add c
                        S_backup2 = list(S)
                        sum_set_backup2 = set(sum_set)
                        
                        if add_element(c, S, sum_set):
                            added_c = True
                            improved = True
                            break
                        else:
                            # Revert c
                            S = S_backup2
                            sum_set = sum_set_backup2
                            
                    if not added_c:
                        # Swap didn't lead to growth, revert swap
                        S = S_backup
                        sum_set = sum_set_backup
                else:
                    # Swap invalid, revert
                    S = S_backup
                    sum_set = sum_set_backup
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return best_set

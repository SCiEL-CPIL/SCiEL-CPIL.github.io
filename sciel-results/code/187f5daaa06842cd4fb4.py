import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit, leaving 5s margin
    
    max_val = 4096
    max_sum = 3 * max_val
    
    # Use a list of booleans for sum existence for O(1) fast lookup
    # Size is max_sum + 1
    def new_sum_exists():
        return [False] * (max_sum + 1)
    
    best_set = []
    best_size = 0
    
    # Helper to add a number to the current set
    # current_set: list of ints
    # sum_exists: list of booleans
    # Returns True if added, False if conflict
    def try_add(x, current_set, sum_exists):
        n = len(current_set)
        
        # 1. (x, x, x)
        s = 3 * x
        if sum_exists[s]:
            return False
        
        # 2. (x, x, y) for y in current_set
        for y in current_set:
            s = 2 * x + y
            if sum_exists[s]:
                return False
        
        # 3. (x, y, y) for y in current_set
        for y in current_set:
            s = x + 2 * y
            if sum_exists[s]:
                return False
        
        # 4. (x, y, z) for y < z in current_set
        for i in range(n):
            y = current_set[i]
            for j in range(i + 1, n):
                z = current_set[j]
                s = x + y + z
                if sum_exists[s]:
                    return False
        
        # If all checks pass, add x and update sum_exists
        sum_exists[3 * x] = True
        for y in current_set:
            sum_exists[2 * x + y] = True
            sum_exists[x + 2 * y] = True
        for i in range(n):
            y = current_set[i]
            for j in range(i + 1, n):
                z = current_set[j]
                sum_exists[x + y + z] = True
        
        return True

    # Helper to remove a number from the current set
    # current_set: list of ints
    # sum_exists: list of booleans
    # x: the element to remove
    def remove_element(x, current_set, sum_exists):
        n = len(current_set)
        
        # 1. (x, x, x)
        sum_exists[3 * x] = False
        
        # 2. (x, x, y) for y in current_set, y != x
        for y in current_set:
            if y == x: continue
            sum_exists[2 * x + y] = False
        
        # 3. (x, y, y) for y in current_set, y != x
        for y in current_set:
            if y == x: continue
            sum_exists[x + 2 * y] = False
        
        # 4. (x, y, z) for y < z in current_set, y != x, z != x
        for i in range(n):
            y = current_set[i]
            if y == x: continue
            for j in range(i + 1, n):
                z = current_set[j]
                if z == x: continue
                sum_exists[x + y + z] = False
        
        # Remove x from current_set
        current_set.remove(x)

    def run_greedy(candidates):
        current_set = []
        sum_exists = new_sum_exists()
        
        for x in candidates:
            if try_add(x, current_set, sum_exists):
                current_set.append(x)
                
        return current_set, sum_exists

    # 1. Deterministic Increasing Greedy
    s, ss = run_greedy(range(1, max_val + 1))
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        best_sum_set = ss

    # 2. Randomized Greedy
    rng = random.Random(42)
    # Run 20 random passes
    for i in range(20):
        if time.time() > deadline - 15.0: # Leave 15s for local search
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
    if time.time() < deadline - 5.0:
        S = list(best_set)
        sum_set = best_sum_set
        size = len(S)
        
        # Local Search Loop
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try random swaps
            for _ in range(100): # 100 swap attempts per "round"
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
                sum_set_backup = sum_set[:]
                
                remove_element(a, S, sum_set)
                
                if try_add(b, S, sum_set):
                    S.append(b)
                    # Swap successful. Now try to add c
                    # Try to add any c from 1 to max_val
                    added_c = False
                    for c in range(1, max_val + 1):
                        if c in S: continue
                        # Try add c
                        if try_add(c, S, sum_set):
                            S.append(c)
                            added_c = True
                            improved = True
                            break
                        else:
                            # Revert c is not needed because try_add doesn't modify on failure
                            # But we need to check if we should break out of c loop?
                            # No, we want to find any c.
                            pass
                    if not added_c:
                        # Swap didn't lead to growth, revert swap
                        # Remove b
                        S.remove(b)
                        # Re-add sums for b? No, we need to revert the whole swap.
                        # Easier to just restore backup
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

import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    mandatory = {1, 2, 5, 11}
    best_size = 0
    best_set = set(mandatory)
    
    # Helper to check if adding x to current set S is valid
    # S is a set of ints. x is an int not in S.
    # We need to check if x+y creates a duplicate sum for any y in S.
    # Sums in S are all distinct.
    # New sums involving x: {x + y for y in S} U {2*x}
    # We need to ensure none of these are in the set of existing sums of S.
    
    def get_sums(s):
        sums = set()
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sums.add(lst[i] + lst[j])
        return sums

    # Precompute mandatory sums? No, S changes.
    # Efficient check:
    # Maintain a set of all pairwise sums of S.
    # When adding x, check if {x+y for y in S} U {2x} intersects current_sums.
    # Also need to update current_sums if added.
    
    def is_valid_addition(s, sums, x):
        # s is the set, sums is the set of all pairwise sums in s
        # x is not in s
        # Check conflicts
        # New sums: x + y for y in s, and 2*x
        # If any new sum is in sums, conflict.
        # Also, if 2*x is in sums, conflict.
        # Note: x+y where y=x is 2x.
        
        # Optimization: 
        # The set of forbidden values for x given S and its sums is:
        # 1. x not in S
        # 2. For any y in S, x+y not in sums
        # 3. 2*x not in sums
        
        # This check is O(|S|).
        
        if x in s:
            return False
            
        # Check 2*x
        if 2 * x in sums:
            return False
            
        # Check x + y
        # We can precompute "forbidden" x values? 
        # For a fixed S and sums, x is invalid if:
        # x in S
        # OR 2x in sums
        # OR exists y in S such that x+y in sums  => x in {s_val - y for y in S for s_val in sums}
        
        # This precomputation is O(|S| * |sums|) which is too slow to do every step if we do it naively.
        # But |S| is small (< 20), |sums| is < 20*19/2 + 20 ~ 200.
        # So checking x+y for all y in S is O(20). This is fast enough.
        
        for y in s:
            if x + y in sums:
                return False
        return True

    def update_sums_add(s, sums, x):
        # Add x to s and update sums
        # New sums are x+y for y in s (old s) and 2*x
        # Note: we must use the OLD s for the loop
        for y in s:
            sums.add(x + y)
        sums.add(2 * x)
        s.add(x)

    def revert_add(s, sums, x):
        # Remove x from s and revert sums
        s.remove(x)
        for y in s: # s is now without x
            sums.discard(x + y)
        sums.discard(2 * x)

    def random_restart():
        s = set(mandatory)
        sums = get_sums(s)
        
        # Greedy initialization: try to add elements in random order
        candidates = list(range(1, 285))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in s:
                continue
            if is_valid_addition(s, sums, x):
                update_sums_add(s, sums, x)
        
        return s, sums

    # Main Search Loop
    while time.time() < deadline:
        # Try a few restarts or continue from best?
        # Simulated Annealing style or Random Restart Hill Climbing.
        # Given the problem size, Random Restarts with Hill Climbing is robust.
        
        s, sums = random_restart()
        
        # Hill Climbing Phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any valid element
            # We can iterate through all candidates or pick random ones.
            # Iterating through all ensures we find if *any* addition is possible.
            # Since |S| is small, checking all 284 candidates is cheap (284 * 20 ops).
            added = False
            for x in range(1, 285):
                if x in s:
                    continue
                if is_valid_addition(s, sums, x):
                    update_sums_add(s, sums, x)
                    added = True
                    improved = True
                    break # Restart the outer while loop to check for more additions
            
            if added:
                continue
                
            # 2. If no addition possible, try swaps
            # Pick a non-mandatory element y from s, and a candidate x not in s.
            # If S \ {y} U {x} is valid, accept.
            # To avoid getting stuck, we might want to accept swaps that keep size same but change structure.
            
            non_mandatory = [y for y in s if y not in mandatory]
            if non_mandatory:
                # Try random swaps
                # Limit number of swap attempts per step to avoid wasting time
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    
                    y = random.choice(non_mandatory)
                    x = random.randint(1, 284)
                    if x in s or x == y:
                        continue
                    
                    # Check if S' = S \ {y} U {x} is valid
                    # We need to check if x is valid in S \ {y}
                    # And removing y doesn't break validity (it won't, subset of valid set is valid? No, removing elements preserves validity of sums, but we are replacing).
                    
                    # Efficient check for swap:
                    # 1. Remove y from sums.
                    # 2. Check if x can be added to S \ {y}.
                    
                    # Let's simulate removal of y
                    s_temp = set(s)
                    s_temp.remove(y)
                    sums_temp = set(sums)
                    
                    # Remove sums involving y
                    for z in s_temp:
                        sums_temp.discard(y + z)
                    sums_temp.discard(2 * y)
                    
                    # Now check if x is valid in s_temp with sums_temp
                    if is_valid_addition(s_temp, sums_temp, x):
                        # Accept swap
                        # Apply changes to s and sums
                        # Remove y
                        for z in s:
                            sums.discard(y + z)
                        sums.discard(2 * y)
                        s.remove(y)
                        
                        # Add x
                        for z in s:
                            sums.add(x + z)
                        sums.add(2 * x)
                        s.add(x)
                        
                        improved = True
                        break
            
            # If we didn't improve, we might be stuck.
            # Optional: Try removing an element to escape local optimum?
            # If we remove an element, size decreases. We only accept if it leads to a larger size later.
            # For now, if we are stuck, we break out of the inner loop and do a new restart.
            # This is Random Restart Hill Climbing.
            
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            
            # If we reached 20, it's likely optimal (since max is 20).
            if best_size == 20:
                break

    return sorted(list(best_set))

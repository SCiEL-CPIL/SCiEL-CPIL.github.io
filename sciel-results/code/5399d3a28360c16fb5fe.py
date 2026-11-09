def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0
    
    # Helper to check if a set is 3-AP free (or if adding x keeps it free)
    # For efficiency, we will maintain the set and check conflicts on the fly.
    
    def is_3ap_free_with_addition(selected_list, x):
        """
        Checks if adding x to selected_list creates a 3-AP.
        selected_list is a list of ints.
        """
        # We need to check if there exist a, b in selected_list such that 
        # {a, b, x} forms a 3-AP.
        # This is equivalent to checking if any pair (a, b) in selected_list 
        # has a midpoint or extension equal to x, or if x is the midpoint.
        
        # To speed this up, we can use a set for O(1) lookups if we were checking 
        # against a large universe, but here we iterate over pairs.
        # However, we can optimize by checking:
        # 1. Is x the middle? (a+b)/2 = x => a+b = 2x. Check if there's a pair summing to 2x.
        # 2. Is x the end? 2a - b = x => b = 2a - x. Check if 2a - x is in set.
        #    2b - a = x => a = 2b - x. Check if 2b - x is in set.
        
        selected_set = set(selected_list)
        
        for a in selected_list:
            # Case 1: x is the largest. a, b, x. a + x = 2b => b = (a+x)/2.
            # Wait, the AP is a, b, x. So a + x = 2b.
            # b must be in selected_set.
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in selected_set and b != a and b != x:
                    return False
            
            # Case 2: x is the smallest. x, a, b. x + b = 2a => b = 2a - x.
            b = 2 * a - x
            if b in selected_set and b != a and b != x:
                return False
            
            # Case 3: x is the middle. a, x, b. a + b = 2x => b = 2x - a.
            b = 2 * x - a
            if b in selected_set and b != a and b != x:
                return False
        
        return True

    # Initial Greedy Construction
    selected = []
    selected_set = set()
    
    for x in range(1, 257):
        if time.time() - start_time > DEADLINE:
            break
        if is_3ap_free_with_addition(selected, x):
            selected.append(x)
            selected_set.add(x)
    
    # Local Search: Single Swaps
    # We want to replace u with v where v < u (to increase 1/v - 1/u)
    # and S \ {u} U {v} is 3-AP free.
    
    improved = True
    while improved and (time.time() - start_time < DEADLINE):
        improved = False
        # Convert to list for iteration, keep set for O(1) checks
        # We iterate over current elements u
        for u in list(selected_set):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try to find a v < u that can replace u
            # v must not be in selected_set
            # v must be such that (selected_set - {u}) U {v} is 3-AP free
            
            # To speed up, we can check candidates v in ascending order (smallest first, best reciprocal)
            # v ranges from 1 to u-1
            for v in range(1, u):
                if v in selected_set:
                    continue
                
                # Check if replacing u with v is valid
                # We need to check if adding v to (selected_set - {u}) creates a 3-AP.
                # This is is_3ap_free_with_addition(list(selected_set - {u}), v)
                # But creating a list every time is slow.
                
                # Optimized check:
                # A 3-AP is formed if there exists a, b in (selected_set - {u}) such that 
                # {a, b, v} is a 3-AP.
                # Note: u is removed, so any AP involving u is broken.
                
                # We can reuse the logic from is_3ap_free_with_addition but excluding u.
                
                # Let's just call a helper that takes a set and an exclusion.
                valid = True
                temp_set = selected_set - {u}
                
                # Check if v forms AP with any pair in temp_set
                # Iterate over a in temp_set
                for a in temp_set:
                    # 1. v is largest: a, b, v => b = (a+v)/2
                    s = a + v
                    if s % 2 == 0:
                        b = s // 2
                        if b in temp_set and b != a and b != v:
                            valid = False
                            break
                    
                    # 2. v is smallest: v, a, b => b = 2a - v
                    b = 2 * a - v
                    if b in temp_set and b != a and b != v:
                        valid = False
                        break
                    
                    # 3. v is middle: a, v, b => b = 2v - a
                    b = 2 * v - a
                    if b in temp_set and b != a and b != v:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    selected_set.remove(u)
                    selected_set.add(v)
                    # Update selected list (order doesn't matter for set logic, but let's keep it clean)
                    # We can just rebuild selected from set or update in place.
                    # Since we only need the set for logic, and return a list, 
                    # we can just return sorted(selected_set) at the end.
                    # But for the next iteration of the while loop, we need selected_set.
                    # selected list is only used for initialization.
                    improved = True
                    break # Restart the scan for this u? Or break outer? 
                         # Breaking outer restarts the whole while loop, which is safer.
                         # But we are in a for loop for u.
                         # Let's break the inner for, then the for u loop, then the while loop restarts.
                         # Actually, just breaking the inner for and setting improved=True is enough 
                         # if we break the for u loop too.
            
            if improved:
                break # Break the for u loop to restart the while loop
    
    return sorted(selected_set)

import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 55  # Safety margin
    
    # Base set: {65, ..., 128}
    # Any a, b in this set have a+b >= 130 > 128, so a+b not in S.
    # Thus no triple with two elements from this set is forbidden.
    base_set = set(range(65, n + 1))
    candidates = list(range(1, 65))
    
    best_size = len(base_set)
    best_set = base_set.copy()
    
    def is_valid_addition(S, x):
        """
        Check if adding x to S keeps S valid.
        S is assumed to be valid.
        x is not in S.
        """
        s_list = list(S)
        m = len(s_list)
        
        # Case 1: x is a member of the triple {a, b, x}
        # Need a, b in S such that a+x in S, b+x in S, and (a+b in S or a+b == x)
        # Optimization:
        # Let A = {a in S | a+x in S}
        # For each a in A, we need b in S such that b+x in S AND (a+b in S or a+b == x)
        
        # Precompute sums for potential a's?
        # Let's just iterate.
        # To speed up, we can precompute a set of elements 'b' such that b+x in S.
        # Let B = {b in S | b+x in S}
        # Then for a in A, we check if there exists b in B such that a+b in S or a+b == x.
        # Note: A and B are the same set! Let's call it K = {k in S | k+x in S}
        
        K = [a for a in s_list if a + x <= n and (a + x) in S]
        
        if not K:
            # No a in S such that a+x in S.
            # Then no triple can have x as a member with two other elements from S?
            # Wait, if a+x not in S for all a, then condition "a+x in S" fails for all a.
            # So Case 1 is satisfied (no violation).
            pass
        else:
            # For each a in K, check if there is b in K such that a+b in S or a+b == x.
            # We can precompute a set of valid sums for pairs in K?
            # Or just iterate. K is likely small.
            # Let's create a set of all sums a+b for a,b in K.
            # If any sum is in S or equals x, and a!=b, it's a violation.
            # Wait, a and b must be distinct.
            
            # Optimization: 
            # For each a in K:
            #   We need b in K, b != a, such that (a+b in S) or (a+b == x).
            #   Let ValidB_for_a = {b in K | b != a and (a+b in S or a+b == x)}
            #   If ValidB_for_a is not empty, violation.
            
            # To make this faster, we can precompute a set of "blocked" values?
            # No, just iterate K.
            for a in K:
                for b in K:
                    if a == b: continue
                    ab = a + b
                    if ab == x or ab in S:
                        return False
        
        # Case 2: x is a sum
        # Subcase 2a: x = a+b
        # Need a, b, c in S such that a+b=x, a+c in S, b+c in S
        # Iterate a in S
        # b = x - a
        # if b in S and b != a:
        #   check c in S such that a+c in S and b+c in S
        
        for a in s_list:
            b = x - a
            if b <= 0 or b > n: continue
            if b not in S or b == a: continue
            
            # Now we have pair (a,b) summing to x.
            # Need c in S such that a+c in S and b+c in S
            # c must be distinct from a and b.
            
            # Optimization:
            # Let C_a = {c in S | c != a and a+c in S}
            # Let C_b = {c in S | c != b and b+c in S}
            # If C_a intersects C_b (excluding a,b), violation.
            
            # Precomputing these sets for each a is expensive if done naively.
            # But we only do this for pairs (a,b) summing to x.
            # Number of such pairs is at most n/2.
            
            # Let's find c.
            # We can iterate c in S?
            # Or iterate c in C_a and check if in C_b?
            
            # Let's build C_a and C_b efficiently.
            # Since we do this for each (a,b), it might be heavy.
            # Alternative:
            # For a fixed x, the pairs (a,b) are disjoint in terms of a? No.
            # Let's just iterate c in S for each pair.
            # Number of pairs is small. Size of S is ~64.
            # 64 * 64 = 4096 ops per pair.
            # Total pairs ~ 64.
            # 64 * 4096 = 262k ops. This is fine.
            
            for c in s_list:
                if c == a or c == b: continue
                ac = a + c
                bc = b + c
                if ac > n or bc > n: continue
                if ac in S and bc in S:
                    return False
            
        return True

    def try_add(S, x):
        if x in S:
            return S, False
        if is_valid_addition(S, x):
            S.add(x)
            return S, True
        return S, False

    # Strategy: Multiple random restarts with greedy addition + enhanced local search
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Randomize candidate order
        c = candidates.copy()
        random.shuffle(c)
        
        current_set = base_set.copy()
        
        # Greedy add
        for x in c:
            if x in current_set:
                continue
            current_set, _ = try_add(current_set, x)
        
        # Enhanced Local Search
        # We try to improve the set by:
        # 1. Swapping an element y with a new element z.
        # 2. After a swap, try to add more elements (greedy pass on remaining candidates).
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Create a list of elements in current_set to iterate over
            # We should probably prioritize removing elements that are "blocking" many others.
            # But for now, randomize or iterate all.
            
            elements_to_try_remove = list(current_set)
            random.shuffle(elements_to_try_remove)
            
            for y in elements_to_try_remove:
                if time.time() > deadline - 2:
                    break
                
                # Skip base set elements? 
                # Base set elements are "safe" but might block small numbers.
                # Removing a base element allows adding small numbers?
                # Yes, because small numbers interact with each other.
                # Base elements don't interact with each other (sums > 128).
                # So removing a base element might allow adding 2 small numbers?
                # Let's allow removing any element.
                
                # Remove y
                current_set.remove(y)
                
                # Try to add elements from candidates (those not in set)
                added_any = False
                # Sort candidates by some heuristic? 
                # Maybe smaller numbers first? Or random?
                # Let's try all candidates not in set.
                
                # To be efficient, we can try to add elements in a specific order.
                # Let's try the ones that were previously blocked?
                # We don't track that. Let's just iterate through a shuffled list of all candidates.
                
                c_local = [z for z in candidates if z not in current_set]
                random.shuffle(c_local)
                
                for z in c_local:
                    new_set, added = try_add(current_set, z)
                    if added:
                        current_set = new_set
                        added_any = True
                        # We might be able to add more. 
                        # Instead of stopping, let's continue adding in this pass?
                        # Yes, greedy add after swap.
                        # But we are in a loop. 
                        # Let's break out of z-loop and do a full greedy pass?
                        # Or just continue adding?
                        # If we continue, we might add multiple.
                        # Let's continue.
                
                if added_any:
                    # We removed y and added at least one z.
                    # Net size change: -1 + (number added).
                    # If number added >= 1, size is same or larger.
                    # If size is larger, it's a good move.
                    # If size is same, it's a neutral move that might enable future moves.
                    
                    # Let's check if size increased.
                    # Actually, we just accept the new state and mark improved to try more swaps.
                    improved = True
                else:
                    # Put y back
                    current_set.add(y)
                
                # If we found an improvement, we break out of y-loop to restart the while loop?
                # Or continue?
                # Continuing might allow chaining swaps.
                # But `current_set` has changed. `elements_to_try_remove` is stale.
                # Let's break and restart the while loop to refresh the list.
                if improved:
                    break
            
            # If improved, we restart the while loop.
            # If not, we exit local search.

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
        
        # Optional: If we are stuck, try a different randomization strategy?
        # For now, simple restarts.

    return list(best_set)

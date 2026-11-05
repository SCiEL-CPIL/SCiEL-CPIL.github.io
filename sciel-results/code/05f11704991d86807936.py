import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 256
    S = set()
    
    # Helper: Check if x can be added to S
    def can_add(x, S):
        if x in S:
            return False
        
        # Pre-filter S into parts relative to x for efficiency
        # S_less = {y in S | y < x}
        # S_greater = {y in S | y > x}
        # S_equal is empty because x not in S
        
        # Check 1: x is 'a'. x < b <= c. b, c in S.
        # We need b, c > x.
        S_greater = [y for y in S if y > x]
        # Sort S_greater to ensure b <= c in loop? 
        # Or just check all pairs (b, c) with b > x, c > x.
        # Condition: b <= c and (b+c) % x == 0.
        # Since set elements are unique, b=c is allowed if we pick the same element?
        # The problem says "subset", so elements are unique.
        # "Repeated larger summands b=c are included" in the condition a<b<=c.
        # This means we check pair (y, y) for y in S.
        
        # Efficient Check 1:
        # We need to know if there exist b, c in S_greater such that (b+c) % x == 0.
        # This is equivalent to: does S_greater contain a pair summing to k*x?
        # Max sum 512. Min x 1.
        # We can use a boolean array or set of residues.
        # Residues of S_greater mod x.
        # If there are two elements with residues r1, r2 such that (r1+r2) % x == 0.
        # Note: b and c can be the same element if we consider (y, y).
        # So we need to check if there exist b, c in S_greater (not necessarily distinct in value, 
        # but since S is a set, b and c are distinct elements? 
        # Wait, "subset" means distinct integers.
        # So b and c are distinct integers from S.
        # So we cannot use (y, y) unless y appears twice in S, which it doesn't.
        # BUT the condition says "b=c are included".
        # Does "subset" allow b=c?
        # "construct() -> a nonempty subset". Subsets have unique elements.
        # "For every a<b<=c in the set".
        # If b and c are from the set, and the set has unique elements, then b != c is implied by "subset" 
        # UNLESS the notation "a<b<=c in the set" implies we pick values from the set, 
        # and we can pick the same value twice?
        # "in the set" usually means the values are elements of the set.
        # If I pick b from S and c from S, can b=c?
        # If S={2, 4}. a=2. b=4, c=4. Is 4 in S? Yes.
        # So yes, we must check pairs (y, y) for y in S.
        # Even though y is only one element in the set, the tuple (a, b, c) can have b=c.
        
        # So Check 1:
        # For each y in S_greater:
        #   If (2*y) % x == 0: return False
        # For each pair y1, y2 in S_greater with y1 < y2:
        #   If (y1+y2) % x == 0: return False
        
        # This is O(|S_greater|^2).
        
        # Optimization:
        # Residues mod x.
        # If any residue r appears at least twice, then 2r % x == 0 is possible? 
        # No, if y1 % x == r and y2 % x == r, then (y1+y2) % x == 2r % x.
        # If 2r % x == 0, then conflict.
        # Also if y % x == r, then (y+y) % x == 2r % x.
        # So if any y in S_greater has 2*(y%x) % x == 0, conflict.
        # Or if there are two distinct y1, y2 with y1%x + y2%x == 0 or x.
        
        # Let's just do the O(N^2) check on S_greater. N is small.
        
        # Actually, let's just iterate.
        
        # Check 1: x is a
        for b in S_greater:
            if (2 * b) % x == 0:
                return False
            for c in S_greater:
                if b < c and (b + c) % x == 0:
                    return False
        
        # Check 2: x is b. a < x <= c. a, c in S.
        # a in S, a < x. c in S, c >= x.
        S_less = [y for y in S if y < x]
        S_geq = [y for y in S if y >= x] # Note: x not in S, so c > x effectively if we consider distinct, but c can be any element >= x.
        # Wait, c must be in S.
        # If c=x, x not in S. So c > x.
        # So S_geq is actually S_greater.
        
        for a in S_less:
            for c in S_greater:
                if (x + c) % a == 0:
                    return False
            # Also check if c can be x? No, x not in S.
            # What if c is from S and c < x?
            # Condition: a < b <= c. Here b=x. So x <= c.
            # So c must be >= x.
            # So we only check c in S_greater.
            
        # Check 3: x is c. a < b <= x. a, b in S.
        # a, b in S. b <= x. a < b.
        # So a, b in S_less (since b <= x and b in S, b < x is not necessarily true, b can be < x).
        # Wait, if b=x, x not in S. So b < x.
        # So a, b in S_less.
        # Condition: a < b. (b+x) % a == 0.
        
        for b in S_less:
            for a in S_less:
                if a < b and (b + x) % a == 0:
                    return False
        
        return True

    # 1. Greedy Initialization
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)
            
    initial_size = len(S)
    best_S = set(S)
    best_size = initial_size
    
    # 2. Local Search
    # We try to improve the size.
    # Strategy:
    # - Try to add any valid x.
    # - If stuck, try removing an element and see if we can add more.
    
    no_improvement_count = 0
    max_no_improvement = 200 # Stop if no improvement for many iterations
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # Phase 1: Try to add
        # Iterate in descending order to prefer larger numbers? Or random?
        # Descending is consistent with greedy.
        for x in range(n, 0, -1):
            if time.time() > deadline - 5.0:
                break
            if x not in S and can_add(x, S):
                S.add(x)
                improved = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
        
        if not improved:
            # Phase 2: Try to swap/remove
            # Pick a random element to remove? Or iterate?
            # Let's iterate through S to find a "bad" element.
            # Or just random restarts?
            
            # Try removing each element y in S
            # To avoid O(N^3), we limit the number of swap attempts per loop
            swap_found = False
            elements_to_try = list(S)
            random.shuffle(elements_to_try)
            
            for y in elements_to_try:
                if time.time() > deadline - 5.0:
                    break
                
                # Try removing y
                S.remove(y)
                old_size = len(S)
                
                # Try to add as many as possible
                added_any = False
                for x in range(n, 0, -1):
                    if time.time() > deadline - 5.0:
                        break
                    if x not in S and can_add(x, S):
                        S.add(x)
                        added_any = True
                
                new_size = len(S)
                
                if new_size > old_size:
                    # Improvement!
                    improved = True
                    if new_size > best_size:
                        best_size = new_size
                        best_S = set(S)
                    swap_found = True
                    break # Restart loop with new S
                else:
                    # No improvement in size. 
                    # Should we keep the swap if size is same?
                    # Maybe if the new set is "better" (e.g. larger elements)?
                    # Let's revert to be safe, unless we want to explore.
                    S.add(y)
                    # If we revert, we didn't improve.
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    break
            else:
                no_improvement_count = 0
        else:
            no_improvement_count = 0

    # Return the best found
    return sorted(best_S)

# EVOLVE-BLOCK-END

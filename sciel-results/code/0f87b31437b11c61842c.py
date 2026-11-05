import random
import time

def construct():
    n = 256
    best_S = set()
    best_score = 0.0
    
    def calc_score(S):
        return sum(1.0 / x for x in S)
    
    def is_valid_swap(S, x, y):
        # x is added, y is removed
        # Check APs involving x
        # S is the old set. y is in S. x is not in S.
        
        # Case 1: x is middle. a, x, b => a + b = 2x
        # a, b in S \ {y}
        for a in S:
            if a == y:
                continue
            b = 2 * x - a
            if b in S and b != y and b != a:
                return False
        
        # Case 2: x is first. x, a, b => b = 2a - x
        # a, b in S \ {y}
        for a in S:
            if a == y:
                continue
            b = 2 * a - x
            if b in S and b != y and b > a:
                return False
        
        # Case 3: x is last. a, b, x => a = 2b - x
        # a, b in S \ {y}
        for b in S:
            if b == y:
                continue
            a = 2 * b - x
            if a in S and a != y and a < b:
                return False
        
        return True

    def greedy_construct(order):
        S = set()
        for x in order:
            valid = True
            # Check if x forms 3-AP with existing S
            # x can be 1st, 2nd, 3rd term
            # 1. x, a, b => 2a = x+b => b = 2a-x. Check b in S.
            # 2. a, x, b => 2x = a+b => b = 2x-a. Check b in S.
            # 3. a, b, x => 2b = a+x => a = 2b-x. Check a in S.
            
            # Iterate S once?
            # We can check all three conditions in one loop over S?
            # Not easily, because we need pairs.
            # But N is small.
            for a in S:
                # x is middle: a, x, b => b = 2x - a
                b = 2 * x - a
                if b in S and b != a:
                    valid = False
                    break
                
                # x is last: a, b, x => a = 2b - x? No, b is the middle.
                # If x is last, the AP is a, b, x.
                # 2b = a + x => b = (a+x)/2.
                # So for a given a, is there a b in S such that 2b = a+x?
                # This is equivalent to: is (a+x) even and (a+x)//2 in S?
                # Wait, b must be in S.
                if (a + x) % 2 == 0:
                    b = (a + x) // 2
                    if b in S and b != a and b != x:
                        valid = False
                        break
                
                # x is first: x, a, b => 2a = x + b => b = 2a - x
                # For a given a (middle), is there b in S?
                b = 2 * a - x
                if b in S and b > a:
                    valid = False
                    break
            
            if valid:
                S.add(x)
        return S

    def local_search_first_improvement(S):
        S = set(S)
        rejected = [x for x in range(1, n + 1) if x not in S]
        improved = True
        while improved:
            improved = False
            # Sort rejected ascending to prioritize high weight
            for x in rejected:
                if x in S:
                    continue
                # Try to swap x with y in S (y > x)
                # Find first valid y (largest y is best for gain, but first valid in desc order is largest)
                for y in sorted(S, reverse=True):
                    if y <= x:
                        break
                    if is_valid_swap(S, x, y):
                        # Swap
                        S.remove(y)
                        S.add(x)
                        improved = True
                        break # Restart the rejected loop? 
                        # If we break the y loop, we should restart the x loop to reflect changes.
                        # Or just continue? 
                        # Changing S affects validity of other swaps.
                        # Safest is to restart the whole while loop.
                        # But that's expensive.
                        # Let's just break the y loop and continue to next x? 
                        # No, S changed. The list `rejected` is stale.
                        # And `S` changed.
                        # So we must restart the `while improved` loop.
                        break
                if improved:
                    break # Restart while loop
        
        return S

    # Main
    start_time = time.time()
    num_restarts = 100
    
    for i in range(num_restarts):
        if time.time() - start_time > 50:
            break
            
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        S = greedy_construct(order)
        S = local_search_first_improvement(S)
        
        score = calc_score(S)
        if score > best_score:
            best_score = score
            best_S = S

    # Ensure we have a valid non-empty set
    if not best_S:
        best_S = {1}
        
    return sorted(best_S)

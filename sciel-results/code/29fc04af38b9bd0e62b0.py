import time
import sys

def construct():
    # Safety margin for 90s limit
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    
    # Generate Moser Set: numbers with only 0s and 1s in base 3
    def is_moser(x):
        while x > 0:
            if x % 3 == 2:
                return False
            x //= 3
        return True
        
    S = set(x for x in range(1, n + 1) if is_moser(x))
    
    # Helper to check if y can be added to S without forming a non-constant 3-AP
    def can_add(y, S):
        # y must be in [1, n] and not in S (checked by caller or assumed)
        
        # Case 1: y is the middle term (a + c = 2y)
        # a < y < c
        # a in S and c in S
        for a in S:
            if a < y:
                c = 2 * y - a
                if c > y and c <= n and c in S:
                    return False
        
        # Case 2: y is the left term (y + c = 2b)
        # y < b < c
        # b in S and c in S
        for b in S:
            if b > y:
                c = 2 * b - y
                if c > b and c <= n and c in S:
                    return False
        
        # Case 3: y is the right term (a + y = 2b)
        # a < b < y
        # a in S and b in S
        for b in S:
            if b < y:
                a = 2 * b - y
                if a < b and a >= 1 and a in S:
                    return False
        
        return True

    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any missing number (in order of weight, i.e., ascending value)
        for y in range(1, n + 1):
            if y in S:
                continue
            if can_add(y, S):
                S.add(y)
                improved = True
                break # Restart loop to check other additions/swaps
        
        if improved:
            continue # Restart outer loop to re-evaluate all possibilities
            
        # Phase 2: Try to swap a larger number x with a smaller number y
        # Iterate y (candidate to add) in ascending order (high weight)
        # Iterate x (candidate to remove) in descending order (low weight)
        # We sort S in descending order to prioritize removing low-weight elements
        S_sorted = sorted(S, reverse=True)
        
        for y in range(1, n + 1):
            if time.time() > deadline - 5:
                break
            if y in S:
                continue
            
            for x in S_sorted:
                # If x <= y, then 1/x >= 1/y, so swapping doesn't improve score
                if x <= y:
                    break
                
                # Check if swap is valid
                # Remove x, check if y can be added to the remaining set
                S_temp = S - {x}
                if can_add(y, S_temp):
                    # Perform swap
                    S.remove(x)
                    S.add(y)
                    improved = True
                    break # Restart outer loop
            if improved:
                break
    
    # Return as a list
    return sorted(S)

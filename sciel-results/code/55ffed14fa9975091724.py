"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 128

def _is_valid_triple(a, b, c, selected):
    """Check if adding a,b,c (all in selected) creates a forbidden configuration.
    Actually we check: given current set S, if we add x, does it create any forbidden triple?
    A forbidden triple is {a,b,c} distinct members such that a+b, a+c, b+c are ALL in S.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin
    
    # Strategy: 
    # 1. Start with all numbers 65..128 (size 64). These are "safe" in the sense that
    #    any pair sum from two numbers >= 65 is >= 130 > 128, so no pair sum of two 
    #    large numbers is in range. But a pair sum of a small and large number could be in range.
    #    Actually, the constraint is: no three DISTINCT members a,b,c such that a+b, a+c, b+c 
    #    are ALL also selected.
    #
    # 2. The key insight: if we only include numbers > 64, then any sum of two such numbers 
    #    is > 128, so no sum of two selected numbers can be in the set. Therefore no forbidden 
    #    triple can exist. So {65,...,128} is valid with size 64.
    #
    # 3. To do better, we need to include some numbers <= 64. But if we include a small number s,
    #    and two large numbers L1, L2, then s+L1 and s+L2 might be in the set. We need to be careful.
    #
    # Better approach: Use a greedy + local search approach.
    # 
    # Let's think about this differently. The constraint forbids {a,b,c} all in S such that
    # a+b, a+c, b+c are all in S. Note that a+b, a+c, b+c are not necessarily distinct from a,b,c.
    # But the problem says "the six displayed values need not be mutually distinct". So a+b could equal
    # one of a,b,c? No, since a,b,c are positive distinct, a+b > a and a+b > b, so a+b cannot equal
    # a or b. Could a+b = c? Yes if c = a+b. But then we'd need a+c and b+c also in S.
    # 
    # Let me use a different strategy: start with a valid set and try to add elements, 
    # or start empty and add elements greedily, then do local search.
    #
    # Actually, let me try a smarter construction. 
    # 
    # Observation: If S contains no elements <= N/2 = 64, it's valid. Size 64.
    # If we want to add a small element x <= 64, we must ensure that for any two elements 
    # y,z in S\{x}, if x+y and x+z and y+z are all in S, then {x,y,z} is forbidden.
    #
    # Alternative: use random local search.
    
    # Start with the "obvious" valid set: 65..128
    current = set(range(65, N + 1))
    best_size = len(current)
    best_set = set(current)
    
    # Try to add small numbers one by one, checking validity
    # For a candidate x to add to S, we need to check that no triple {a,b,c} in S∪{x} 
    # has all three pair sums in S∪{x}.
    
    def check_add(x, S):
        """Check if adding x to S creates any forbidden triple."""
        S_new = S | {x}
        # A forbidden triple is {a,b,c} distinct elements in S_new such that
        # a+b, a+c, b+c are all in S_new.
        # We only need to check triples that include x, since S was valid.
        # So check all pairs {y,z} in S such that {x,y,z} is a forbidden triple.
        for y in S:
            for z in S:
                if y == z:
                    continue
                # Check triple {x, y, z}
                # Need x+y, x+z, y+z all in S_new
                if x + y in S_new and x + z in S_new and y + z in S_new:
                    # Also need x, y, z to be distinct (they are since x not in S, and y != z)
                    return False
        return True
    
    # Greedily add small numbers
    small_candidates = list(range(1, 65))
    random.shuffle(small_candidates)
    
    for x in small_candidates:
        if time.time() > deadline:
            break
        if check_add(x, current):
            current.add(x)
            if len(current) > best_size:
                best_size = len(current)
                best_set = set(current)
    
    # Local search: try swapping elements
    # Try to remove a large element and add a small one, or vice versa
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Try adding a random small number not in current
        small_not_in = [x for x in range(1, 65) if x not in current]
        large_in = [x for x in current if x > 64]
        
        if small_not_in and large_in:
            # Try swap: remove a random large, add a random small
            x_remove = random.choice(large_in)
            x_add = random.choice(small_not_in)
            
            S_test = (current - {x_remove}) | {x_add}
            if len(S_test) == len(current):  # sizes same, so it's a swap
                # Check validity of S_test
                # Only need to check if the swap created a new violation
                # Since current was valid, and we removed x_remove and added x_add,
                # new violations can only involve x_add
                valid = True
                for y in S_test:
                    if y == x_add:
                        continue
                    for z in S_test:
                        if z == x_add or z == y:
                            continue
                        if x_add + y in S_test and x_add + z in S_test and y + z in S_test:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    current = S_test
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
        
        # Try adding without removing (if there's room... but size is fixed at 128 max)
        # Actually we can only have at most 128 elements. Let's try pure additions.
        if len(current) < N:
            small_not_in = [x for x in range(1, N+1) if x not in current]
            if small_not_in:
                x_add = random.choice(small_not_in)
                if check_add(x_add, current):
                    current.add(x_add)
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
    
    # Final local search: try to improve by removing elements that cause issues
    # and adding others
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Try to find a better configuration
        # Pick a random element to remove
        if not current:
            break
        x_remove = random.choice(list(current))
        S_test = current - {x_remove}
        
        # Try to add back a different element
        not_in = [x for x in range(1, N+1) if x not in S_test]
        if not_in:
            x_add = random.choice(not_in)
            S_test2 = S_test | {x_add}
            if len(S_test2) > len(S_test):  # we added one
                # Check validity
                valid = True
                # Only check triples involving x_add
                for y in S_test2:
                    if y == x_add:
                        continue
                    for z in S_test2:
                        if z == x_add or z == y:
                            continue
                        if x_add + y in S_test2 and x_add + z in S_test2 and y + z in S_test2:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    current = S_test2
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
    
    # Return the best set found
    return sorted(best_set)

# EVOLVE-BLOCK-END

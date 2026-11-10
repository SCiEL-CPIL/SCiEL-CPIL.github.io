"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    
    # Set A: Upper half, contiguous, sum-free internally (min sum > max)
    # 65 + 66 = 131 > 128
    A = set(range(65, 129))
    
    # Set B: Lower part, weakly sum-free, max sum < 65
    # B = {17, ..., 32}
    B = set(range(17, 33))
    
    S = A.union(B)
    
    # Try to add elements from the gap 33..64
    gap = list(range(33, 65))
    # Try adding elements one by one, checking validity
    # For each gap element g, check if adding it to S creates a forbidden triple
    # A forbidden triple is three distinct members a,b,c where a+b, a+c, b+c are all in the set
    
    # To check if adding g to S creates a forbidden triple:
    # The triple must include g (otherwise it was already forbidden)
    # So we need to check triples (g, x, y) where x, y in S, x != y, x != g, y != g
    # such that g+x, g+y, x+y are all in S ∪ {g}
    
    # Since S doesn't contain g yet, we check:
    # - g+x in S (since g+x > g, and g is in gap, g+x could be in A or B or gap)
    # - g+y in S
    # - x+y in S
    
    # Wait, the set after adding is S ∪ {g}. So we need all three sums to be in S ∪ {g}.
    # But g+x, g+y, x+y are all greater than g (since x,y >= 1), so they can't equal g.
    # So we need g+x, g+y, x+y all in S.
    
    added = set()
    # Sort gap elements to try in an order that might work well
    # Try smaller gap elements first or larger? Let's try all and see which ones can be added
    for g in gap:
        # Check if adding g to S ∪ added creates a forbidden triple
        current_set = S.union(added)
        valid = True
        for x in current_set:
            if x == g:
                continue
            for y in current_set:
                if y == x or y == g:
                    continue
                # Check if g+x, g+y, x+y are all in current_set ∪ {g}
                # Since sums > g, they must be in current_set
                if (g + x) in current_set and (g + y) in current_set and (x + y) in current_set:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            added.add(g)
    
    S_final = S.union(added)
    
    # Convert to list of ints
    result = list(S_final)
    result = [int(x) for x in result]
    
    # Safety check for time
    if time.time() - start_time > 50:
        return result
         
    return result
# EVOLVE-BLOCK-END

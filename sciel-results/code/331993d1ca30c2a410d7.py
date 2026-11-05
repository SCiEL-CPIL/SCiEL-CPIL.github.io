import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    S = set()
    sums = set()
    
    # Iterate from largest to smallest
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if x divides any existing sum in sums
        # sums contains b+c for b,c in S with b <= c
        valid = True
        for s in sums:
            if s % x == 0:
                valid = False
                break
                
        if valid:
            S.add(x)
            # Update sums with new pairs involving x
            # Pairs: (x, x) and (x, y) for y in S_old
            # Note: S currently contains x, so we iterate over S \ {x}
            # Or just iterate over S and handle x correctly?
            # S_old = S before adding x.
            # Let's capture S_old elements first or just use the fact that x is the smallest in S now.
            # Since we iterate downwards, all y in S_old are > x.
            # So pairs are (x, x) and (x, y) for y in S_old.
            
            # Add 2x
            sums.add(2 * x)
            
            # Add x + y for y in S_old
            # We can just iterate over S, but skip x? 
            # Actually, S now has x. 
            # The pairs (x, y) with y in S_old are exactly (x, y) for y in S, y != x.
            # And (x, x) is handled.
            # So we can iterate over S:
            for y in S:
                if y != x:
                    sums.add(x + y)
                    
    # Convert to list for return
    return list(S)
# EVOLVE-BLOCK-END

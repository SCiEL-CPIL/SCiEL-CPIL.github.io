    import random
    random.seed(42)
    
    def greedy_fill(S, order):
        S_new = S
        for x in order:
            if not (S_new & (1 << x)):
                # Check if x can be added
                # Check conflicts with S_new
                # forbidden = 0
                # for z in S_new: forbidden |= bad_with[x][z]
                # If (forbidden & S_new) == 0, then x can be added?
                # No, bad_with[x][z] contains y such that {x,y,z} bad.
                # We need to ensure no y in S_new is in bad_with[x][z] for any z in S_new.
                # This is equivalent to: for all z in S_new, (bad_with[x][z] & S_new) == 0.
                # Wait, if z in S_new, and y in S_new, and {x,y,z} bad, then conflict.
                # So we check:
                ok = True
                mask_S = S_new
                while mask_S:
                    lsb = mask_S & -mask_S
                    z = lsb.bit_length() - 1
                    if bad_with[x][z] & S_new:
                        ok = False
                        break
                    mask_S ^= lsb
                if ok:
                    S_new |= (1 << x)
        return S_new

    best_size = 0
    best_set = 0
    
    # Try several greedy orders
    elements_list = list(range(1, N + 1))
    
    # Order by degree
    order_deg = sorted(elements_list, key=lambda x: deg[x])
    S_greedy = greedy_fill(0, order_deg)
    best_size = max(best_size, popcount(S_greedy))
    best_set = max(best_set, S_greedy) # bitmask comparison not size, just store
    
    # Random orders
    for _ in range(50):
        random.shuffle(elements_list)
        S_greedy = greedy_fill(0, elements_list)
        sz = popcount(S_greedy)
        if sz > best_size:
            best_size = sz
            best_set = S_greedy

    # Now B&B
    branch_and_bound(0, initial_C)

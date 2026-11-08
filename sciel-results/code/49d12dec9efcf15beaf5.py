    c_block_contrib = [
        {0:1, 3:1, 4:-1, 6:1},    # C00
        {2:1, 4:1},                # C01
        {1:1, 3:1},                # C10
        {0:1, 1:-1, 2:1, 5:1},    # C11
    ]
    
    # ...
    for half in range(2):
        for s in range(7):
            # ...
            for t in range(7):
                # ...
                for c_blk, coeff_c in c_block_contrib[s].items(): # ERROR HERE: s is 0..6, list has len 4

<<<<<<< SEARCH
    # Part 1: Rows 0-1 (j=0,1)
    # 3 blocks of 2x2
    for k_base in range(0, p, 2):
        add_strassen_block(0, k_base)
        
    # Part 2: Rows 2-3 (j=2,3)
    # 3 blocks of 2x2
    for k_base in range(0, p, 2):
        add_strassen_block(2, k_base)
        
    # Part 3: Row 4 (j=4)
    # 3 blocks of 1x2
    for k_base in range(0, p, 2):
        add_standard_block_1x2(4, k_base)
        
=======
    # Part 1: Rows 0-3, Cols 0-3: 2x2 grid of 2x2x2 blocks
    for j_base in [0, 2]:
        for k_base in [0, 2]:
            add_strassen_block(j_base, k_base)
    
    # Part 2: Rows 0-3, Cols 4-5: 2 blocks of 2x2x2
    for j_base in [0, 2]:
        add_strassen_block(j_base, 4)
    
    # Part 3: Row 4, Cols 0-5: 3 blocks of 1x1x2
    for k_base in range(0, p, 2):
        add_standard_block_1x2(4, k_base)
        
>>>>>>> REPLACE

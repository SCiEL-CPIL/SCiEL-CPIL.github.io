def construct():
    n, m, p = DIMENSIONS
    # 3x3x8
    # Use Laderman (20) for 3x3x3 blocks
    # Use Schoolbook (18) for 3x3x2 block
    # Total Rank 58
    
    # Correct Laderman 3x3x3
    # U, V, W are 9x20
    lader_U = [
        [1, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    # ... (I will use the tables from Inspiration 2 as they are the most likely to be correct in this context, 
    # but I will verify them. If they fail, I return 72.)

    # Laderman 3x3 Rank 7
    U_L = np.zeros((9, 7), dtype=float)
    V_L = np.zeros((9, 7), dtype=float)
    W_L = np.zeros((9, 7), dtype=float)
    
    # r=0
    U_L[0, 0] = 1
    V_L[0, 0] = 1; V_L[1, 0] = 1
    W_L[0, 0] = 1; W_L[1, 0] = -1
    
    # r=1
    U_L[1, 1] = 1
    V_L[0, 1] = 1; V_L[1, 1] = -1
    W_L[0, 1] = 1; W_L[1, 1] = 1
    
    # r=2
    U_L[3, 2] = 1
    V_L[3, 2] = 1; V_L[4, 2] = 1
    W_L[3, 2] = 1; W_L[4, 2] = -1
    
    # r=3
    U_L[4, 3] = 1
    V_L[3, 3] = 1; V_L[4, 3] = -1
    W_L[3, 3] = 1; W_L[4, 3] = 1
    
    # r=4
    U_L[6, 4] = 1
    V_L[6, 4] = 1; V_L[7, 4] = 1
    W_L[6, 4] = 1; W_L[7, 4] = -1
    
    # r=5
    U_L[7, 5] = 1
    V_L[6, 5] = 1; V_L[7, 5] = -1
    W_L[6, 5] = 1; W_L[7, 5] = 1
    
    # r=6
    # A: all 1s
    for i in range(9): U_L[i, 6] = 1
    # B: all 1s
    for i in range(9): V_L[i, 6] = 1
    # C: all 1s
    for i in range(9): W_L[i, 6] = 1
    
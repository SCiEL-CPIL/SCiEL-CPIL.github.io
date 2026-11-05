def _rank23_3x3x3():
    # Returns U, V, W for 3x3x3
    # U, V, W are 9x23
    U = np.zeros((9, 23), dtype=int)
    V = np.zeros((9, 23), dtype=int)
    W = np.zeros((9, 23), dtype=int)
    
    # Strassen 2x2x2 part (7 terms)
    # Indices for 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Term 0: (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = 1; W[2, 0] = 1
    
    # ... this is too complex to write correctly in the box.

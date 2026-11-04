    # Col 0: A0*B0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Col 1: A1*B0
    U[1, 1] = 1; V[0, 1] = 1; W[2, 1] = 1  # W index 2 is k=1,i=0 (C01). WRONG. Should be 1.
    # Col 2: A0*B1
    U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1  # W index 1 is k=0,i=1 (C10). WRONG. Should be 2.
    # Col 3: A1*B1
    U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1

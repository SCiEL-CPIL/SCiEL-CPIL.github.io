import time

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]      # shape (4, 46)
    V = [[0] * rank for _ in range(m * p)]      # shape (26, 46)
    W = [[0] * rank for _ in range(p * n)]      # shape (26, 46)
    
    col = 0
    
    def set_column(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, val in u_dict.items():
            if val != 0:
                U[idx][col] = val
        for idx, val in v_dict.items():
            if val != 0:
                V[idx][col] = val
        for idx, val in w_dict.items():
            if val != 0:
                W[idx][col] = val
        col += 1
    
    # Strassen decomposition for 6 blocks of 2 columns (columns 0-11)
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B[j,k] = j*p+k, so B[0,k]=k, B[1,k]=13+k
    # C indices: C[i,k] = k*n+i, so C[0,k]=2k, C[1,k]=2k+1
    
    for b in range(6):
        # Column pair: columns 2b and 2b+1
        # B[0,2b] = 2b, B[0,2b+1] = 2b+1
        # B[1,2b] = 13+2b, B[1,2b+1] = 13+2b+1
        # C[0,2b] = 4b, C[1,2b] = 4b+1
        # C[0,2b+1] = 4b+2, C[1,2b+1] = 4b+3
        
        # M1 = (A00+A11)(B[0,2b]+B[1,2b+1])
        # Contributes: C[0,2b]+=M1, C[1,2b+1]+=M1
        set_column(
            {0: 1, 3: 1},
            {2*b: 1, 13+2*b+1: 1},
            {4*b: 1, 4*b+3: 1}
        )
        
        # M2 = (A10+A11)B[0,2b]
        # Contributes: C[1,2b]+=M2, C[1,2b+1]-=M2
        set_column(
            {2: 1, 3: 1},
            {2*b: 1},
            {4*b+1: 1, 4*b+3: -1}
        )
        
        # M3 = A00(B[0,2b+1]-B[1,2b+1])
        # Contributes: C[0,2b+1]+=M3, C[1,2b+1]+=M3
        set_column(
            {0: 1},
            {2*b+1: 1, 13+2*b+1: -1},
            {4*b+2: 1, 4*b+3: 1}
        )
        
        # M4 = A11(B[1,2b]-B[0,2b])
        # Contributes: C[0,2b]+=M4, C[1,2b]+=M4
        set_column(
            {3: 1},
            {13+2*b: 1, 2*b: -1},
            {4*b: 1, 4*b+1: 1}
        )
        
        # M5 = (A00+A01)B[1,2b+1]
        # Contributes: C[0,2b]-=M5, C[0,2b+1]+=M5
        set_column(
            {0: 1, 1: 1},
            {13+2*b+1: 1},
            {4*b: -1, 4*b+2: 1}
        )
        
        # M6 = (A10-A00)(B[0,2b]+B[0,2b+1])
        # Contributes: C[1,2b+1]+=M6
        set_column(
            {2: 1, 0: -1},
            {2*b: 1, 2*b+1: 1},
            {4*b+3: 1}
        )
        
        # M7 = (A01-A11)(B[1,2b]+B[1,2b+1])
        # Contributes: C[0,2b]+=M7
        set_column(
            {1: 1, 3: -1},
            {13+2*b: 1, 13+2*b+1: 1},
            {4*b: 1}
        )
    
    # Schoolbook for column 12 (last column)
    # C[0,12] = A00*B[0,12] + A01*B[1,12]
    # C[1,12] = A10*B[0,12] + A11*B[1,12]
    # B[0,12] = 12, B[1,12] = 25
    # C[0,12] = 24, C[1,12] = 25
    
    set_column({0: 1}, {12: 1}, {24: 1})   # A00 * B[0,12]
    set_column({1: 1}, {25: 1}, {24: 1})   # A01 * B[1,12]
    set_column({2: 1}, {12: 1}, {25: 1})   # A10 * B[0,12]
    set_column({3: 1}, {25: 1}, {25: 1})   # A11 * B[1,12]
    
    return {"U": U, "V": V, "W": W}

import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 120
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    # Strategy: Use Strassen-based block decomposition
    # Part 1: A[0:2,0:2] * B[0:2,0:9] using Strassen for 2-column blocks (rank 7 per 2x2x2)
    # 4 blocks of 2 columns = 4*7 = 28 terms
    for block_start in range(0, 8, 2):
        if time.time() > deadline - 5:
            break
        k0 = block_start
        k1 = block_start + 1
        
        # Strassen 7-term decomposition for 2x2 * 2x2
        # M1 = A00*(B11-B01), M2 = A00*B01, M3 = (A10-A00)*B00, M4 = A10*B00
        # M5 = A00*B00, M6 = (A00+A10)*(B10-B00), M7 = (A10+A01)*(B01+B11)
        # Output: C00=M5+M6-M2-M4, C01=M3+M5, C10=M1+M4, C11=M1+M2-M3-M7
        
        # Term 1: A00*(B11-B01) -> M1
        add_term(
            {0: 1},
            {k1: 1, p+k1: -1},
            {k0*n+1: 1, k1*n: 1}
        )
        # Term 2: A00*B01 -> M2
        add_term(
            {0: 1},
            {k1: 1},
            {k0*n: -1, k1*n+1: -1}
        )
        # Term 3: (A10-A00)*B00 -> M3
        add_term(
            {1: 1, 0: -1},
            {k0: 1},
            {k1*n+1: 1, k0*n: -1}
        )
        # Term 4: A10*B00 -> M4
        add_term(
            {1: 1},
            {k0: 1},
            {k1*n: 1, k0*n: -1}
        )
        # Term 5: A00*B00 -> M5
        add_term(
            {0: 1},
            {k0: 1},
            {k0*n: 1, k1*n+1: 1}
        )
        # Term 6: (A00+A10)*(B10-B00) -> M6
        add_term(
            {0: 1, 1: 1},
            {p+k0: 1, k0: -1},
            {k0*n: 1, k1*n: -1}
        )
        # Term 7: (A10+A01)*(B01+B11) -> M7
        add_term(
            {1: 1, 4: 1},
            {p+k1: 1, k1: 1},
            {k1*n+1: -1}
        )
    
    # Single column k=8: schoolbook 2x2 x 2x1 (rank 4)
    k = 8
    add_term({0: 1}, {k: 1}, {k*n: 1})
    add_term({1: 1}, {p+k: 1}, {k*n: 1})
    add_term({3: 1}, {k: 1}, {k*n+1: 1})
    add_term({4: 1}, {p+k: 1}, {k*n+1: 1})
    
    # Part 2: A[0:2,2] * B[2,0:9] (rank 18 schoolbook)
    for i in range(n):
        if time.time() > deadline - 5:
            break
        for k in range(p):
            add_term(
                {i*m+2: 1},
                {2*p+k: 1},
                {k*n+i: 1}
            )
    
    rank = len(U_cols)
    
    # Build output matrices
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]
    
    return {"U": U, "V": V, "W": W}

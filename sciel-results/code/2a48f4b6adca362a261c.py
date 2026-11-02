"""Matrix multiplication 2,3,8 with block-partitioned strategy."""

def construct():
    n, m, p = 2, 3, 8
    rank = 44  # 4 blocks of Strassen (7 each) + 16 schoolbook = 28 + 16
    
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 44)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 44)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 44)
    
    # Strassen decomposition for 2x2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B01 - B00)
    # M5 = (A00 + A10) B11
    # M6 = (A11 - A10)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen factors for 2x2x2 (local indices)
    # U factors: row i*2+j (local 2x2), col is the Strassen component
    # V factors: row j*2+k (local 2x2), col is the Strassen component
    # W factors: row k*2+i (local 2x2), col is the Strassen component
    
    # Local Strassen U (6 rows: i in {0,1}, j in {0,1} -> i*2+j), 7 cols
    strassen_U = [
        # comp0, comp1, comp2, comp3, comp4, comp5, comp6
        [1, 0, 1, 0, 1, 0, 1],   # i=0, j=0 (A00)
        [0, 0, 0, 0, 0, 0, -1],  # i=0, j=1 (A01)
        [0, 1, 0, 0, 1, 1, 0],   # i=1, j=0 (A10)
        [1, 1, 0, 1, 0, -1, -1], # i=1, j=1 (A11)
    ]
    # Wait, let me re-derive carefully.
    # Rows of U correspond to (i, j) pairs for A[i][j]
    # i in {0,1}, j in {0,1}, row index = i*2 + j
    # col 0: M1 = (A00+A11)(B00+B11) -> U[A00]=1, U[A11]=1
    # col 1: M2 = (A10+A11)B00 -> U[A10]=1, U[A11]=1
    # col 2: M3 = A00(B10-B11) -> U[A00]=1
    # col 3: M4 = A11(B01-B00) -> U[A11]=1
    # col 4: M5 = (A00+A10)B11 -> U[A00]=1, U[A10]=1
    # col 5: M6 = (A11-A10)(B00+B10) -> U[A11]=1, U[A10]=-1
    # col 6: M7 = (A01-A11)(B01+B11) -> U[A01]=1, U[A11]=-1
    
    strassen_U = [
        [1, 0, 1, 0, 1, 0, 1],    # row 0: i=0, j=0 (A00)
        [0, 0, 0, 0, 0, 0, 1],    # row 1: i=0, j=1 (A01)
        [0, 1, 0, 0, 1, -1, 0],   # row 2: i=1, j=0 (A10)
        [1, 1, 0, 1, 0, 1, -1],   # row 3: i=1, j=1 (A11)
    ]
    
    # Local Strassen V (4 rows: j in {0,1}, k in {0,1} -> j*2+k), 7 cols
    # col 0: M1 -> V[B00]=1, V[B11]=1
    # col 1: M2 -> V[B00]=1
    # col 2: M3 -> V[B10]=1, V[B11]=-1
    # col 3: M4 -> V[B01]=1, V[B00]=-1
    # col 4: M5 -> V[B11]=1
    # col 5: M6 -> V[B00]=1, V[B10]=1
    # col 6: M7 -> V[B01]=1, V[B11]=1
    
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],   # row 0: j=0, k=0 (B00)
        [0, 0, 0, 1, 0, 0, 1],    # row 1: j=0, k=1 (B01)
        [1, 0, 1, 0, 1, 0, 0],    # row 2: j=1, k=0 (B10)
        [1, 0, -1, 0, 1, 0, 1],   # row 3: j=1, k=1 (B11)
    ]
    
    # Local Strassen W (4 rows: k in {0,1}, i in {0,1} -> k*2+i), 7 cols
    # C00 = M1 + M4 - M5 - M7 -> W[k=0,i=0]: M1=1, M4=1, M5=-1, M7=-1
    # C01 = M3 + M7 -> W[k=1,i=0]: M3=1, M7=1
    # C10 = M2 + M5 -> W[k=0,i=1]: M2=1, M5=1
    # C11 = M1 - M2 + M3 + M6 -> W[k=1,i=1]: M1=1, M2=-1, M3=1, M6=1
    
    strassen_W = [
        [1, 0, 0, 1, -1, 0, -1],  # row 0: k=0, i=0 (C00)
        [0, 1, 1, 0, 1, 0, 0],    # row 1: k=1, i=0 (C01)
        [0, 0, 1, 0, 0, 1, 1],    # row 2: k=0, i=1 (C10)
        [1, 1, 0, -1, 0, -1, 1],  # row 3: k=1, i=1 (C11)
    ]
    
    # Part 1: j in {0,1} (2x2x8), split k into 4 blocks of size 2
    col_offset = 0
    for t in range(4):
        for s in range(7):
            c = col_offset + s
            # U: for i in {0,1}, j in {0,1}, global row = i*3 + j
            for i in range(2):
                for j in range(2):
                    local_row = i * 2 + j
                    global_row = i * 3 + j
                    U[global_row][c] = strassen_U[local_row][s]
            
            # V: for j in {0,1}, k_local in {0,1}, global k = 2*t + k_local
            for j in range(2):
                for k_local in range(2):
                    local_row = j * 2 + k_local
                    global_k = 2 * t + k_local
                    global_row = j * 8 + global_k
                    V[global_row][c] = strassen_V[local_row][s]
            
            # W: for k_local in {0,1}, i in {0,1}, global k = 2*t + k_local
            for k_local in range(2):
                for i in range(2):
                    local_row = k_local * 2 + i
                    global_k = 2 * t + k_local
                    global_row = global_k * 2 + i
                    W[global_row][c] = strassen_W[local_row][s]
        col_offset += 7
    
    # Part 2: j=2 (2x1x8), schoolbook
    for i in range(2):
        for k in range(8):
            c = col_offset
            col_offset += 1
            U[i * 3 + 2][c] = 1
            V[2 * 8 + k][c] = 1
            W[k * 2 + i][c] = 1
    
    return {"U": U, "V": V, "W": W}

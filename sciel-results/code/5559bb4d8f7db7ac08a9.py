import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Initialize U, V, W
    # We don't know the final rank yet, but we can estimate max rank
    # Max rank is n*m*p = 105
    # We will build columns and then slice/trim if needed, or just append.
    # Appending is easier.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Define the 6 Strassen blocks
    # (I_start, J_start, K_start)
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4)
    ]
    
    # Set of indices covered by Strassen blocks
    covered = set()
    for I0, J0, K0 in blocks:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    covered.add((I0+i, J0+j, K0+k))
                    
    # Strassen decomposition for 2x2x2
    # M1: U(A00, A11), V(B00, B11), W(C00, C11)
    # M2: U(A10, A11), V(B00), W(C10, -C11)
    # M3: U(A00), V(B01, -B11), W(C01, C11)
    # M4: U(A11), V(B10, -B00), W(C00, C10)
    # M5: U(A00, A01), V(B11), W(-C00, C01)
    # M6: U(A10, -A00), V(B00, B01), W(C11)
    # M7: U(A01, -A11), V(B10, B11), W(C00)
    
    def add_strassen_block(I0, J0, K0):
        # Local indices relative to block
        # A[i][j] -> U row I0+i, J0+j
        # B[j][k] -> V row J0+j, K0+k
        # C[i][k] -> W row K0+k, I0+i
        
        # M1
        U = { (I0, J0): 1, (I0+1, J0+1): 1 }
        V = { (J0, K0): 1, (J0+1, K0+1): 1 }
        W = { (K0, I0): 1, (K0+1, I0+1): 1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M2
        U = { (I0+1, J0): 1, (I0+1, J0+1): 1 }
        V = { (J0, K0): 1 }
        W = { (K0, I0+1): 1, (K0+1, I0+1): -1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M3
        U = { (I0, J0): 1 }
        V = { (J0, K0+1): 1, (J0+1, K0+1): -1 }
        W = { (K0+1, I0): 1, (K0+1, I0+1): 1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M4
        U = { (I0+1, J0+1): 1 }
        V = { (J0+1, K0): 1, (J0, K0): -1 }
        W = { (K0, I0): 1, (K0, I0+1): 1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M5
        U = { (I0, J0): 1, (I0, J0+1): 1 }
        V = { (J0+1, K0+1): 1 }
        W = { (K0, I0): -1, (K0+1, I0): 1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M6
        U = { (I0+1, J0): 1, (I0, J0): -1 }
        V = { (J0, K0): 1, (J0, K0+1): 1 }
        W = { (K0+1, I0+1): 1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)
        
        # M7
        U = { (I0, J0+1): 1, (I0+1, J0+1): -1 }
        V = { (J0+1, K0): 1, (J0+1, K0+1): 1 }
        W = { (K0, I0): 1 }
        U_cols.append(U); V_cols.append(V); W_cols.append(W)

    for block in blocks:
        if time.time() > deadline:
            break
        add_strassen_block(*block)
        
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                if time.time() > deadline:
                    break
                U = { (i, j): 1 }
                V = { (j, k): 1 }
                W = { (k, i): 1 }
                U_cols.append(U); V_cols.append(V); W_cols.append(W)
                
    # Convert sparse columns to dense matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for (i, j), val in U_cols[c].items():
            U[i * m + j][c] = val
        for (j, k), val in V_cols[c].items():
            V[j * p + k][c] = val
        for (k, i), val in W_cols[c].items():
            W[k * n + i][c] = val
            
    return {"U": U, "V": V, "W": W}

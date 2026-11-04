import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Rank 88
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(a_i, a_j, b_j, b_k, c_i, c_k):
        # a_i, a_j: indices for A block (2x2)
        # b_j, b_k: indices for B block (2x2)
        # c_i, c_k: indices for C block (2x2)
        # This adds 7 terms.
        nonlocal col
        
        # Local indices 0,1 for rows and cols of the block
        # A00, A01, A10, A11
        # B00, B01, B10, B11
        # C00, C01, C10, C11
        
        # U indices: (a_i[0]*m + a_j[0]), etc.
        # V indices: (b_j[0]*p + b_k[0]), etc.
        # W indices: (b_k[0]*n + a_i[0]), etc.
        
        # We will define the 7 terms explicitly.
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11
        U[a_i[0]*m + a_j[0]][col] = 1
        U[a_i[1]*m + a_j[1]][col] = 1
        V[b_j[0]*p + b_k[0]][col] = 1
        V[b_j[1]*p + b_k[1]][col] = 1
        W[b_k[0]*n + a_i[0]][col] = 1
        W[b_k[1]*n + a_i[1]][col] = 1
        col += 1
        
        # Term 2: (A10+A11)B00
        # U: A10, A11
        # V: B00
        # W: C10
        U[a_i[1]*m + a_j[0]][col] = 1
        U[a_i[1]*m + a_j[1]][col] = 1
        V[b_j[0]*p + b_k[0]][col] = 1
        W[b_k[0]*n + a_i[1]][col] = 1
        col += 1
        
        # Term 3: A00(B11-B00)
        # U: A00
        # V: B11, B00
        # W: C01, C11
        U[a_i[0]*m + a_j[0]][col] = 1
        V[b_j[1]*p + b_k[1]][col] = 1
        V[b_j[0]*p + b_k[0]][col] = -1
        W[b_k[1]*n + a_i[0]][col] = 1
        W[b_k[1]*n + a_i[1]][col] = 1
        col += 1
        
        # Term 4: A11(B00-B10)
        # U: A11
        # V: B00, B10
        # W: C00, C10
        U[a_i[1]*m + a_j[1]][col] = 1
        V[b_j[0]*p + b_k[0]][col] = 1
        V[b_j[1]*p + b_k[0]][col] = -1
        W[b_k[0]*n + a_i[0]][col] = 1
        W[b_k[0]*n + a_i[1]][col] = 1
        col += 1
        
        # Term 5: (A00+A01)(B10+B11)
        # U: A00, A01
        # V: B10, B11
        # W: C00, C01
        U[a_i[0]*m + a_j[0]][col] = 1
        U[a_i[0]*m + a_j[1]][col] = 1
        V[b_j[1]*p + b_k[0]][col] = 1
        V[b_j[1]*p + b_k[1]][col] = 1
        W[b_k[0]*n + a_i[0]][col] = 1
        W[b_k[1]*n + a_i[0]][col] = 1
        col += 1
        
        # Term 6: (A10-A00)(B00+B01)
        # U: A10, A00
        # V: B00, B01
        # W: C11
        U[a_i[1]*m + a_j[0]][col] = 1
        U[a_i[0]*m + a_j[0]][col] = -1
        V[b_j[0]*p + b_k[0]][col] = 1
        V[b_j[0]*p + b_k[1]][col] = 1
        W[b_k[1]*n + a_i[1]][col] = 1
        col += 1
        
        # Term 7: (A01-A11)(B10+B11)
        # U: A01, A11
        # V: B10, B11
        # W: C00
        U[a_i[0]*m + a_j[1]][col] = 1
        U[a_i[1]*m + a_j[1]][col] = -1
        V[b_j[1]*p + b_k[0]][col] = 1
        V[b_j[1]*p + b_k[1]][col] = 1
        W[b_k[0]*n + a_i[0]][col] = 1
        col += 1

    def add_schoolbook_124(a_i, a_j, b_j, b_k, c_i, c_k):
        # 1x2x4 schoolbook
        nonlocal col
        for j in a_j:
            for k in b_k:
                U[a_i[0]*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + a_i[0]][col] = 1
                col += 1

    # Block 1: A11 (2x2) * Bt1 (2x4)
    # Split Bt1 into two 2x2 blocks: Bt1a (k=0,1), Bt1b (k=2,3)
    add_strassen_222([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    add_strassen_222([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    
    # Block 2: A11 (2x2) * Bt2 (2x4)
    # Bt2: j=0,1; k=4,5,6,7
    add_strassen_222([0,1], [0,1], [0,1], [4,5], [0,1], [4,5])
    add_strassen_222([0,1], [0,1], [0,1], [6,7], [0,1], [6,7])
    
    # Block 3: A12 (2x2) * Bb1 (2x4)
    # A12: i=0,1; j=2,3
    # Bb1: j=2,3; k=0,1,2,3
    add_strassen_222([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    add_strassen_222([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    
    # Block 4: A12 (2x2) * Bb2 (2x4)
    # Bb2: j=2,3; k=4,5,6,7
    add_strassen_222([0,1], [2,3], [2,3], [4,5], [0,1], [4,5])
    add_strassen_222([0,1], [2,3], [2,3], [6,7], [0,1], [6,7])
    
    # Block 5: A21 (1x2) * Bt1 (2x4)
    # A21: i=2; j=0,1
    # Bt1: j=0,1; k=0,1,2,3
    add_schoolbook_124([2], [0,1], [0,1], [0,1,2,3], [2], [0,1,2,3])
    
    # Block 6: A21 (1x2) * Bt2 (2x4)
    add_schoolbook_124([2], [0,1], [0,1], [4,5,6,7], [2], [4,5,6,7])
    
    # Block 7: A22 (1x2) * Bb1 (2x4)
    # A22: i=2; j=2,3
    add_schoolbook_124([2], [2,3], [2,3], [0,1,2,3], [2], [0,1,2,3])
    
    # Block 8: A22 (1x2) * Bb2 (2x4)
    add_schoolbook_124([2], [2,3], [2,3], [4,5,6,7], [2], [4,5,6,7])
    
    return {"U": U, "V": V, "W": W}

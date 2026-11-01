import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication
    # Products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # Reconstruction:
    # C11 = M1 - M4 + M6 - M2
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M7
    
    # Each term: (a_vec, b_vec, c_vec)
    # a_vec: [A11, A12, A21, A22]
    # b_vec: [B11, B12, B21, B22]
    # c_vec: [C11, C12, C21, C22] -- contribution of this term to each C element
    
    strassen_terms = [
        # M1: (A11+A22)(B11+B22)
        # C11 += 1, C22 += 1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        
        # M2: (A21+A22)B11
        # C11 += -1, C21 += 1
        ([0, 0, 1, 1], [1, 0, 0, 0], [-1, 0, 1, 0]),
        
        # M3: A11(B12-B22)
        # C12 += 1, C22 += -1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]),
        
        # M4: A22(B21-B11)
        # C11 += -1, C21 += 1
        ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]),
        
        # M5: (A11+A12)B22
        # C12 += 1, C22 += 1
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
        
        # M6: (A21-A11)(B11+B12)
        # C11 += 1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        
        # M7: (A12-A22)(B21+B22)
        # C22 += -1
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, -1]),
    ]
    
    col = 0
    for k in range(5):
        for a_vec, b_vec, c_vec in strassen_terms:
            # U: A is 2x2, indices [i*m+j] = [i*2+j]
            for idx in range(4):
                U[idx][col] = a_vec[idx]
            
            # V: B is 2x10, for block k
            # B[0, 2k] -> V[2k]
            # B[0, 2k+1] -> V[2k+1]
            # B[1, 2k] -> V[10+2k]
            # B[1, 2k+1] -> V[10+2k+1]
            V[2*k][col] = b_vec[0]
            V[2*k+1][col] = b_vec[1]
            V[10+2*k][col] = b_vec[2]
            V[10+2*k+1][col] = b_vec[3]
            
            # W: C is 2x10, W[j*n + i] where j is col, i is row
            # C[0, 2k]   -> W[4k]
            # C[1, 2k]   -> W[4k+1]
            # C[0, 2k+1] -> W[4k+2]
            # C[1, 2k+1] -> W[4k+3]
            W[4*k][col] = c_vec[0]
            W[4*k+1][col] = c_vec[2]
            W[4*k+2][col] = c_vec[1]
            W[4*k+3][col] = c_vec[3]
            
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}

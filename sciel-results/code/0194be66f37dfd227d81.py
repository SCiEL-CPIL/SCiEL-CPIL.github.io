import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # Part 1: 4x4 x 4x4 -> 4x4 (Rank 49)
    # Strassen 2x2x2 matrices
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: 0,1,2,3 for 2x2 matrices
    # 0: 00, 1: 01, 2: 10, 3: 11
    
    U2 = np.array([
        [1, 0, 1, 1, 1, 1, 0], # 00
        [0, 1, 0, 0, 1, 0, 1], # 01
        [0, 0, 1, 1, 0, 1, -1], # 10
        [1, 0, 0, 1, 0, 1, -1]  # 11
    ], dtype=np.int64)
    
    V2 = np.array([
        [1, 1, 0, 0, 1, 1, 1], # 00
        [1, 0, 1, 0, 0, 1, 1], # 01
        [0, 1, -1, 1, 1, 0, 1], # 10
        [1, 1, -1, -1, 1, 1, 1]  # 11
    ], dtype=np.int64)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # 00
        [0, 0, 1, 0, 1, 0, 0], # 01
        [0, 1, 0, 0, 0, 0, 0], # 10
        [1, -1, 1, 1, 0, 1, 0]  # 11
    ], dtype=np.int64)
    
    # Verify 2x2x2
    # C = A B
    # M1 = (A00+A11)(B00+B11) -> W00+1, W11+1
    # M2 = (A10+A11)B00 -> W10+1, W11-1
    # M3 = A00(B01-B11) -> W01+1, W11+1
    # M4 = A11(B10-B00) -> W00+1, W10+1
    # M5 = (A00+A01)B11 -> W00-1, W01+1
    # M6 = (A10-A00)(B00+B01) -> W11+1
    # M7 = (A01-A11)(B10+B11) -> W00+1
    
    # My U2, V2, W2 above might be slightly off in signs/indices.
    # Let's use the standard ones from literature or verify carefully.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Let's rebuild U2, V2, W2 correctly.
    # U2 rows: A11, A12, A21, A22
    # V2 rows: B11, B12, B21, B22
    # W2 rows: C11, C12, C21, C22
    
    U2 = np.zeros((4, 7), dtype=np.int64)
    V2 = np.zeros((4, 7), dtype=np.int64)
    W2 = np.zeros((4, 7), dtype=np.int64)
    
    # M1
    U2[0, 0] += 1; U2[3, 0] += 1
    V2[0, 0] += 1; V2[3, 0] += 1
    W2[0, 0] += 1; W2[3, 0] += 1
    
    # M2
    U2[2, 1] += 1; U2[3, 1] += 1
    V2[0, 1] += 1
    W2[2, 1] += 1; W2[3, 1] -= 1
    
    # M3
    U2[0, 2] += 1
    V2[1, 2] += 1; V2[3, 2] -= 1
    W2[1, 2] += 1; W2[3, 2] += 1
    
    # M4
    U2[3, 3] += 1
    V2[2, 3] += 1; V2[0, 3] -= 1
    W2[0, 3] += 1; W2[2, 3] += 1
    
    # M5
    U2[0, 4] += 1; U2[1, 4] += 1
    V2[3, 4] += 1
    W2[0, 4] -= 1; W2[1, 4] += 1
    
    # M6
    U2[2, 5] += 1; U2[0, 5] -= 1
    V2[0, 5] += 1; V2[1, 5] += 1
    W2[3, 5] += 1
    
    # M7
    U2[1, 6] += 1; U2[3, 6] -= 1
    V2[2, 6] += 1; V2[3, 6] += 1
    W2[0, 6] += 1
    
    # Kronecker product for 4x4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Part 2: 4x4 x 4x2 -> 4x2 (Rank 32)
    # Trivial decomposition
    # C_i,j = sum_k A_i,k B_k,j
    # U42: 16x32
    # V42: 8x32 (Rows 0-7 correspond to B rows 0-3, cols 0-1 of the sub-block)
    # W42: 8x32 (Rows 0-7 correspond to C rows 0-3, cols 0-1 of the sub-block)
    
    U42 = np.zeros((16, 32), dtype=np.int64)
    V42 = np.zeros((8, 32), dtype=np.int64)
    W42 = np.zeros((8, 32), dtype=np.int64)
    
    col = 0
    for i in range(4):
        for k in range(4):
            for j in range(2):
                # U index: i*4 + k
                # V index: k*2 + j
                # W index: j*4 + i
                U42[i*4 + k, col] = 1
                V42[k*2 + j, col] = 1
                W42[j*4 + i, col] = 1
                col += 1
                
    # Assemble
    # U: 16 x (49+32)
    # V: 24 x (49+32)
    # W: 24 x (49+32)
    
    rank = 49 + 32
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((24, rank), dtype=np.int64)
    W = np.zeros((24, rank), dtype=np.int64)
    
    # First 49 cols
    U[:, :49] = U4
    # V4 is 16x49. Global V is 24x49.
    # B cols 0-3 correspond to V4 rows 0-15.
    # B cols 4-5 correspond to V42 rows 0-7.
    V[:16, :49] = V4
    # W4 is 16x49. Global W is 24x49.
    # C cols 0-3 correspond to W4 rows 0-15.
    # W index: (col)*4 + row.
    # W4 rows 0-3 are C col 0. Rows 4-7 are C col 1.
    # So W4 is already in the correct order for C cols 0-3.
    W[:16, :49] = W4
    
    # Next 32 cols
    U[:, 49:] = U42
    # V42 is 8x32. Global V rows 16-23 correspond to B cols 4-5.
    # V42 row k*2+j corresponds to B row k, col j (of the 2-col block).
    # So V42 row 0 is B row 0, col 4.
    # Global V row 16 is B row 0, col 4.
    V[16:, 49:] = V42
    # W42 is 8x32. Global W rows 16-23 correspond to C cols 4-5.
    # W42 row j*4+i corresponds to C col j, row i.
    # Global W row 16+j*4+i.
    W[16:, 49:] = W42
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

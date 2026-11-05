import numpy as np

DIMENSIONS = [3, 4, 7]

def strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    # A is 2x2, indexed as A[i,j] with i,j in {0,1}
    # B is 2x2, indexed as B[j,k] with j,k in {0,1}
    # C is 2x2, indexed as C[i,k] with i,k in {0,1}
    # U shape: (4, 7) - A entries
    # V shape: (4, 7) - B entries
    # W shape: (4, 7) - C entries, using trace convention W[k*n+i, r] where n=2
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A part: A11 + A22 -> [1, 0, 0, 1] (indices: 00, 01, 10, 11)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]  # B11 + B22
    # C11 += M1, C22 += M1
    W[0, 0] = 1  # C11: k=0, i=0 -> index 0
    W[3, 0] = 1  # C22: k=1, i=1 -> index 1*2+1=3
    
    # M2 = (A21+A22)B11
    U[:, 1] = [0, 0, 1, 1]  # A21 + A22
    V[:, 1] = [1, 0, 0, 0]  # B11
    # C21 += M2
    W[2, 1] = 1  # C21: k=0, i=1 -> index 0*2+1=1... wait
    
    # Let me re-check the indexing. C[i,k] where i is row of A, k is col of B.
    # Trace convention: W[k*n + i, r] where n=2 (rows of A = cols of C)
    # So C[0,0] -> W[0*2+0] = W[0]
    # C[0,1] -> W[1*2+0] = W[2]
    # C[1,0] -> W[0*2+1] = W[1]
    # C[1,1] -> W[1*2+1] = W[3]
    
    # M2 contributes to C21 = C[1,0] -> W[1]
    W[1, 1] = 1
    
    # M3 = A11(B12-B22)
    U[:, 2] = [1, 0, 0, 0]  # A11
    V[:, 2] = [0, 1, 0, -1]  # B12 - B22
    # C12 += M3 -> C[0,1] -> W[2]
    W[2, 2] = 1
    
    # M4 = A22(B21-B11)
    U[:, 3] = [0, 0, 0, 1]  # A22
    V[:, 3] = [0, 0, 1, -1]  # B21 - B11
    # C11 += M4 -> W[0], C21 += M4 -> W[1]
    W[0, 3] = 1
    W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    U[:, 4] = [1, 1, 0, 0]  # A11 + A12
    V[:, 4] = [0, 0, 0, 1]  # B22
    # C11 -= M5 -> W[0] = -1, C12 += M5 -> W[2] = 1
    W[0, 4] = -1
    W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[:, 5] = [0, 0, 1, -1]  # A21 - A11
    V[:, 5] = [1, 1, 0, 0]  # B11 + B12
    # C22 += M6 -> W[3]
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[:, 6] = [0, 1, 0, -1]  # A12 - A22
    V[:, 6] = [0, 0, 1, 1]  # B21 + B22
    # C11 += M7 -> W[0] = 1
    W[0, 6] = 1
    
    return U, V, W


def trivial_1xn_x_n_p(n, p):
    """1 x n matrix times n x p matrix = 1 x p matrix. Rank = p."""
    # A is 1xn, B is nxp, C is 1xp
    # U shape: (n, p), V shape: (n*p, p), W shape: (p*1, p) = (p, p)
    U = np.zeros((n, p))
    V = np.zeros((n * p, p))
    W = np.zeros((p, p))
    
    for k in range(p):
        # C_k = sum_j A_j * B[j,k]
        U[:, k] = np.ones(n)  # A_j for all j
        # B[j,k] for j in range(n)
        for j in range(n):
            V[j * p + k, k] = 1.0
        W[k, k] = 1.0
    
    return U, V, W


def trivial_nxn_x_nx1(n):
    """n x n matrix times n x 1 matrix = n x 1 matrix. Rank = n."""
    # Actually this is not needed. Let me handle 1xn x nxp generally.
    pass


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Split A (3x4) into A1 (2x4) and A2 (1x4)
    # Split B (4x7) into B1 (4x4) and B2 (4x3)
    # C = A*B, where:
    # C11 (2x4) = A1 * B1
    # C12 (2x3) = A1 * B2
    # C21 (1x4) = A2 * B1
    # C22 (1x3) = A2 * B2
    
    # Decompositions:
    # A1*B1: 2x4x4, rank 14 (two 2x2x2 Strassen)
    # A1*B2: 2x3x4, rank 14 (two 2x2x2 Strassen)
    # A2*B1: 1x4x4, rank 4
    # A2*B2: 1x3x4, rank 3
    
    total_rank = 14 + 14 + 4 + 3
    U = np.zeros((n * m, total_rank))  # (12, 35)
    V = np.zeros((m * p, total_rank))  # (28, 35)
    W = np.zeros((p * n, total_rank))  # (21, 35)
    
    col_offset = 0
    
    # --- A1*B1: 2x4x4 ---
    # A1 is 2x4, B1 is 4x4, C11 is 2x4
    # Split B1 into B1_left (4x2) and B1_right (4x2)
    # C11_left = A1 * B1_left (2x2, Strassen rank 7)
    # C11_right = A1 * B1_right (2x2, Strassen rank 7)
    
    U1, V1, W1 = strassen_2x2x2()
    U2, V2, W2 = strassen_2x2x2()
    
    # Map A1 entries: A1[i,j] for i in {0,1}, j in {0,1,2,3}
    # A1_left uses j in {0,1}, A1_right uses j in {2,3}
    # U1 corresponds to A1_left (columns 0,1 of A1)
    # U2 corresponds to A1_right (columns 2,3 of A1)
    
    # U1 shape (4,7): rows are A1_left entries [0,0],[0,1],[1,0],[1,1]
    # Map to U full: A1[i,j] is at row i*4 + j
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i * 4 + j, col_offset + r] = U1[i * 2 + j, r]
    
    # V1 shape (4,7): rows are B1_left entries [j,k] for j in {0,1,2,3}, k in {0,1}
    # B1 is 4x4, B1_left is columns 0,1 of B1
    # V1[j,k] where j in {0,1,2,3}, k in {0,1}
    for r in range(7):
        for j in range(4):
            for k in range(2):
                V[j * 4 + k, col_offset + r] = V1[j * 2 + k, r]
    
    # W1 shape (4,7): C11_left[i,k] for i in {0,1}, k in {0,1}
    # C11 is 2x4, C11_left is columns 0,1
    # W1[k*2+i, r] for C11_left[i,k]
    for r in range(7):
        for i in range(2):
            for k in range(2):
                W[k * 3 + i, col_offset + r] = W1[k * 2 + i, r]
    
    col_offset += 7
    
    # A1 * B1_right: B1_right is columns 2,3 of B1
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j + 2), col_offset + r] = U2[i * 2 + j, r]
    
    for r in range(7):
        for j in range(4):
            for k in range(2):
                V[j * 4 + (k + 2), col_offset + r] = V2[j * 2 + k, r]
    
    for r in range(7):
        for i in range(2):
            for k in range(2):
                W[k * 3 + i, col_offset + r] = W2[k * 2 + i, r]
    
    col_offset += 7
    
    # --- A1*B2: 2x3x4 ---
    # A1 is 2x4, B2 is 4x3, C12 is 2x3
    # Split B2 into B2_left (4x2) and B2_right (4x1)
    # C12_left = A1 * B2_left (2x2, Strassen rank 7)
    # C12_right = A1 * B2_right (2x1, rank 2)
    
    U3, V3, W3 = strassen_2x2x2()
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i * 4 + j, col_offset + r] = U3[i * 2 + j, r]
    
    # B2 is 4x3, B2_left is columns 0,1 of B2
    for r in range(7):
        for j in range(4):
            for k in range(2):
                V[j * 7 + (4 + k), col_offset + r] = V3[j * 2 + k, r]
    
    # C12 is 2x3, C12_left is columns 0,1
    for r in range(7):
        for i in range(2):
            for k in range(2):
                W[(4 + k) * 3 + i, col_offset + r] = W3[k * 2 + i, r]
    
    col_offset += 7
    
    # C12_right = A1 * B2_right (2x1)
    # A1 is 2x4, B2_right is 4x1 (column 2 of B2)
    # C12_right is 2x1
    # Rank 2: for each output row i, C_i = sum_j A1[i,j] * B2[j,2]
    for i in range(2):
        for j in range(4):
            U[i * 4 + j, col_offset + i] = 1.0
            V[j * 7 + (4 + 2), col_offset + i] = 1.0
        W[(4 + 2) * 3 + i, col_offset + i] = 1.0
    
    col_offset += 2
    
    # --- A2*B1: 1x4x4 ---
    # A2 is 1x4, B1 is 4x4, C21 is 1x4
    # Rank 4
    for k in range(4):
        for j in range(4):
            U[2 * 4 + j, col_offset + k] = 1.0  # A2[j]
            V[j * 7 + k, col_offset + k] = 1.0  # B1[j,k]
        W[k * 3 + 2, col_offset + k] = 1.0  # C21[k]
    
    col_offset += 4
    
    # --- A2*B2: 1x4x3 ---
    # A2 is 1x4, B2 is 4x3, C22 is 1x3
    # Rank 3
    for k in range(3):
        for j in range(4):
            U[2 * 4 + j, col_offset + k] = 1.0  # A2[j]
            V[j * 7 + (4 + k), col_offset + k] = 1.0  # B2[j,k]
        W[(4 + k) * 3 + 2, col_offset + k] = 1.0  # C22[k]
    
    col_offset += 3
    
    assert col_offset == total_rank, f"col_offset {col_offset} != total_rank {total_rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

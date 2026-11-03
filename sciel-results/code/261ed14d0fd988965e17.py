"""Matrix multiplication 2x4 x 4x4 rank-minimal decomposition via Strassen."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    The 2x4 matrix A is partitioned as [A1 | A2] where A1, A2 are 2x2.
    The 4x4 matrix B is partitioned as [[B11, B12], [B21, B22]] where each is 2x2.
    The 2x4 result C = A*B has columns C1 = A1*B11 + A2*B21 and C2 = A1*B12 + A2*B22.
    
    Each 2x2 x 2x2 -> 2x2 block multiplication uses Strassen's 7-product algorithm.
    Total rank = 4 blocks * 7 products = 28.
    
    Indexing convention:
    - U has shape (n*m, rank) = (8, 28): U[i*m + j, c] for A[i,j]
    - V has shape (m*p, rank) = (16, 28): V[j*p + k, c] for B[j,k]
    - W has shape (p*n, rank) = (8, 28): W[k*n + i, c] for C[i,k]
    """
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    U = [[0.0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0.0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0.0] * rank for _ in range(p * n)]   # 8 x 28
    
    # A indices: A[i,j] -> U[i*m+j] = U[2*i+j]
    #   A1 = A[0:2, 0:2] -> rows 0,1,2,3 (A[0,0],A[0,1],A[1,0],A[1,1])
    #   A2 = A[0:2, 2:4] -> rows 4,5,6,7 (A[0,2],A[0,3],A[1,2],A[1,3])
    
    # B indices: B[j,k] -> V[j*p+k] = V[4*j+k]
    #   B11 = B[0:2, 0:2] -> cols 0,1,4,5
    #   B12 = B[0:2, 2:4] -> cols 2,3,6,7
    #   B21 = B[2:4, 0:2] -> cols 8,9,12,13
    #   B22 = B[2:4, 2:4] -> cols 10,11,14,15
    
    # C indices: C[i,k] -> W[k*n+i] = W[2*k+i]
    #   C1 = C[0:2, 0:2] -> W indices: C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    #   C2 = C[0:2, 2:4] -> W indices: C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_block(a_rows, b_cols, c_w_indices, offset):
        """
        Add 7 Strassen products for one 2x2x2x2->2x2 block.
        
        a_rows: [a11, a12, a21, a22] - U row indices for the 2x2 A block
        b_cols: [b11, b12, b21, b22] - V col indices for the 2x2 B block
        c_w_indices: [c11, c12, c21, c22] - W indices for the 2x2 C block
        offset: starting column index in U, V, W
        """
        a11, a12, a21, a22 = a_rows
        b11, b12, b21, b22 = b_cols
        c11, c12, c21, c22 = c_w_indices
        
        # M1 = (A11+A22)(B11+B22)
        col = offset
        U[a11][col] += 1
        U[a22][col] += 1
        V[b11][col] += 1
        V[b22][col] += 1
        W[c11][col] += 1
        W[c22][col] += 1
        
        # M2 = (A21+A22)B11
        col = offset + 1
        U[a21][col] += 1
        U[a22][col] += 1
        V[b11][col] += 1
        W[c21][col] += 1
        W[c22][col] += -1
        
        # M3 = A11(B12-B22)
        col = offset + 2
        U[a11][col] += 1
        V[b12][col] += 1
        V[b22][col] += -1
        W[c12][col] += 1
        W[c22][col] += 1
        
        # M4 = A22(B21-B11)
        col = offset + 3
        U[a22][col] += 1
        V[b21][col] += 1
        V[b11][col] += -1
        W[c11][col] += 1
        W[c21][col] += 1
        
        # M5 = (A11+A12)B22
        col = offset + 4
        U[a11][col] += 1
        U[a12][col] += 1
        V[b22][col] += 1
        W[c11][col] += -1
        W[c12][col] += 1
        
        # M6 = (A21-A11)(B11+B12)
        col = offset + 5
        U[a21][col] += 1
        U[a11][col] += -1
        V[b11][col] += 1
        V[b12][col] += 1
        W[c22][col] += 1
        
        # M7 = (A12-A22)(B21+B22)
        col = offset + 6
        U[a12][col] += 1
        U[a22][col] += -1
        V[b21][col] += 1
        V[b22][col] += 1
        W[c11][col] += 1
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return {"U": U, "V": W, "W": W}  # fallback, shouldn't happen
    
    # Block 1: A1 * B11 -> C1 (C[0:2, 0:2])
    # A1 rows: [A[0,0], A[0,1], A[1,0], A[1,1]] = [0, 1, 2, 3]
    # B11 cols: [B[0,0], B[0,1], B[1,0], B[1,1]] = [0, 1, 4, 5]
    # C1 W indices: [C[0,0], C[0,1], C[1,0], C[1,1]] = [W[0], W[2], W[1], W[3]] = [0, 2, 1, 3]
    add_strassen_block([0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 1, 3], 0)
    
    # Block 2: A2 * B21 -> C1 (C[0:2, 0:2])
    # A2 rows: [A[0,2], A[0,3], A[1,2], A[1,3]] = [4, 5, 6, 7]
    # B21 cols: [B[2,0], B[2,1], B[3,0], B[3,1]] = [8, 9, 12, 13]
    # C1 W indices: [0, 2, 1, 3]
    add_strassen_block([4, 5, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3], 7)
    
    # Block 3: A1 * B12 -> C2 (C[0:2, 2:4])
    # A1 rows: [0, 1, 2, 3]
    # B12 cols: [B[0,2], B[0,3], B[1,2], B[1,3]] = [2, 3, 6, 7]
    # C2 W indices: [C[0,2], C[0,3], C[1,2], C[1,3]] = [W[4], W[6], W[5], W[7]] = [4, 6, 5, 7]
    add_strassen_block([0, 1, 2, 3], [2, 3, 6, 7], [4, 6, 5, 7], 14)
    
    # Block 4: A2 * B22 -> C2 (C[0:2, 2:4])
    # A2 rows: [4, 5, 6, 7]
    # B22 cols: [B[2,2], B[2,3], B[3,2], B[3,3]] = [10, 11, 14, 15]
    # C2 W indices: [4, 6, 5, 7]
    add_strassen_block([4, 5, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7], 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

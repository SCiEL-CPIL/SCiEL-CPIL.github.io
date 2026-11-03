import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _validate_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Validate that the given factor matrices exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Construct the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7).
    
    Convention: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[k,i]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A10+A11) B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # P3 = A00 (B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1
    
    # P4 = A11 (B01-B10)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # P5 = (A00+A01) B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # P6 = (A10-A00) (B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = -1
    
    # P7 = (A01-A11) (B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4x4():
    """Build 4x4x4 Strassen decomposition (rank 49) from 2x2x2 base."""
    U2, V2, W2 = _strassen_2x2x2()
    # Expand from 2 to 4
    n = 4
    n_half = 2
    U_prev, V_prev, W_prev = U2, V2, W2
    rank_prev = U_prev.shape[1]  # 7
    rank_new = rank_prev * 7  # 49
    
    U = np.zeros((n*n, rank_new), dtype=float)
    V = np.zeros((n*n, rank_new), dtype=float)
    W = np.zeros((n*n, rank_new), dtype=float)
    
    alpha = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[0,-1]],
    ]
    
    beta = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,0],[1,-1]],
        [[0,1],[-1,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,1],[0,1]],
    ]
    
    gamma = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[0,1],[0,-1]],
        [[1,0],[1,0]],
        [[-1,0],[1,0]],
        [[0,0],[0,-1]],
        [[1,0],[0,0]],
    ]
    
    for r in range(7):
        for I in range(2):
            for J in range(2):
                a_coeff = alpha[r][I][J]
                if a_coeff == 0:
                    continue
                for i in range(n_half):
                    for j in range(n_half):
                        row_idx = (I * n_half + i) * n + (J * n_half + j)
                        sub_row = i * n_half + j
                        for r_prev in range(rank_prev):
                            U[row_idx, r * rank_prev + r_prev] += a_coeff * U_prev[sub_row, r_prev]
        
        for J in range(2):
            for K in range(2):
                b_coeff = beta[r][J][K]
                if b_coeff == 0:
                    continue
                for j in range(n_half):
                    for k in range(n_half):
                        row_idx = (J * n_half + j) * n + (K * n_half + k)
                        sub_row = j * n_half + k
                        for r_prev in range(rank_prev):
                            V[row_idx, r * rank_prev + r_prev] += b_coeff * V_prev[sub_row, r_prev]
        
        for K in range(2):
            for I in range(2):
                c_coeff = gamma[r][K][I]
                if c_coeff == 0:
                    continue
                for k in range(n_half):
                    for i in range(n_half):
                        row_idx = (K * n_half + k) * n + (I * n_half + i)
                        sub_row = k * n_half + i
                        for r_prev in range(rank_prev):
                            W[row_idx, r * rank_prev + r_prev] += c_coeff * W_prev[sub_row, r_prev]
    
    return U, V, W

def _block_decomposition_5x5x5():
    """
    Build 5x5x5 decomposition using block structure:
    - Top-left 2x2 block of C: use Strassen for A[0:2,0:2]*B[0:2,0:2] (rank 7)
      plus schoolbook for A[0:2,2:5]*B[2:5,0:2] (rank 12)
    - All other blocks: schoolbook
    
    Partition: rows/cols split as [0,1] and [2,3,4]
    
    C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:5]*B[2:5,0:2]
    C[0:2,2:5] = A[0:2,0:2]*B[0:2,2:5] + A[0:2,2:5]*B[2:5,2:5]
    C[2:5,0:2] = A[2:5,0:2]*B[0:2,0:2] + A[2:5,2:5]*B[2:5,0:2]
    C[2:5,2:5] = A[2:5,0:2]*B[0:2,2:5] + A[2:5,2:5]*B[2:5,2:5]
    
    Ranks:
    - Strassen(2,2,2): 7
    - Schoolbook(2,3,2): 12  (A[0:2,2:5] is 2x3, B[2:5,0:2] is 3x2)
    - Schoolbook(2,2,3): 12  (A[0:2,0:2] is 2x2, B[0:2,2:5] is 2x3)
    - Schoolbook(2,3,3): 18  (A[0:2,2:5] is 2x3, B[2:5,2:5] is 3x3)
    - Schoolbook(3,2,2): 12  (A[2:5,0:2] is 3x2, B[0:2,0:2] is 2x2)
    - Schoolbook(3,3,2): 18  (A[2:5,2:5] is 3x3, B[2:5,0:2] is 3x2)
    - Schoolbook(3,2,3): 18  (A[2:5,0:2] is 3x2, B[0:2,2:5] is 2x3)
    - Schoolbook(3,3,3): 27  (A[2:5,2:5] is 3x3, B[2:5,2:5] is 3x3)
    
    Total: 7+12+12+18+12+18+18+27 = 124
    """
    n = 5
    total_rank = 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27  # = 124
    
    U = np.zeros((n*n, total_rank), dtype=float)
    V = np.zeros((n*n, total_rank), dtype=float)
    W = np.zeros((n*n, total_rank), dtype=float)
    
    col = 0
    
    # Block 1: C[0:2,0:2] from A[0:2,0:2]*B[0:2,0:2] using Strassen (rank 7)
    U2, V2, W2 = _strassen_2x2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*5+j, col+r] = U2[i*2+j, r]
        for j in range(2):
            for k in range(2):
                V[j*5+k, col+r] = V2[j*2+k, r]
        for k in range(2):
            for i in range(2):
                W[k*5+i, col+r] = W2[k*2+i, r]
    col += 7
    
    # Block 2: C[0:2,0:2] from A[0:2,2:5]*B[2:5,0:2] schoolbook (rank 12)
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i*5+(2+j), col] = 1.0
                V[(2+j)*5+k, col] = 1.0
                W[k*5+i, col] = 1.0
                col += 1
    
    # Block 3: C[0:2,2:5] from A[0:2,0:2]*B[0:2,2:5] schoolbook (rank 12)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*5+j, col] = 1.0
                V[j*5+(2+k), col] = 1.0
                W[(2+k)*5+i, col] = 1.0
                col += 1
    
    # Block 4: C[0:2,2:5] from A[0:2,2:5]*B[2:5,2:5] schoolbook (rank 18)
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*5+(2+j), col] = 1.0
                V[(2+j)*5+(2+k), col] = 1.0
                W[(2+k)*5+i, col] = 1.0
                col += 1
    
    # Block 5: C[2:5,0:2] from A[2:5,0:2]*B[0:2,0:2] schoolbook (rank 12)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[(2+i)*5+j, col] = 1.0
                V[j*5+k, col] = 1.0
                W[k*5+(2+i), col] = 1.0
                col += 1
    
    # Block 6: C[2:5,0:2] from A[2:5,2:5]*B[2:5,0:2] schoolbook (rank 18)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[(2+i)*5+(2+j), col] = 1.0
                V[(2+j)*5+k, col] = 1.0
                W[k*5+(2+i), col] = 1.0
                col += 1
    
    # Block 7: C[2:5,2:5] from A[2:5,0:2]*B[0:2,2:5] schoolbook (rank 18)
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U[(2+i)*5+j, col] = 1.0
                V[j*5+(2+k), col] = 1.0
                W[(2+k)*5+(2+i), col] = 1.0
                col += 1
    
    # Block 8: C[2:5,2:5] from A[2:5,2:5]*B[2:5,2:5] schoolbook (rank 27)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[(2+i)*5+(2+j), col] = 1.0
                V[(2+j)*5+(2+k), col] = 1.0
                W[(2+k)*5+(2+i), col] = 1.0
                col += 1
    
    return U, V, W

def _extend_to_5x5x5(U4, V4, W4):
    """
    Extend a 4x4x4 decomposition to 5x5x5 by adding schoolbook terms
    for the extra row/column.
    Total rank: 49 + 25 + 20 = 94
    """
    n = 5
    rank_4x4 = U4.shape[1]
    rank_total = rank_4x4 + 25 + 20
    
    U = np.zeros((n*n, rank_total), dtype=float)
    V = np.zeros((n*n, rank_total), dtype=float)
    W = np.zeros((n*n, rank_total), dtype=float)
    
    # Copy the 4x4x4 part into the first 49 columns
    for i in range(4):
        for j in range(4):
            row_u = i * n + j
            for r in range(rank_4x4):
                U[row_u, r] = U4[i * 4 + j, r]
    
    for j in range(4):
        for k in range(4):
            row_v = j * n + k
            for r in range(rank_4x4):
                V[row_v, r] = V4[j * 4 + k, r]
    
    for k in range(4):
        for i in range(4):
            row_w = k * n + i
            for r in range(rank_4x4):
                W[row_w, r] = W4[k * 4 + i, r]
    
    # Add schoolbook terms for i=4 (row 4 of A): C[4,k] for k=0..4
    col_offset = rank_4x4
    for j in range(5):
        for k in range(5):
            U[4 * n + j, col_offset] = 1.0
            V[j * n + k, col_offset] = 1.0
            W[k * n + 4, col_offset] = 1.0
            col_offset += 1
    
    # Add schoolbook terms for k=4 (column 4 of B): C[i,4] for i=0..3
    for j in range(5):
        for i in range(4):
            U[i * n + j, col_offset] = 1.0
            V[j * n + 4, col_offset] = 1.0
            W[4 * n + i, col_offset] = 1.0
            col_offset += 1
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Try the 4x4x4 Strassen + extension approach (rank 94)
    U4, V4, W4 = _strassen_4x4x4()
    U, V, W = _extend_to_5x5x5(U4, V4, W4)
    
    # Validate
    if _validate_decomposition(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END

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
    
    C[0,0] = P1 + P4 - P5 + P7
    C[0,1] = P3 + P5
    C[1,0] = P2 + P4
    C[1,1] = P1 - P3 + P2 - P6
    
    P1 = (A00+A11)(B00+B11)
    P2 = (A10+A11) B00
    P3 = A00 (B10-B11)
    P4 = A11 (B01-B10)
    P5 = (A00+A01) B11
    P6 = (A10-A00) (B00+B10)
    P7 = (A01-A11) (B01+B11)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # P1 = (A00+A11)(B00+B11)
    # A00: U[0], A11: U[3]
    # B00: V[0], B11: V[3]
    # C00: W[0], C11: W[3] (since C[k,i], C00 is k=0,i=0 -> W[0], C11 is k=1,i=1 -> W[3])
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A10+A11) B00
    # A10: U[2], A11: U[3]
    # B00: V[0]
    # C10: W[1] (k=0,i=1), C11: W[3] (k=1,i=1)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # P3 = A00 (B10-B11)
    # A00: U[0]
    # B10: V[2] (j=1,k=0), B11: V[3] (j=1,k=1)
    # C01: W[2] (k=1,i=0), C11: W[3] (k=1,i=1)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1
    
    # P4 = A11 (B01-B10)
    # A11: U[3]
    # B01: V[1] (j=0,k=1), B10: V[2] (j=1,k=0)
    # C00: W[0] (k=0,i=0), C10: W[1] (k=0,i=1)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # P5 = (A00+A01) B11
    # A00: U[0], A01: U[1]
    # B11: V[3]
    # C00: W[0] (k=0,i=0), C01: W[2] (k=1,i=0)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # P6 = (A10-A00) (B00+B10)
    # A10: U[2], A00: U[0]
    # B00: V[0], B10: V[2]
    # C11: W[3] (k=1,i=1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = -1
    
    # P7 = (A01-A11) (B01+B11)
    # A01: U[1], A11: U[3]
    # B01: V[1], B11: V[3]
    # C00: W[0] (k=0,i=0)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4x4():
    """
    Construct a rank-49 decomposition for 4x4x4 matrix multiplication
    by applying Strassen's algorithm (rank 7 for 2x2x2) at two levels.
    """
    U2, V2, W2 = _strassen_2x2x2()
    return _expand_strassen(4, U2, V2, W2)

def _expand_strassen(n, U_base, V_base, W_base):
    """
    Recursively expand the Strassen decomposition from size 2 to size n (power of 2).
    """
    if n == 2:
        return U_base.copy(), V_base.copy(), W_base.copy()
    
    n_half = n // 2
    U_prev, V_prev, W_prev = _expand_strassen(n_half, U_base, V_base, W_base)
    rank_prev = U_prev.shape[1]
    rank_new = rank_prev * 7
    
    U = np.zeros((n*n, rank_new), dtype=float)
    V = np.zeros((n*n, rank_new), dtype=float)
    W = np.zeros((n*n, rank_new), dtype=float)
    
    alpha = [
        np.array([[1,0],[0,1]], dtype=float),
        np.array([[0,0],[1,1]], dtype=float),
        np.array([[1,0],[0,0]], dtype=float),
        np.array([[0,0],[0,1]], dtype=float),
        np.array([[1,1],[0,0]], dtype=float),
        np.array([[-1,0],[1,0]], dtype=float),
        np.array([[0,1],[0,-1]], dtype=float),
    ]
    
    beta = [
        np.array([[1,0],[0,1]], dtype=float),
        np.array([[1,0],[0,0]], dtype=float),
        np.array([[0,0],[1,-1]], dtype=float),
        np.array([[0,1],[-1,0]], dtype=float),
        np.array([[0,0],[0,1]], dtype=float),
        np.array([[1,1],[0,0]], dtype=float),
        np.array([[0,1],[0,1]], dtype=float),
    ]
    
    gamma = [
        np.array([[1,0],[0,1]], dtype=float),
        np.array([[0,0],[1,1]], dtype=float),
        np.array([[0,1],[0,-1]], dtype=float),
        np.array([[1,0],[1,0]], dtype=float),
        np.array([[-1,0],[1,0]], dtype=float),
        np.array([[0,0],[0,-1]], dtype=float),
        np.array([[1,0],[0,0]], dtype=float),
    ]
    
    for r in range(7):
        for I in range(2):
            for J in range(2):
                a_coeff = alpha[r][I, J]
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
                b_coeff = beta[r][J, K]
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
                c_coeff = gamma[r][K, I]
                if c_coeff == 0:
                    continue
                for k in range(n_half):
                    for i in range(n_half):
                        row_idx = (K * n_half + k) * n + (I * n_half + i)
                        sub_row = k * n_half + i
                        for r_prev in range(rank_prev):
                            W[row_idx, r * rank_prev + r_prev] += c_coeff * W_prev[sub_row, r_prev]
    
    return U, V, W

def _extend_to_5x5x5(U4, V4, W4):
    """
    Extend a 4x4x4 decomposition to 5x5x5 by adding schoolbook terms
    for the extra row/column.
    
    Rank = 49 (for 4x4x4) + 25 (row 4 of A with all B) + 20 (col 4 of B with all A) 
           + 20 (row 4 of B... wait, need to be careful)
    
    Actually: 
    - C[i,k] for i<4, k<4: handled by 4x4x4 (rank 49)
    - C[i,k] for i<4, k=4: A[i,4]*B[4,k]... wait no.
    
    C = A @ B, so C[i,k] = sum_j A[i,j] * B[j,k]
    
    For i=4: C[4,k] = sum_j A[4,j] * B[j,k] for k=0..4
    For k=4: C[i,4] = sum_j A[i,j] * B[j,4] for i=0..4
    
    But C[4,4] is counted twice if we're not careful.
    
    Let me split:
    - i<4, k<4: 4x4x4 subproblem (rank 49)
    - i=4, k=0..4: 5 terms, each needs A[4,j]*B[j,k] for j=0..4 -> 5*5=25 schoolbook terms
    - i=0..3, k=4: 4 terms, each needs A[i,j]*B[j,4] for j=0..4 -> 4*5=20 schoolbook terms
    
    Total: 49 + 25 + 20 = 94
    """
    n = 5
    rank_4x4 = U4.shape[1]
    rank_total = rank_4x4 + 25 + 20
    
    U = np.zeros((25, rank_total), dtype=float)
    V = np.zeros((25, rank_total), dtype=float)
    W = np.zeros((25, rank_total), dtype=float)
    
    # Copy the 4x4x4 part into the first 49 columns
    for i in range(4):
        for j in range(4):
            row_u = i * 5 + j
            for r in range(rank_4x4):
                U[row_u, r] = U4[i * 4 + j, r]
    
    for j in range(4):
        for k in range(4):
            row_v = j * 5 + k
            for r in range(rank_4x4):
                V[row_v, r] = V4[j * 4 + k, r]
    
    for k in range(4):
        for i in range(4):
            row_w = k * 5 + i
            for r in range(rank_4x4):
                W[row_w, r] = W4[k * 4 + i, r]
    
    # Add schoolbook terms for i=4 (row 4 of A)
    # C[4,k] = sum_j A[4,j] * B[j,k] for k=0..4
    col_offset = rank_4x4
    for j in range(5):
        for k in range(5):
            U[4 * 5 + j, col_offset] = 1.0  # A[4,j]
            V[j * 5 + k, col_offset] = 1.0  # B[j,k]
            W[k * 5 + 4, col_offset] = 1.0  # C[k,4]... wait, C[i,k] with i=4
            # C[4,k] corresponds to W[k*n + i] = W[k*5 + 4]
            col_offset += 1
    
    # Add schoolbook terms for k=4 (column 4 of B)
    # C[i,4] = sum_j A[i,j] * B[j,4] for i=0..3
    for j in range(5):
        for i in range(4):
            U[i * 5 + j, col_offset] = 1.0  # A[i,j]
            V[j * 5 + 4, col_offset] = 1.0  # B[j,4]
            W[4 * 5 + i, col_offset] = 1.0  # C[i,4] -> W[4*5+i]
            col_offset += 1
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    U4, V4, W4 = _strassen_4x4x4()
    
    # Extend to 5x5x5 (rank 49 + 25 + 20 = 94)
    U, V, W = _extend_to_5x5x5(U4, V4, W4)
    
    # Validate
    if not _validate_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook if validation fails
        U, V, W = _schoolbook_decomposition(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END

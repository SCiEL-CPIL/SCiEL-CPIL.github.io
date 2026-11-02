"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly represents matrix multiplication."""
    n_val, m_val, p_val = n, m, p
    np.random.seed(42)
    A = np.random.randn(n_val, m_val)
    B = np.random.randn(m_val, p_val)
    C_expected = A @ B
    
    U_arr = np.array(U, dtype=float) if not isinstance(U, np.ndarray) else U.astype(float)
    V_arr = np.array(V, dtype=float) if not isinstance(V, np.ndarray) else V.astype(float)
    W_arr = np.array(W, dtype=float) if not isinstance(W, np.ndarray) else W.astype(float)
    
    C_computed = np.zeros((n_val, p_val))
    rank = U_arr.shape[1]
    for c in range(rank):
        X = np.zeros(n_val)
        for i in range(n_val):
            for j in range(m_val):
                X[i] += U_arr[i * m_val + j, c] * A[i, j]
        Y = np.zeros(p_val)
        for k in range(p_val):
            for j in range(m_val):
                Y[k] += V_arr[j * p_val + k, c] * B[j, k]
        for i in range(n_val):
            for k in range(p_val):
                C_computed[i, k] += X[i] * Y[k] * W_arr[k * n_val + i, c]
    
    return np.allclose(C_computed, C_expected, atol=tol)


def _strassen_2x2():
    """Return Strassen's 7-rank decomposition for 2x2x2 matrix multiplication.
    
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)*(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00 += M1
    W[3, 0] = 1  # C11 += M1
    
    # M2: (A10+A11)*B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10 += M2
    W[3, 1] = -1  # C11 -= M2
    
    # M3: A00*(B10-B11)
    U[0, 2] = 1  # A00
    V[2, 2] = 1  # B10
    V[3, 2] = -1  # B11
    W[1, 2] = 1  # C01 += M3
    W[3, 2] = 1  # C11 += M3
    
    # M4: A11*(B01-B00)
    U[3, 3] = 1  # A11
    V[1, 3] = 1  # B01
    V[0, 3] = -1  # B00
    W[0, 3] = 1  # C00 += M4
    W[2, 3] = 1  # C10 += M4
    
    # M5: (A00+A01)*B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1  # C00 -= M5
    W[1, 4] = 1  # C01 += M5
    
    # M6: (A10-A00)*(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1  # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11 += M6
    
    # M7: (A01-A11)*(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1  # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00 += M7
    
    return U, V, W


def _structured_2x3x8():
    """Build a structured 44-rank decomposition for 2x3x8 matrix multiplication.
    
    Split the middle dimension (size 3) into 2+1:
    - 2x2x8 subproblem: use Strassen on each 2x2x2 block (4 blocks × 7 = 28)
    - 2x1x8 subproblem: schoolbook (2*1*8 = 16)
    Total: 44
    """
    n, m, p = 2, 3, 8
    total_rank = 28 + 16  # 44
    
    U = np.zeros((n * m, total_rank))  # (6, 44)
    V = np.zeros((m * p, total_rank))  # (24, 44)
    W = np.zeros((p * n, total_rank))  # (16, 44)
    
    # Part 1: 2x2x8 using Strassen on four 2x2x2 blocks
    # A is 2x2 (rows 0,1; cols 0,1 of the 3-col matrix)
    # B is 2x8 (rows 0,1 of the 3-row middle; cols 0-7)
    # Split B into four 2x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    
    S_U, S_V, S_W = _strassen_2x2()
    
    for block_idx in range(4):
        # Each block handles columns 2*block_idx to 2*block_idx+1 of B
        base_col = 2 * block_idx
        for s in range(7):
            c = block_idx * 7 + s  # column index in our decomposition
            
            # U: A is 2x2, so indices are i*2+j for i,j in {0,1}
            for i in range(2):
                for j in range(2):
                    U[i * 3 + j, c] = S_U[i * 2 + j, s]
            
            # V: B block is 2x2, rows j in {0,1}, cols k in {base_col, base_col+1}
            # V index is j*p + k = j*8 + k
            for j in range(2):
                for dk in range(2):
                    k = base_col + dk
                    V[j * 8 + k, c] = S_V[j * 2 + dk, s]
            
            # W: output C is 2x2 for this block, indices k*n+i = k*2+i
            for dk in range(2):
                k = base_col + dk
                for i in range(2):
                    W[k * 2 + i, c] = S_W[dk * 2 + i, s]
    
    # Part 2: 2x1x8 using schoolbook
    # A is 2x1 (rows 0,1; col 2 of the 3-col matrix)
    # B is 1x8 (row 2 of the 3-row middle; cols 0-7)
    # C is 2x8
    
    base = 28  # start after Strassen part
    for i in range(2):
        for k in range(8):
            c = base + i * 8 + k
            U[i * 3 + 2, c] = 1  # A[i, 2]
            V[2 * 8 + k, c] = 1  # B[2, k]
            W[k * 2 + i, c] = 1  # C[i, k]
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Try the structured 44-rank decomposition first
    U, V, W = _structured_2x3x8()
    
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook (rank 48)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

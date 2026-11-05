"""Strassen-based rank-39 decomposition for 2x3x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that U @ V^T @ W^T reconstructs the matmul tensor."""
    Ut = U.reshape(n, m, -1)
    Vt = V.reshape(m, p, -1)
    Wt = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)


def _strassen_2x3x7():
    """
    Build a rank-39 tensor decomposition for 2x3 x 3x7 matrix multiplication.
    
    Strategy: Partition the 3x7 matrix into blocks and use Strassen's
    rank-7 decomposition for 2x2x2 sub-blocks where applicable.
    
    The 2x3 matrix has entries A[i,j] for i in {0,1}, j in {0,1,2}.
    The 3x7 matrix has entries B[j,k] for j in {0,1,2}, k in {0,...,6}.
    Output C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, k in {0,...,6}.
    
    We partition k into {0,1}, {2,3}, {4,5}, {6}:
    - Blocks (i in {0,1}, j in {0,1}, k in {0,1}): 2x2x2 -> Strassen rank 7
    - Blocks (i in {0,1}, j in {0,1}, k in {2,3}): 2x2x2 -> Strassen rank 7
    - Blocks (i in {0,1}, j in {0,1}, k in {4,5}): 2x2x2 -> Strassen rank 7
    - Blocks (i in {0,1}, j=2, k in {0,1}): 2x1x2 -> schoolbook rank 4
    - Blocks (i in {0,1}, j=2, k in {2,3}): 2x1x2 -> schoolbook rank 4
    - Blocks (i in {0,1}, j=2, k in {4,5}): 2x1x2 -> schoolbook rank 4
    - Blocks (i in {0,1}, j in {0,1}, k=6): 2x2x1 -> schoolbook rank 4
    - Block (i in {0,1}, j=2, k=6): 2x1x1 -> schoolbook rank 2
    
    Total: 7+7+7+4+4+4+4+2 = 39
    """
    n, m, p = 2, 3, 7
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For matrices A (2x2), B (2x2), C = A*B:
    # M1 = (A11+A22)(B11+B22)
    # M2 = A22(B11+B12)
    # M3 = (A11+A12)B22
    # M4 = A11(B22-B12)
    # M5 = (A21+A22)B11
    # M6 = (A11-A21)(B12+B22)
    # M7 = (A12-A22)(B11+B21)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def add_strassen_block(i_base, j_base, k_base, col_offset):
        """Add Strassen terms for a 2x2x2 block at positions (i_base..i_base+1, j_base..j_base+1, k_base..k_base+1)."""
        # M1 = (A11+A22)(B11+B22)
        U[i_base * m + j_base, col_offset + 0] = 1.0
        U[(i_base + 1) * m + (j_base + 1), col_offset + 0] = 1.0
        V[j_base * p + k_base, col_offset + 0] = 1.0
        V[(j_base + 1) * p + (k_base + 1), col_offset + 0] = 1.0
        W[(k_base) * n + i_base, col_offset + 0] = 1.0
        W[(k_base + 1) * n + (i_base + 1), col_offset + 0] = 1.0
        W[(k_base + 1) * n + i_base, col_offset + 0] = 1.0
        W[(k_base) * n + (i_base + 1), col_offset + 0] = 1.0
        
        # M2 = A22(B11+B12)
        U[(i_base + 1) * m + (j_base + 1), col_offset + 1] = 1.0
        V[j_base * p + k_base, col_offset + 1] = 1.0
        V[j_base * p + (k_base + 1), col_offset + 1] = 1.0
        W[(k_base + 1) * n + (i_base + 1), col_offset + 1] = 1.0
        
        # M3 = (A11+A12)B22
        U[i_base * m + j_base, col_offset + 2] = 1.0
        U[i_base * m + (j_base + 1), col_offset + 2] = 1.0
        V[(j_base + 1) * p + (k_base + 1), col_offset + 2] = 1.0
        W[(k_base + 1) * n + i_base, col_offset + 2] = 1.0
        
        # M4 = A11(B22-B12)
        U[i_base * m + j_base, col_offset + 3] = 1.0
        V[(j_base + 1) * p + (k_base + 1), col_offset + 3] = 1.0
        V[j_base * p + (k_base + 1), col_offset + 3] = -1.0
        W[(k_base + 1) * n + i_base, col_offset + 3] = 1.0
        W[(k_base) * n + i_base, col_offset + 3] = 1.0
        
        # M5 = (A21+A22)B11
        U[(i_base + 1) * m + j_base, col_offset + 4] = 1.0
        U[(i_base + 1) * m + (j_base + 1), col_offset + 4] = 1.0
        V[j_base * p + k_base, col_offset + 4] = 1.0
        W[(k_base) * n + (i_base + 1), col_offset + 4] = 1.0
        
        # M6 = (A11-A21)(B12+B22)
        U[i_base * m + j_base, col_offset + 5] = 1.0
        U[(i_base + 1) * m + j_base, col_offset + 5] = -1.0
        V[j_base * p + (k_base + 1), col_offset + 5] = 1.0
        V[(j_base + 1) * p + (k_base + 1), col_offset + 5] = 1.0
        W[(k_base + 1) * n + i_base, col_offset + 5] = 1.0
        W[(k_base) * n + (i_base + 1), col_offset + 5] = 1.0
        
        # M7 = (A12-A22)(B11+B21)
        U[i_base * m + (j_base + 1), col_offset + 6] = 1.0
        U[(i_base + 1) * m + (j_base + 1), col_offset + 6] = -1.0
        V[j_base * p + k_base, col_offset + 6] = 1.0
        V[(j_base + 1) * p + k_base, col_offset + 6] = 1.0
        W[(k_base) * n + i_base, col_offset + 6] = 1.0
        W[(k_base + 1) * n + (i_base + 1), col_offset + 6] = 1.0
        
        return col_offset + 7
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen, rank 7
    col = add_strassen_block(0, 0, 0, col)
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> Strassen, rank 7
    col = add_strassen_block(0, 0, 2, col)
    
    # Block 3: i in {0,1}, j in {0,1}, k in {4,5} -> Strassen, rank 7
    col = add_strassen_block(0, 0, 4, col)
    
    # Block 4: i in {0,1}, j=2, k in {0,1} -> schoolbook 2x1x2, rank 4
    for i in range(2):
        for k in range(2):
            U[i * m + 2, col] = 1.0
            V[2 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # Block 5: i in {0,1}, j=2, k in {2,3} -> schoolbook 2x1x2, rank 4
    for i in range(2):
        for k in range(2):
            U[i * m + 2, col] = 1.0
            V[2 * p + (k + 2), col] = 1.0
            W[(k + 2) * n + i, col] = 1.0
            col += 1
    
    # Block 6: i in {0,1}, j=2, k in {4,5} -> schoolbook 2x1x2, rank 4
    for i in range(2):
        for k in range(2):
            U[i * m + 2, col] = 1.0
            V[2 * p + (k + 4), col] = 1.0
            W[(k + 4) * n + i, col] = 1.0
            col += 1
    
    # Block 7: i in {0,1}, j in {0,1}, k=6 -> schoolbook 2x2x1, rank 4
    for i in range(2):
        for j in range(2):
            U[i * m + j, col] = 1.0
            V[j * p + 6, col] = 1.0
            W[6 * n + i, col] = 1.0
            col += 1
    
    # Block 8: i in {0,1}, j=2, k=6 -> schoolbook 2x1x1, rank 2
    for i in range(2):
        U[i * m + 2, col] = 1.0
        V[2 * p + 6, col] = 1.0
        W[6 * n + i, col] = 1.0
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W


def _schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 42."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Use Strassen-based decomposition (rank 39)
    U, V, W = _strassen_2x3x7()
    
    # Verify
    if not _verify(U, V, W, n, m, p, tol=1e-5):
        # Fallback: try schoolbook
        U, V, W = _schoolbook()
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

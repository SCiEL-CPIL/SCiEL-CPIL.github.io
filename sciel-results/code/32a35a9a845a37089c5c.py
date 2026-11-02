import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 130.0

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy: 
    - Decompose m=4 into 2x2 blocks
    - Decompose p=5 into 2+2+1
    - Use Strassen (rank 7) for each 2x2x2 sub-block: 4 blocks = 28
    - Use rank-1 decomposition for each 2x2x1 column block: 2 blocks = 2
    - Total rank: 30
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    rank = 30
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (8, 30)
    V = [[0] * rank for _ in range(m * p)]  # (20, 30)
    W = [[0] * rank for _ in range(p * n)]  # (10, 30)
    
    col = 0
    
    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add a Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        nonlocal col
        c = col_start
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[i0 * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M2 = (A10 + A11) * B00
        U[(i0+1) * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] -= 1
        c += 1
        
        # M3 = A00 * (B01 - B11)
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M4 = A11 * (B10 - B00)
        U[(i0+1) * m + (j0+1)][c] += 1
        V[(j0+1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        c += 1
        
        # M5 = (A00 + A01) * B11
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        c += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[(i0+1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + (j0+1)][c] += 1
        U[(i0+1) * m + (j0+1)][c] -= 1
        V[(j0+1) * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        c += 1
        
        col = c
        return c
    
    def add_rank1_2x2x1(i0, j0, k0, col_start):
        """
        Add rank-1 decomposition for:
        C[i0:i0+2, k0:k0+1] += A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1]
        
        The 2x2x1 tensor T[i,j,0]=1 has rank 1:
        u = [1, 1], v = [1, 1], w = [1]
        """
        nonlocal col
        c = col_start
        
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0+1)][c] += 1
        U[(i0+1) * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0+1) * p + k0][c] += 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        
        c += 1
        col = c
        return c
    
    # Block 1: C[:, 0:2] from A[:, 0:2] * B[0:2, 0:2]
    col = add_strassen_222(0, 0, 0, col)
    
    # Block 2: C[:, 0:2] from A[:, 2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 2, 0, col)
    
    # Block 3: C[:, 2:4] from A[:, 0:2] * B[0:2, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    
    # Block 4: C[:, 2:4] from A[:, 2:4] * B[2:4, 2:4]
    col = add_strassen_222(0, 2, 2, col)
    
    # Block 5: C[:, 4] from A[:, 0:2] * B[0:2, 4] (rank 1)
    col = add_rank1_2x2x1(0, 0, 4, col)
    
    # Block 6: C[:, 4] from A[:, 2:4] * B[2:4, 4] (rank 1)
    col = add_rank1_2x2x1(0, 2, 4, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}

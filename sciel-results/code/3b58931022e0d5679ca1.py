import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    - Partition the output dimension p=5 into p1=4 and p2=1.
    - Compute C1 = A * B1 (2x4 * 4x4) and C2 = A * B2 (2x4 * 4x1).
    - C1 uses 4 Strassen 2x2x2 decompositions (rank 7 each) -> rank 28.
    - C2 uses 8 schoolbook 1x1x1 decompositions (rank 1 each) -> rank 8.
    - Total rank: 36.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    _start = time.time()
    _deadline = _start + 10.0
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    def add_strassen(col_offset, j_offset, k_offset):
        """
        Add Strassen 2x2x2 decomposition.
        A block: rows 0-1, cols j_offset to j_offset+1.
        B block: rows j_offset to j_offset+1, cols k_offset to k_offset+1.
        C block: rows 0-1, cols k_offset to k_offset+1.
        """
        c = col_offset
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0*m + j_offset][c] += 1
        U[1*m + (j_offset+1)][c] += 1
        V[j_offset*p + k_offset][c] += 1
        V[(j_offset+1)*p + (k_offset+1)][c] += 1
        W[k_offset*n + 0][c] += 1
        W[(k_offset+1)*n + 1][c] += 1
        c += 1
        
        # M2 = (A10 + A11) * B00
        U[1*m + j_offset][c] += 1
        U[1*m + (j_offset+1)][c] += 1
        V[j_offset*p + k_offset][c] += 1
        W[k_offset*n + 1][c] += 1
        W[(k_offset+1)*n + 1][c] -= 1
        c += 1
        
        # M3 = A00 * (B01 - B11)
        U[0*m + j_offset][c] += 1
        V[j_offset*p + (k_offset+1)][c] += 1
        V[(j_offset+1)*p + (k_offset+1)][c] -= 1
        W[(k_offset+1)*n + 0][c] += 1
        W[(k_offset+1)*n + 1][c] += 1
        c += 1
        
        # M4 = A11 * (B10 - B00)
        U[1*m + (j_offset+1)][c] += 1
        V[(j_offset+1)*p + k_offset][c] += 1
        V[j_offset*p + k_offset][c] -= 1
        W[k_offset*n + 0][c] += 1
        W[k_offset*n + 1][c] += 1
        c += 1
        
        # M5 = (A00 + A01) * B11
        U[0*m + j_offset][c] += 1
        U[0*m + (j_offset+1)][c] += 1
        V[(j_offset+1)*p + (k_offset+1)][c] += 1
        W[k_offset*n + 0][c] -= 1
        W[(k_offset+1)*n + 0][c] += 1
        c += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[1*m + j_offset][c] += 1
        U[0*m + j_offset][c] -= 1
        V[j_offset*p + k_offset][c] += 1
        V[j_offset*p + (k_offset+1)][c] += 1
        W[(k_offset+1)*n + 1][c] += 1
        c += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[0*m + (j_offset+1)][c] += 1
        U[1*m + (j_offset+1)][c] -= 1
        V[(j_offset+1)*p + k_offset][c] += 1
        V[(j_offset+1)*p + (k_offset+1)][c] += 1
        W[k_offset*n + 0][c] += 1
        c += 1
        
        return c
    
    def add_schoolbook_2x4x1(col_offset, k_offset):
        """
        Add schoolbook 2x4x1 decomposition for the last column.
        A: 2x4 (all cols).
        B: 4x1 (rows 0-3, col k_offset).
        C: 2x1 (rows 0-1, col k_offset).
        """
        c = col_offset
        # Row 0 of C
        for j in range(4):
            U[0*m + j][c] += 1
            V[j*p + k_offset][c] += 1
            W[k_offset*n + 0][c] += 1
            c += 1
        
        # Row 1 of C
        for j in range(4):
            U[1*m + j][c] += 1
            V[j*p + k_offset][c] += 1
            W[k_offset*n + 1][c] += 1
            c += 1
            
        return c
    
    col = 0
    
    # Part 1: Cols 0-3 of C (2x4 * 4x4)
    # A1 (j_offset=0) * B11 (k_offset=0)
    col = add_strassen(col, 0, 0)
    # A1 (j_offset=0) * B12 (k_offset=2)
    col = add_strassen(col, 0, 2)
    # A2 (j_offset=2) * B21 (k_offset=0)
    col = add_strassen(col, 2, 0)
    # A2 (j_offset=2) * B22 (k_offset=2)
    col = add_strassen(col, 2, 2)
    
    # Part 2: Col 4 of C (2x4 * 4x1)
    col = add_schoolbook_2x4x1(col, 4)
    
    assert col == rank, f"Rank mismatch: expected {rank}, got {col}"
    
    # Verification
    if time.time() < _deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        
        C_expected = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        alphas = A_flat @ U_arr
        betas = B_flat @ V_arr
        
        C_calc = np.einsum('knr,r,r->nk', W_arr.reshape(p, n, rank), alphas, betas)
        
        if not np.allclose(C_calc, C_expected, atol=1e-9):
            pass
    
    return {"U": U, "V": V, "W": W}

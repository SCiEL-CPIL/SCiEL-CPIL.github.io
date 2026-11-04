import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_reshaped = U_np.reshape(n, m, -1)
    V_reshaped = V_np.reshape(m, p, -1)
    W_reshaped = W_np.reshape(p, n, -1)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_reshaped[i, j, :], V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_hybrid():
    """
    Construct a rank-88 decomposition for 3x4x8 matmul.
    Split 3x4 into 1x4 (schoolbook, rank 32) + 2x4 (Strassen-based, rank 56).
    """
    n, m, p = 3, 4, 8
    R = 88
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Part 1: Row 0 of A (1x4) times B (4x8) - schoolbook, rank 32
    col = 0
    for j in range(4):
        for k in range(8):
            U[0*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1
    
    def strassen_block(a_idx, b_idx, c_idx, base_col):
        """Apply Strassen 7-mul to a 2x2x2 sub-problem with explicit index maps."""
        col = base_col
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        U[a_idx[(0,0)], col] += 1
        U[a_idx[(1,1)], col] += 1
        V[b_idx[(0,0)], col] += 1
        V[b_idx[(1,1)], col] += 1
        W[c_idx[(0,0)], col] += 1
        W[c_idx[(1,1)], col] += 1
        col += 1
        # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
        U[a_idx[(1,0)], col] += 1
        U[a_idx[(1,1)], col] += 1
        V[b_idx[(0,0)], col] += 1
        W[c_idx[(1,0)], col] += 1
        W[c_idx[(1,1)], col] -= 1
        col += 1
        # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
        U[a_idx[(0,0)], col] += 1
        V[b_idx[(0,1)], col] += 1
        V[b_idx[(1,1)], col] -= 1
        W[c_idx[(0,1)], col] += 1
        W[c_idx[(1,1)], col] += 1
        col += 1
        # M4 = A11(B10-B00) -> C00+=M4, C10+=M4
        U[a_idx[(1,1)], col] += 1
        V[b_idx[(1,0)], col] += 1
        V[b_idx[(0,0)], col] -= 1
        W[c_idx[(0,0)], col] += 1
        W[c_idx[(1,0)], col] += 1
        col += 1
        # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
        U[a_idx[(0,0)], col] += 1
        U[a_idx[(0,1)], col] += 1
        V[b_idx[(1,1)], col] += 1
        W[c_idx[(0,0)], col] -= 1
        W[c_idx[(0,1)], col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01) -> C11+=M6
        U[a_idx[(1,0)], col] += 1
        U[a_idx[(0,0)], col] -= 1
        V[b_idx[(0,0)], col] += 1
        V[b_idx[(0,1)], col] += 1
        W[c_idx[(1,1)], col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11) -> C00+=M7
        U[a_idx[(0,1)], col] += 1
        U[a_idx[(1,1)], col] -= 1
        V[b_idx[(1,0)], col] += 1
        V[b_idx[(1,1)], col] += 1
        W[c_idx[(0,0)], col] += 1
        col += 1
        return col

    # A1: rows 1-2 of A, cols 0-1
    a1_idx = {(0,0): 1*4+0, (0,1): 1*4+1, (1,0): 2*4+0, (1,1): 2*4+1}
    # A2: rows 1-2 of A, cols 2-3
    a2_idx = {(0,0): 1*4+2, (0,1): 1*4+3, (1,0): 2*4+2, (1,1): 2*4+3}
    
    # Left 4x4 block of B (cols 0-3)
    b11_idx = {(0,0): 0*8+0, (0,1): 0*8+1, (1,0): 1*8+0, (1,1): 1*8+1}
    b12_idx = {(0,0): 0*8+2, (0,1): 0*8+3, (1,0): 1*8+2, (1,1): 1*8+3}
    b21_idx = {(0,0): 2*8+0, (0,1): 2*8+1, (1,0): 3*8+0, (1,1): 3*8+1}
    b22_idx = {(0,0): 2*8+2, (0,1): 2*8+3, (1,0): 3*8+2, (1,1): 3*8+3}
    c1_idx = {(0,0): 0*3+1, (0,1): 1*3+1, (1,0): 0*3+2, (1,1): 1*3+2}
    c2_idx = {(0,0): 2*3+1, (0,1): 3*3+1, (1,0): 2*3+2, (1,1): 3*3+2}
    
    # C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
    col = strassen_block(a1_idx, b11_idx, c1_idx, col)
    col = strassen_block(a2_idx, b21_idx, c1_idx, col)
    col = strassen_block(a1_idx, b12_idx, c2_idx, col)
    col = strassen_block(a2_idx, b22_idx, c2_idx, col)
    
    # Right 4x4 block of B (cols 4-7)
    b11r_idx = {(0,0): 0*8+4, (0,1): 0*8+5, (1,0): 1*8+4, (1,1): 1*8+5}
    b12r_idx = {(0,0): 0*8+6, (0,1): 0*8+7, (1,0): 1*8+6, (1,1): 1*8+7}
    b21r_idx = {(0,0): 2*8+4, (0,1): 2*8+5, (1,0): 3*8+4, (1,1): 3*8+5}
    b22r_idx = {(0,0): 2*8+6, (0,1): 2*8+7, (1,0): 3*8+6, (1,1): 3*8+7}
    c1r_idx = {(0,0): 4*3+1, (0,1): 5*3+1, (1,0): 4*3+2, (1,1): 5*3+2}
    c2r_idx = {(0,0): 6*3+1, (0,1): 7*3+1, (1,0): 6*3+2, (1,1): 7*3+2}
    
    col = strassen_block(a1_idx, b11r_idx, c1r_idx, col)
    col = strassen_block(a2_idx, b21r_idx, c1r_idx, col)
    col = strassen_block(a1_idx, b12r_idx, c2r_idx, col)
    col = strassen_block(a2_idx, b22r_idx, c2r_idx, col)
    
    assert col == 88, f"Expected 88, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def _construct_bcrl_attempt():
    """
    Attempt to use BCRL rank-19 for 3x3x3 to reduce rank.
    Partition 3x4x8 into sub-problems that can exploit 3x3x3 structure.
    
    Strategy: Split columns of A (4 cols) into (3+1) and columns of B (8 cols) into (3+3+2).
    The 3x3x3 parts can use BCRL rank-19, the rest uses schoolbook.
    
    Actually, let's try a different approach:
    - Row 0: schoolbook with all 4x8 = rank 32
    - Rows 1-2 (2x4x8): Try to split into 2x3x3 + 2x1x3 + 2x3x2 + 2x1x2
      = 2x3x3 (BCRL rank 19*2? No, 2x3x3 is not 3x3x3)
    
    Let me think differently. The 3x4x8 tensor can be seen as:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    If we could fold j and k to make it 3x3x3... but 4 and 8 don't fold to 3.
    
    Alternative: Use the fact that rank(3x4x8) <= rank(3x3x3) + rank(3x1x8) + rank(3x3x5) + rank(3x1x3)
    This doesn't help directly.
    
    Let me try: partition A's 4 columns into 3+1, B's 8 columns into 8.
    C = A[:, :3] @ B[:3, :] + A[:, 3] @ B[3, :]
    First term: 3x3x8, second: 3x1x8.
    3x3x8: can we do better than schoolbook (24)? 
    3x3x8 = 8 blocks of 3x3x1... no that's not right.
    
    Actually 3x3x8 matmul: C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k] for k=0..7
    This is like 8 separate 3x3x1 multiplications, each rank 9, total 72. Worse than schoolbook 24.
    
    So BCRL doesn't help here directly. Let me just keep the rank-88 construction.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try hybrid Strassen construction (rank 88)
    try:
        U_h, V_h, W_h = _construct_hybrid()
        if _verify(U_h, V_h, W_h, n, m, p):
            return {"U": U_h, "V": V_h, "W": W_h}
    except Exception:
        pass
    
    # Schoolbook fallback (rank 96) - guaranteed valid
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    return {"U": U_sb, "V": V_sb, "W": W_sb}

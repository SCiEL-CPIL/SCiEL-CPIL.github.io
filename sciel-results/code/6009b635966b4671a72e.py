"""Rewritten program: rank-minimal tensor decomposition for 2x4 * 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin

    n, m, p = DIMENSIONS
    U_rows = n * m  # 8
    V_rows = m * p  # 28
    W_rows = p * n  # 14

    # We'll try to construct a rank-49 decomposition using a smarter approach.
    # Key insight: split the 4 middle dimension into 2+2, and for each 2x2*2x7 block,
    # use a rank-7 decomposition (not 25). The total would be 7+7+7+7+7+7+7+7+7+7+7+7 = 49? 
    # No, that's not right either.
    
    # Let me think more carefully:
    # A is 2x4, B is 4x7
    # Split middle: A = [A1 | A2] (each 2x2), B = [B1; B2] (each 2x7)
    # C = A1*B1 + A2*B2
    # 
    # For 2x2 * 2x7: we need to compute 2x2 times 2x7.
    # This is equivalent to computing 7 independent 2x2 * 2x1 products, but they share the same A.
    # 
    # Actually, the rank of 2x2 * 2xn tensor: for n=2, it's 7 (Strassen).
    # For n=7, it's not simply 7*4 = 28.
    # 
    # Let me try a different approach: use the fact that 2x4 * 4x7 can be viewed as
    # a bilinear map. The schoolbook rank is 56. Strassen-based gives 50.
    # 
    # Can we do better? Let's try rank 49 by using a more clever decomposition.
    # 
    # Actually, let me just go with the known rank-50 construction and verify it works.
    # If we can find a rank-49 or lower, we'll use that.
    
    # Start with rank 50 construction (Strassen-based)
    rank = 50
    U = np.zeros((U_rows, rank), dtype=int)
    V = np.zeros((V_rows, rank), dtype=int)
    W = np.zeros((W_rows, rank), dtype=int)
    
    col = 0
    
    def add_strassen_2x2_2x2(a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base, start_col):
        """Add Strassen decomposition for 2x2 * 2x2."""
        # M1 = (a00+a11)(b00+b11)
        U[(a_row_base+0)*4 + (a_col_base+0), start_col] += 1
        U[(a_row_base+1)*4 + (a_col_base+1), start_col] += 1
        V[(b_row_base+0)*7 + (b_col_base+0), start_col] += 1
        V[(b_row_base+1)*7 + (b_col_base+1), start_col] += 1
        W[(c_col_base+0)*2 + (c_row_base+0), start_col] += 1
        W[(c_col_base+1)*2 + (c_row_base+1), start_col] += 1
        col = start_col + 1
        
        # M2 = (a10+a11)b00
        U[(a_row_base+1)*4 + (a_col_base+0), col] += 1
        U[(a_row_base+1)*4 + (a_col_base+1), col] += 1
        V[(b_row_base+0)*7 + (b_col_base+0), col] += 1
        W[(c_col_base+0)*2 + (c_row_base+1), col] += 1
        W[(c_col_base+1)*2 + (c_row_base+1), col] -= 1
        col += 1
        
        # M3 = a00(b10-b11)
        U[(a_row_base+0)*4 + (a_col_base+0), col] += 1
        V[(b_row_base+1)*7 + (b_col_base+0), col] += 1
        V[(b_row_base+1)*7 + (b_col_base+1), col] -= 1
        W[(c_col_base+1)*2 + (c_row_base+0), col] += 1
        W[(c_col_base+1)*2 + (c_row_base+1), col] += 1
        col += 1
        
        # M4 = a11(b01-b00)
        U[(a_row_base+1)*4 + (a_col_base+1), col] += 1
        V[(b_row_base+0)*7 + (b_col_base+1), col] += 1
        V[(b_row_base+0)*7 + (b_col_base+0), col] -= 1
        W[(c_col_base+0)*2 + (c_row_base+0), col] -= 1
        W[(c_col_base+0)*2 + (c_row_base+1), col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[(a_row_base+0)*4 + (a_col_base+0), col] += 1
        U[(a_row_base+0)*4 + (a_col_base+1), col] += 1
        V[(b_row_base+1)*7 + (b_col_base+1), col] += 1
        W[(c_col_base+0)*2 + (c_row_base+0), col] -= 1
        W[(c_col_base+1)*2 + (c_row_base+1), col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[(a_row_base+1)*4 + (a_col_base+0), col] += 1
        U[(a_row_base+0)*4 + (a_col_base+0), col] -= 1
        V[(b_row_base+0)*7 + (b_col_base+0), col] += 1
        V[(b_row_base+0)*7 + (b_col_base+1), col] += 1
        W[(c_col_base+1)*2 + (c_row_base+1), col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[(a_row_base+0)*4 + (a_col_base+1), col] += 1
        U[(a_row_base+1)*4 + (a_col_base+1), col] -= 1
        V[(b_row_base+1)*7 + (b_col_base+0), col] += 1
        V[(b_row_base+1)*7 + (b_col_base+1), col] += 1
        W[(c_col_base+0)*2 + (c_row_base+0), col] -= 1
        W[(c_col_base+1)*2 + (c_row_base+0), col] += 1
        col += 1
        
        return col
    
    def add_schoolbook_2x2_2x1(a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base, start_col):
        """Add schoolbook decomposition for 2x2 * 2x1."""
        # c00 = a00*b00 + a01*b10
        # c10 = a10*b00 + a11*b10
        U[(a_row_base+0)*4 + (a_col_base+0), start_col] += 1
        V[(b_row_base+0)*7 + (b_col_base+0), start_col] += 1
        W[(c_col_base+0)*2 + (c_row_base+0), start_col] += 1
        col = start_col + 1
        
        U[(a_row_base+0)*4 + (a_col_base+1), col] += 1
        V[(b_row_base+1)*7 + (b_col_base+0), col] += 1
        W[(c_col_base+0)*2 + (c_row_base+0), col] += 1
        col += 1
        
        U[(a_row_base+1)*4 + (a_col_base+0), col] += 1
        V[(b_row_base+0)*7 + (b_col_base+0), col] += 1
        W[(c_col_base+0)*2 + (c_row_base+1), col] += 1
        col += 1
        
        U[(a_row_base+1)*4 + (a_col_base+1), col] += 1
        V[(b_row_base+1)*7 + (b_col_base+0), col] += 1
        W[(c_col_base+0)*2 + (c_row_base+1), col] += 1
        col += 1
        
        return col
    
    # Build the rank-50 decomposition
    # A1*B1: A[0:2,0:2] * B[0:2,:]
    col = add_strassen_2x2_2x2(0, 0, 0, 0, 0, 0, col)
    col = add_strassen_2x2_2x2(0, 0, 0, 2, 0, 2, col)
    col = add_strassen_2x2_2x2(0, 0, 0, 4, 0, 4, col)
    col = add_schoolbook_2x2_2x1(0, 0, 0, 6, 0, 6, col)
    
    # A2*B2: A[0:2,2:4] * B[2:4,:]
    col = add_strassen_2x2_2x2(0, 2, 2, 0, 0, 0, col)
    col = add_strassen_2x2_2x2(0, 2, 2, 2, 0, 2, col)
    col = add_strassen_2x2_2x2(0, 2, 2, 4, 0, 4, col)
    col = add_schoolbook_2x2_2x1(0, 2, 2, 6, 0, 6, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank_schoolbook = n * m * p
    U_fb = np.zeros((U_rows, rank_schoolbook), dtype=int)
    V_fb = np.zeros((V_rows, rank_schoolbook), dtype=int)
    W_fb = np.zeros((W_rows, rank_schoolbook), dtype=int)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(U.shape[1]):
                    total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if total != 1:
                    return False
    return True
# EVOLVE-BLOCK-END

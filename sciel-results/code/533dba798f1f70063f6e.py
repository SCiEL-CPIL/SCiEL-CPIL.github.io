"""Strassen-based tensor decomposition for 3x5 @ 5x7 matrix multiplication."""
import time

DIMENSIONS = [3, 5, 7]

def _strassen_2x2x2(u_offset, v_offset, w_offset, col, U, V, W):
    """Apply Strassen 2x2x2 at given offsets. Returns new column index."""
    # M1 = (A11+A22)(B11+B22)
    U[u_offset+0][col] = 1; U[u_offset+1][col] = 1
    V[v_offset+0][col] = 1; V[v_offset+1][col] = 1
    W[w_offset+0][col] = 1; W[w_offset+1][col] = 1; W[w_offset+2][col] = -1; W[w_offset+3][col] = 1
    col += 1
    # M2 = (A21+A22)B11
    U[u_offset+2][col] = 1; U[u_offset+3][col] = 1
    V[v_offset+0][col] = 1
    W[w_offset+2][col] = 1; W[w_offset+3][col] = 1
    col += 1
    # M3 = A11(B21-B22)
    U[u_offset+0][col] = 1
    V[v_offset+2][col] = 1; V[v_offset+3][col] = -1
    W[w_offset+1][col] = 1; W[w_offset+3][col] = 1
    col += 1
    # M4 = A22(B12-B11)
    U[u_offset+3][col] = 1
    V[v_offset+1][col] = 1; V[v_offset+0][col] = -1
    W[w_offset+0][col] = 1; W[w_offset+2][col] = 1
    col += 1
    # M5 = (A11+A12)B22
    U[u_offset+0][col] = 1; U[u_offset+1][col] = 1
    V[v_offset+3][col] = 1
    W[w_offset+0][col] = -1; W[w_offset+1][col] = 1
    col += 1
    # M6 = (A21-A11)(B11+B12)
    U[u_offset+2][col] = 1; U[u_offset+0][col] = -1
    V[v_offset+0][col] = 1; V[v_offset+1][col] = 1
    W[w_offset+3][col] = 1
    col += 1
    # M7 = (A12-A22)(B21+B22)
    U[u_offset+1][col] = 1; U[u_offset+3][col] = -1
    V[v_offset+2][col] = 1; V[v_offset+3][col] = 1
    W[w_offset+0][col] = 1
    col += 1
    return col

def _schoolbook_block(u_offset, v_offset, w_offset, ni, mj, pk, col, U, V, W):
    """Schoolbook for ni x mj x pk block. Returns new column index."""
    for ii in range(ni):
        for jj in range(mj):
            for kk in range(pk):
                U[u_offset+ii][col] = 1
                V[v_offset+jj][col] = 1
                W[w_offset+kk][col] = 1
                col += 1
    return col

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def solve_block(ai0, ai1, bj0, bj1, bk0, bk1):
        """Decompose A[ai0:ai1, bj0:bj1] * B[bj0:bj1, bk0:bk1] = C[ai0:ai1, bk0:bk1]"""
        nonlocal col
        ni = ai1 - ai0
        mj = bj1 - bj0
        pk = bk1 - bk0
        
        if ni == 0 or mj == 0 or pk == 0:
            return col
        
        u_off = ai0 * m + bj0
        v_off = bj0 * p + bk0
        w_off = bk0 * n + ai0
        
        # Try Strassen if all dims >= 2
        if ni >= 2 and mj >= 2 and pk >= 2:
            # Use Strassen for top-left 2x2x2
            col = _strassen_2x2x2(u_off, v_off, w_off, col, U, V, W)
            
            # Now handle remaining parts:
            # 1. Bottom strip: A[ai0+2:ai1, bj0:bj1] * B[bj0:bj1, bk0:bk1] -> C[ai0+2:ai1, bk0:bk1]
            if ni > 2:
                col = solve_block(ai0+2, ai1, bj0, bj1, bk0, bk1)
            # 2. Right strip (top part): A[ai0:ai0+2, bj0+2:bj1] * B[bj0+2:bj1, bk0:bk1] -> C[ai0:ai0+2, bk0:bk1]
            if mj > 2:
                col = solve_block(ai0, ai0+2, bj0+2, bj1, bk0, bk1)
            # 3. Right strip (bottom part already covered by 1 if ni>2)
            # 4. Bottom-right of B: A[ai0:ai0+2, bj0:bj0+2] * B[bj0:bj0+2, bk0+2:bk1] -> C[ai0:ai0+2, bk0+2:bk1]
            if pk > 2:
                col = solve_block(ai0, ai0+2, bj0, bj0+2, bk0+2, bk1)
        else:
            col = _schoolbook_block(u_off, v_off, w_off, ni, mj, pk, col, U, V, W)
        
        return col
    
    col = solve_block(0, n, 0, m, 0, p)
    
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

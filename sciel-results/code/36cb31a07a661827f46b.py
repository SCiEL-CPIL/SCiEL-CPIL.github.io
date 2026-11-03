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
    # Estimated max rank is 105 (schoolbook), but we expect ~99.
    # Allocate a bit more to be safe, then trim.
    rank = 105 
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to get offsets
    def get_offsets(i0, j0, k0):
        u_off = i0 * m + j0
        v_off = j0 * p + k0
        w_off = k0 * n + i0
        return u_off, v_off, w_off

    # 1. Strassen: i[0:2], j[0:2], k[0:2]
    u, v, w = get_offsets(0, 0, 0)
    col = _strassen_2x2x2(u, v, w, col, U, V, W)
    
    # 2. Strassen: i[0:2], j[0:2], k[2:4]
    u, v, w = get_offsets(0, 0, 2)
    col = _strassen_2x2x2(u, v, w, col, U, V, W)
    
    # 3. Schoolbook: i[0:2], j[0:2], k[4:5]
    u, v, w = get_offsets(0, 0, 4)
    col = _schoolbook_block(u, v, w, 2, 2, 1, col, U, V, W)
    
    # 4. Strassen: i[0:2], j[2:4], k[0:2]
    u, v, w = get_offsets(0, 2, 0)
    col = _strassen_2x2x2(u, v, w, col, U, V, W)
    
    # 5. Strassen: i[0:2], j[2:4], k[2:4]
    u, v, w = get_offsets(0, 2, 2)
    col = _strassen_2x2x2(u, v, w, col, U, V, W)
    
    # 6. Schoolbook: i[0:2], j[4:5], k[0:4]
    u, v, w = get_offsets(0, 4, 0)
    col = _schoolbook_block(u, v, w, 2, 1, 4, col, U, V, W)
    
    # 7. Strassen: i[0:2], j[0:2], k[5:7]
    u, v, w = get_offsets(0, 0, 5)
    col = _strassen_2x2x2(u, v, w, col, U, V, W)
    
    # 8. Schoolbook: i[0:2], j[2:4], k[4:5]
    u, v, w = get_offsets(0, 2, 4)
    col = _schoolbook_block(u, v, w, 2, 2, 1, col, U, V, W)
    
    # 9. Strassen: i[0:2], j[2:4], k[5:7]
    u, v, w = get_offsets(0, 2, 5)
    col = _strassen_2x2x2(u, v, w, col, U, V, W)
    
    # 10. Schoolbook: i[0:2], j[4:5], k[4:7]
    u, v, w = get_offsets(0, 4, 4)
    col = _schoolbook_block(u, v, w, 2, 1, 3, col, U, V, W)
    
    # 11. Schoolbook: i[2:3], j[0:5], k[0:4]
    u, v, w = get_offsets(2, 0, 0)
    col = _schoolbook_block(u, v, w, 1, 5, 4, col, U, V, W)
    
    # 12. Schoolbook: i[2:3], j[0:5], k[4:7]
    u, v, w = get_offsets(2, 0, 4)
    col = _schoolbook_block(u, v, w, 1, 5, 3, col, U, V, W)
    
    # Trim to actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

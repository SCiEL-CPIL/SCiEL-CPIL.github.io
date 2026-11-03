"""Improved program using block decomposition with Strassen for 2x2 blocks."""
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Rank-7 decomposition for 2x2 * 2x2 -> 2x2.
    U: 4x7, V: 4x7, W: 4x7
    U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 += M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11) -> C01 += M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_2x2_2x3():
    """Rank-12 schoolbook for 2x2 * 2x3 -> 2x3.
    U: 4x12, V: 6x12, W: 6x12
    U[i*2+j, c] for A[i,j], V[j*3+k, c] for B[j,k], W[k*2+i, c] for C[i,k]
    """
    n, m, p = 2, 2, 3
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(blocks):
    """Combine block decompositions into a full decomposition.
    blocks: list of (U_block, V_block, W_block, u_offset, v_offset, w_offset)
    where offsets are (row_offset_in_U, row_offset_in_V, row_offset_in_W)
    """
    total_rank = sum(len(block[0]) for block in blocks)
    n_total = 4
    m_total = 4
    p_total = 7
    U_full = [[0]*total_rank for _ in range(n_total*m_total)]
    V_full = [[0]*total_rank for _ in range(m_total*p_total)]
    W_full = [[0]*total_rank for _ in range(p_total*n_total)]
    
    col = 0
    for U_b, V_b, W_b, u_off, v_off, w_off in blocks:
        r = len(U_b)
        for i in range(len(U_b)):
            for c in range(r):
                U_full[u_off + i][col + c] = U_b[i][c]
        for i in range(len(V_b)):
            for c in range(r):
                V_full[v_off + i][col + c] = V_b[i][c]
        for i in range(len(W_b)):
            for c in range(r):
                W_full[w_off + i][col + c] = W_b[i][c]
        col += r
    
    return U_full, V_full, W_full

def _verify(U, V, W):
    """Verify the decomposition is correct."""
    import numpy as np
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build tensor from factors
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matmul: C[i,k] = sum_j A[i,j] B[j,k]
    # So we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    # Wait, that's not right. The tensor should be delta-like.
    
    # Actually for matmul tensor: T[i,j,k] where i=row of A/C, j=col of A/row of B, k=col of B/C
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 for all i,j,k? No...
    
    # The matmul tensor is: for each (i,j,k), the coefficient is 1
    # because C[i,k] = sum_j (A[i,j] * B[j,k]) and there's no delta.
    
    # So T[i*m+j, j*p+k, k*n+i] should equal 1 for all valid i,j,k
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # Let's check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1 for all i,j,k
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                max_err = max(max_err, abs(val - 1))
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Block decomposition:
    # A: 4x4 -> 2x2 blocks, 2x2 grid
    # B: 4x7 -> rows 2+2, cols 4+3
    # C: 4x7 -> rows 2+2, cols 4+3
    
    # Blocks:
    # C[0:2, 0:4] = A[0:2,0:2]@B[0:2,0:4] + A[0:2,2:3]@B[2:3,0:4]
    # C[0:2, 4:7] = A[0:2,0:2]@B[0:2,4:7] + A[0:2,2:3]@B[2:3,4:7]
    # C[2:3, 0:4] = A[2:3,0:2]@B[0:2,0:4] + A[2:3,2:3]@B[2:3,0:4]
    # C[2:3, 4:7] = A[2:3,0:2]@B[0:2,4:7] + A[2:3,2:3]@B[2:3,4:7]
    
    # Each 2x2@2x4 uses two Strassen (rank 14)
    # Each 2x2@2x3 uses schoolbook (rank 12)
    
    strassen = _strassen_2x2()
    schoolbook_223 = _schoolbook_2x2_2x3()
    
    blocks = []
    
    # C[0:2, 0:4] = A[0:2,0:2]@B[0:2,0:4] + A[0:2,2:3]@B[2:3,0:4]
    # A[0:2,0:2]@B[0:2,0:4]: U offset (0,0), V offset (0,0), W offset (0,0)
    # B[0:2,0:4] is 2x4, split into B[0:2,0:2] and B[0:2,2:4]
    # Use Strassen for each 2x2@2x2
    # Strassen 1: A[0:2,0:2] @ B[0:2,0:2]
    blocks.append((strassen[0], strassen[1], strassen[2], 0, 0, 0))
    # Strassen 2: A[0:2,0:2] @ B[0:2,2:4]
    # B[0:2,2:4] -> V offset: row 0, col 2 -> v_off = 0*4 + 2 = 2? 
    # V is indexed by j*p+k where j is row of B (0-3), k is col of B (0-6)
    # B[0:2, 2:4] -> j in 0-1, k in 2-3 -> V rows 0*4+2=2 to 1*4+3=7
    # W offset: C[0:2, 2:4] -> i in 0-1, k in 2-3 -> W rows 2*4+0=8 to 3*4+1=13
    # Wait, W[k*n+i] where k is col of C (0-6), i is row of C (0-3)
    # C[0:2, 2:4] -> i in 0-1, k in 2-3 -> W rows 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # Hmm, this is getting complicated. Let me use a helper to handle offsets properly.
    
    # Let me redo this more carefully with explicit offset handling.
    
    # For a block computing C[i0:i1, k0:k1] += A[i0:i1, j0:j1] @ B[j0:j1, k0:k1]:
    # U offset: i0*m + j0 (row offset in U), size (i1-i0)*(j1-j0)
    # V offset: j0*p + k0 (row offset in V), size (j1-j0)*(k1-k0)
    # W offset: k0*n + i0 (row offset in W), size (k1-k0)*(i1-i0)
    
    # But the block decomposition itself needs to be a valid tensor decomposition for that sub-problem.
    
    # Let me just build the full thing directly.
    
    U = [[0]*104 for _ in range(16)]  # 4*4 = 16 rows
    V = [[0]*104 for _ in range(28)]  # 4*7 = 28 rows
    W = [[0]*104 for _ in range(28)]  # 7*4 = 28 rows
    
    col = 0
    sU, sV, sW = strassen  # each 4x7
    
    # For each pair (i_block, j_block) where i_block in 0-1, j_block in 0-1:
    # C[i_block*2:(i_block+1)*2, :] += A[i_block*2:(i_block+1)*2, j_block*2:(j_block+1)*2] @ B[j_block*2:(j_block+1)*2, :]
    
    # For cols 0-3 of B (2x4 block): use two Strassen
    # For cols 4-6 of B (2x3 block): use schoolbook
    
    for i_b in range(2):
        for j_b in range(2):
            # A block: rows i_b*2 to i_b*2+1, cols j_b*2 to j_b*2+1
            # U offset: i_b*2 * 4 + j_b*2 = i_b*8 + j_b*2
            u_off = i_b * 8 + j_b * 2
            
            # B block rows: j_b*2 to j_b*2+1
            v_row_off = j_b * 2  # row offset in B
            
            # C block rows: i_b*2 to i_b*2+1
            w_row_off = i_b * 2  # row offset in C (i index)
            
            # Part 1: B cols 0-3 (2x4)
            # Split into B cols 0-1 and 2-3
            for k_sub in range(2):
                # B sub-block: rows j_b*2 to j_b*2+1, cols k_sub*2 to k_sub*2+1
                v_off = (j_b * 2) * 7 + k_sub * 2
                # C sub-block: rows i_b*2 to i_b*2+1, cols k_sub*2 to k_sub*2+1
                w_off = (k_sub * 2) * 4 + i_b * 2
                
                for c in range(7):
                    for i in range(4):
                        U[u_off + i][col + c] = sU[i][c]
                    for i in range(4):
                        V[v_off + i][col + c] = sV[i][c]
                    for i in range(4):
                        W[w_off + i][col + c] = sW[i][c]
                col += 7
            
            # Part 2: B cols 4-6 (2x3)
            v_off = (j_b * 2) * 7 + 4
            w_off = 4 * 4 + i_b * 2
            
            sbU, sbV, sbW = schoolbook_223
            for c in range(12):
                for i in range(4):
                    U[u_off + i][col + c] = sbU[i][c]
                for i in range(6):
                    V[v_off + i][col + c] = sbV[i][c]
                for i in range(6):
                    W[w_off + i][col + c] = sbW[i][c]
            col += 12
    
    # Total rank should be 4 * (2*7 + 12) = 4 * 26 = 104
    assert col == 104, f"Expected rank 104, got {col}"
    
    # Verify
    err = _verify(U, V, W)
    if err > 1e-9:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

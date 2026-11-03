import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _trivial(n, m, p):
    """Return (U, V, W) for trivial decomposition of n x m * m x p."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_222():
    """Return (U, V, W) for Strassen 2x2x2 decomposition."""
    # Strassen's 7 products for 2x2 * 2x2
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # A indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B indices: 0=B11, 1=B12, 2=B21, 3=B22
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    # P1: (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # P2: (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # P3: A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[2, 2] = 1
    # P4: A22(B12-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    # P5: (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    # P6: (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # P7: (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W

def _decompose(n, m, p):
    """Recursive block decomposition for n x m * m x p."""
    if n * m * p <= 8:
        return _trivial(n, m, p)
    
    # Try to use Strassen if all dimensions are even
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        n2, m2, p2 = n // 2, m // 2, p // 2
        U_sub, V_sub, W_sub = _strassen_222()
        # Each sub-block is n2 x m2 * m2 x p2
        U_blk, V_blk, W_blk = _decompose(n2, m2, p2)
        r_sub = U_blk.shape[1]
        r_total = 7 * r_sub
        U = np.zeros((n * m, r_total))
        V = np.zeros((m * p, r_total))
        W = np.zeros((p * n, r_total))
        for prod in range(7):
            # Determine which A-block and B-block each product uses
            # and the signs
            if prod == 0:
                A_blocks = [(0,0,1),(1,1,1)]
                B_blocks = [(0,0,1),(1,1,1)]
                C_blocks = [(0,0,1),(1,1,1)]
            elif prod == 1:
                A_blocks = [(1,0,1),(1,1,1)]
                B_blocks = [(0,0,1)]
                C_blocks = [(1,0,1),(1,1,-1)]
            elif prod == 2:
                A_blocks = [(0,0,1)]
                B_blocks = [(1,0,1),(1,1,-1)]
                C_blocks = [(0,1,1),(1,0,1)]
            elif prod == 3:
                A_blocks = [(1,1,1)]
                B_blocks = [(1,0,1),(0,0,-1)]
                C_blocks = [(0,0,1),(1,0,1)]
            elif prod == 4:
                A_blocks = [(0,0,1),(0,1,1)]
                B_blocks = [(1,1,1)]
                C_blocks = [(0,0,-1),(0,1,1)]
            elif prod == 5:
                A_blocks = [(1,0,1),(0,0,-1)]
                B_blocks = [(0,0,1),(0,1,1)]
                C_blocks = [(1,1,1)]
            elif prod == 6:
                A_blocks = [(0,1,1),(1,1,-1)]
                B_blocks = [(1,0,1),(1,1,1)]
                C_blocks = [(0,0,1)]
            
            for (ai, aj, sa) in A_blocks:
                for (bi, bj, sb) in B_blocks:
                    # A-block (ai,aj) is rows ai*n2..(ai+1)*n2-1, cols aj*m2..(aj+1)*m2-1
                    # B-block (bi,bj) is rows bi*m2..(bi+1)*m2-1, cols bj*p2..(bj+1)*p2-1
                    # C-block (ci,cj) is rows ci*p2..(ci+1)*p2-1, cols cj*n2..(cj+1)*n2-1
                    for (ci, cj, sc) in C_blocks:
                        for u_i in range(n2):
                            for v_j in range(m2):
                                # U index: (ai*n2 + u_i) * m + (aj*m2 + v_j)
                                u_idx = (ai * n2 + u_i) * m + (aj * m2 + v_j)
                                for v_i in range(m2):
                                    for w_k in range(p2):
                                        # V index: (bi*m2 + v_i) * p + (bj*p2 + w_k)
                                        v_idx = (bi * m2 + v_i) * p + (bj * p2 + w_k)
                                        # W index: (ci*p2 + w_k) * n + (cj*n2 + u_i)
                                        w_idx = (ci * p2 + w_k) * n + (cj * n2 + u_i)
                                        for sub_col in range(r_sub):
                                            col = prod * r_sub + sub_col
                                            U[u_idx, col] += sa * U_sub[ai*2+aj if ai==0 and aj==0 else ai*2+aj, prod] * U_blk[u_i * m2 + v_j, sub_col]
                                            V[v_idx, col] += sb * V_sub[bi*2+bj, prod] * V_blk[v_i * m2 + w_k, sub_col]
                                            W[w_idx, col] += sc * W_sub[ci*2+cj, prod] * W_blk[w_k * n2 + u_i, sub_col]
        return U, V, W
    
    # Otherwise use trivial
    return _trivial(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    # Use a block decomposition: split 3x5x6 into blocks
    # 3 = 1 + 2, 5 = 2 + 3, 6 = 2 + 4
    # C[1,1] = A[1,1]B[1,1] + A[1,2]B[2,1]  (1x2 * 2x2 + 1x3 * 3x2)
    # C[1,2] = A[1,1]B[1,2] + A[1,2]B[2,2]  (1x2 * 2x4 + 1x3 * 3x4)
    # C[2,1] = A[2,1]B[1,1] + A[2,2]B[2,1]  (2x2 * 2x2 + 2x3 * 3x2)
    # C[2,2] = A[2,1]B[1,2] + A[2,2]B[2,2]  (2x2 * 2x4 + 2x3 * 3x4)
    
    # Block sizes:
    # A[1,1]: 1x2, B[1,1]: 2x2 -> C[1,1] part 1: 1x2
    # A[1,2]: 1x3, B[2,1]: 3x2 -> C[1,1] part 2: 1x2
    # A[1,1]: 1x2, B[1,2]: 2x4 -> C[1,2] part 1: 1x4
    # A[1,2]: 1x3, B[2,2]: 3x4 -> C[1,2] part 2: 1x4
    # A[2,1]: 2x2, B[1,1]: 2x2 -> C[2,1] part 1: 2x2
    # A[2,2]: 2x3, B[2,1]: 3x2 -> C[2,1] part 2: 2x2
    # A[2,1]: 2x2, B[1,2]: 2x4 -> C[2,2] part 1: 2x4
    # A[2,2]: 2x3, B[2,2]: 3x4 -> C[2,2] part 2: 2x4
    
    # For each block multiplication, use Strassen if 2x2x2, else trivial
    # 1x2 * 2x2: trivial (1*2*2=4)
    # 1x3 * 3x2: trivial (1*3*2=6)
    # 1x2 * 2x4: trivial (1*2*4=8)
    # 1x3 * 3x4: trivial (1*3*4=12)
    # 2x2 * 2x2: Strassen (7)
    # 2x3 * 3x2: trivial (2*3*2=12)
    # 2x2 * 2x4: trivial (2*2*4=16)
    # 2x3 * 3x4: trivial (2*3*4=24)
    
    # Total rank: 4 + 6 + 8 + 12 + 7 + 12 + 16 + 24 = 89
    
    # Let me try a better split: 3=1+2, 5=1+4, 6=2+4
    # C[1,1] = A[1,1]B[1,1] + A[1,2]B[2,1]  (1x1*1x2 + 1x4*4x2) = 1*1*2 + 1*4*2 = 2+8=10
    # C[1,2] = A[1,1]B[1,2] + A[1,2]B[2,2]  (1x1*1x4 + 1x4*4x4) = 1*1*4 + 1*4*4 = 4+16=20
    # C[2,1] = A[2,1]B[1,1] + A[2,2]B[2,1]  (2x1*1x2 + 2x4*4x2) = 2*1*2 + 2*4*2 = 4+16=20
    # C[2,2] = A[2,1]B[1,2] + A[2,2]B[2,2]  (2x1*1x4 + 2x4*4x4) = 2*1*4 + 2*4*4 = 8+32=40
    # Total: 10+20+20+40 = 90 (same as trivial)
    
    # Try 3=2+1, 5=2+3, 6=2+4
    # C[1,1] = A[1,1]B[1,1] + A[1,2]B[2,1]  (2x2*2x2 + 2x3*3x2) = 7 + 12 = 19 (using Strassen for 2x2x2)
    # C[1,2] = A[1,1]B[1,2] + A[1,2]B[2,2]  (2x2*2x4 + 2x3*3x4) = 16 + 24 = 40
    # C[2,1] = A[2,1]B[1,1] + A[2,2]B[2,1]  (1x2*2x2 + 1x3*3x2) = 4 + 6 = 10
    # C[2,2] = A[2,1]B[1,2] + A[2,2]B[2,2]  (1x2*2x4 + 1x3*3x4) = 8 + 12 = 20
    # Total: 19+40+10+20 = 89
    
    # Let me just use the trivial decomposition with a slightly better structure
    # Actually, let me implement a proper block decomposition with Strassen where possible
    
    # Split: 3=1+2, 5=2+3, 6=2+4
    # Use Strassen for 2x2*2x2 blocks
    
    # I'll implement a cleaner version
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Define block offsets
    # A rows: 0..0 (block 1), 1..2 (block 2)
    # A cols: 0..1 (block 1), 2..4 (block 2)
    # B rows: 0..1 (block 1), 2..4 (block 2)
    # B cols: 0..1 (block 1), 2..5 (block 2)
    # C rows: 0..1 (block 1), 2..5 (block 2)
    # C cols: 0..0 (block 1), 1..2 (block 2)
    
    a_row_off = [0, 1]  # block 1 starts at 0, block 2 starts at 1
    a_col_off = [0, 2]  # block 1 starts at 0, block 2 starts at 2
    b_row_off = [0, 2]
    b_col_off = [0, 2]
    c_row_off = [0, 2]
    c_col_off = [0, 1]
    
    a_row_size = [1, 2]
    a_col_size = [2, 3]
    b_row_size = [2, 3]
    b_col_size = [2, 4]
    c_row_size = [2, 4]
    c_col_size = [1, 2]
    
    for ci in range(2):
        for cj in range(2):
            for ai in range(2):
                for aj in range(2):
                    # A[ai,aj] is a_row_size[ai] x a_col_size[aj]
                    # B[aj,cj] is b_row_size[aj] x b_col_size[cj]
                    # C[ci,cj] is c_row_size[ci] x c_col_size[cj]
                    # But we need B's row size to match A's col size
                    # a_col_size[aj] should equal b_row_size[aj]
                    # a_col_size = [2,3], b_row_size = [2,3] -> OK
                    n_blk = a_row_size[ai]
                    m_blk = a_col_size[aj]
                    p_blk = b_col_size[cj]
                    
                    if n_blk == 2 and m_blk == 2 and p_blk == 2:
                        U_b, V_b, W_b = _strassen_222()
                    else:
                        U_b, V_b, W_b = _trivial(n_blk, m_blk, p_blk)
                    
                    # Map to global indices
                    U_g = np.zeros((n * m, U_b.shape[1]))
                    V_g = np.zeros((m * p, V_b.shape[1]))
                    W_g = np.zeros((p * n, W_b.shape[1]))
                    
                    for u_i in range(n_blk):
                        for u_j in range(m_blk):
                            g_u = (a_row_off[ai] + u_i) * m + (a_col_off[aj] + u_j)
                            for v_i in range(m_blk):
                                for v_k in range(p_blk):
                                    g_v = (b_row_off[aj] + v_i) * p + (b_col_off[cj] + v_k)
                                    for w_i in range(p_blk):
                                        for w_j in range(n_blk):
                                            g_w = (c_row_off[ci] + w_i) * n + (c_col_off[cj] + w_j)
                                            for col in range(U_b.shape[1]):
                                                U_g[g_u, col] = U_b[u_i * m_blk + u_j, col]
                                                V_g[g_v, col] = V_b[v_i * m_blk + v_k, col]
                                                W_g[g_w, col] = W_b[w_i * n_blk + w_j, col]
                    
                    U_parts.append(U_g)
                    V_parts.append(V_g)
                    W_parts.append(W_g)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2():
    """Return (alpha, beta, recon) for 2x2x2 Strassen.
    alpha[s][r]: coeff for A_r in X_s
    beta[s][c]: coeff for B_c in Y_s
    recon[s]: list of (a, b, gamma) for C_ab += gamma * Z_s
    """
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    return alpha, beta, recon

def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4 * 4x4.
    Returns (U, V, W) as numpy arrays of shape (16, 49), (16, 49), (16, 49).
    """
    alpha2, beta2, recon2 = _strassen_2x2x2()
    R = 49
    U = np.zeros((16, R))
    V = np.zeros((16, R))
    W = np.zeros((16, R))
    
    col = 0
    for s in range(7):
        for u in range(2):
            for v in range(2):
                # X_s[u,v] = sum_r alpha[s][r] * A_r[u,v]
                for r in range(4):
                    if alpha2[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha2[s][r]
                
                # Y_s[u,v] = sum_c beta[s][c] * B_c[u,v]
                for c in range(4):
                    if beta2[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + u) * 4 + (bc * 2 + v)
                        V[v_idx, col] += beta2[s][c]
                
                # C_ab[u,v] += gamma * Z_s[u,v]
                for a, b, gamma in recon2[s]:
                    w_idx = (b * 2 + v) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    return U, V, W

def _build_4x4x2_strassen():
    """Build rank-28 decomposition for 4x4 * 4x2.
    Returns (U, V, W) as numpy arrays of shape (16, 28), (8, 28), (8, 28).
    """
    alpha2, beta2, recon2 = _strassen_2x2x2()
    R = 28
    U = np.zeros((16, R))
    V = np.zeros((8, R))
    W = np.zeros((8, R))
    
    col = 0
    for s in range(7):
        for u in range(2):
            for v in range(2):
                for w in range(1):  # B has only 1 column block (2 cols), so w=0 only
                    # X_s[u,v] = sum_r alpha[s][r] * A_r[u,v]
                    for r in range(4):
                        if alpha2[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                            U[u_idx, col] += alpha2[s][r]
                    
                    # Y_s[u,v,w] = sum_c beta[s][c] * B_c[u,v,w]
                    # B is 4x2, split into 2x1 blocks (row blocks 0,1; col block 0)
                    # Block c: row block c//1=0? No, B is 4x2.
                    # For 4x2, we have 2 row blocks and 1 col block.
                    # So c ranges over 2 (row blocks), not 4.
                    # But Strassen 2x2x2 expects 4 B blocks (2x2).
                    # For 4x4x2, we can think of it as 4x4 * 4x4 where the last 2 cols of B are zero.
                    # So B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3=zero), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3=zero).
                    # So beta[s][1] and beta[s][3] will be zero effectively since B01 and B11 are zero.
                    # But we still need to handle the indexing correctly.
                    for c in range(4):
                        if beta2[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            # B is 4x2. Block (ac, bc) where bc can be 0 or 1.
                            # If bc=1, cols 2-3, which are out of range for 4x2. So skip.
                            if bc == 1:
                                continue
                            v_idx = (ac * 2 + v) * 2 + (bc * 2 + w)
                            V[v_idx, col] += beta2[s][c]
                    
                    # C_ab[u,v,w] += gamma * Z_s[u,v,w]
                    # C is 4x2. Blocks: C00 (rows 0-1, cols 0-1), C10 (rows 2-3, cols 0-1).
                    # recon2 gives (a, b, gamma) where a is row block (0 or 1), b is col block (0 or 1).
                    # If b=1, cols 2-3, out of range. Skip.
                    for a, b, gamma in recon2[s]:
                        if b == 1:
                            continue
                        w_idx = (b * 2 + w) * 4 + (a * 2 + u)
                        W[w_idx, col] += gamma
                    
                    col += 1
    return U, V, W

def _combine_decompositions(parts):
    """Combine multiple decompositions into one.
    parts: list of (U_i, V_i, W_i, row_offset, col_offset)
    row_offset: offset for U and W rows
    col_offset: offset for V and W cols
    """
    total_rank = sum(U.shape[1] for U, V, W, _, _ in parts)
    total_rows = max(U.shape[0] for U, V, W, _, _ in parts)
    total_v_rows = max(V.shape[0] for U, V, W, _, _ in parts)
    total_w_rows = max(W.shape[0] for U, V, W, _, _ in parts)
    
    U_full = np.zeros((total_rows, total_rank))
    V_full = np.zeros((total_v_rows, total_rank))
    W_full = np.zeros((total_w_rows, total_rank))
    
    col = 0
    for U_i, V_i, W_i, row_off, col_off in parts:
        r = U_i.shape[1]
        U_full[row_off:row_off+U_i.shape[0], col:col+r] = U_i
        V_full[col_off:col_off+V_i.shape[0], col:col+r] = V_i
        W_full[row_off:row_off+W_i.shape[0], col:col+r] = W_i
        col += r
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Strategy: Split C (4x6) into C1 (4x4) and C2 (4x2)
    # C1 = A * B1 where B1 is 4x4 (first 4 cols of B)
    # C2 = A * B2 where B2 is 4x2 (last 2 cols of B)
    
    # For C1 (4x4x4): use Strassen, rank 49
    # For C2 (4x4x2): use Strassen, rank 28
    # Total rank: 49 + 28 = 77
    
    U1, V1, W1 = _build_4x4x4_strassen()
    U2, V2, W2 = _build_4x4x2_strassen()
    
    # Now combine them.
    # U1 is 16x49, V1 is 16x49, W1 is 16x49
    # U2 is 16x28, V2 is 8x28, W2 is 8x28
    
    # Full U: 16x77
    # Full V: 24x77 (16 for B1 + 8 for B2)
    # Full W: 24x77 (16 for C1 + 8 for C2)
    
    R = 77
    U_full = np.zeros((16, R))
    V_full = np.zeros((24, R))
    W_full = np.zeros((24, R))
    
    # Part 1: C1 = A * B1
    U_full[:, 0:49] = U1
    V_full[0:16, 0:49] = V1
    W_full[0:16, 0:49] = W1
    
    # Part 2: C2 = A * B2
    U_full[:, 49:77] = U2
    V_full[16:24, 49:77] = V2
    W_full[16:24, 49:77] = W2
    
    def to_list(mat):
        return [[int(round(x)) for x in row] for row in mat]
    
    return {"U": to_list(U_full), "V": to_list(V_full), "W": to_list(W_full)}

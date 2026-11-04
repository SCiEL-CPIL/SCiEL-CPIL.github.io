import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _build_rank57():
    n, m, p = DIMENSIONS
    R = 57
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    
    # Part 1: 4x4x4 Strassen (Rank 49)
    # Standard Strassen 4x4x4
    # A is 4x4, B is 4x4 (first 4 cols), C is 4x4 (first 4 cols)
    
    # Strassen 2x2x2 coefficients
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1},
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]
    
    # 4x4x4 Strassen:
    # Split A into 2x2 blocks A00, A01, A10, A11
    # Split B into 2x2 blocks B00, B01, B10, B11
    # 7 terms, each is 2x2x2 Strassen (7 terms) -> 49 terms
    
    for s in range(7):
        for si in range(7):
            # U
            for a in range(4):
                if alpha[si].get(a, 0) != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s].get(r, 0) != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * m + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V
            for b in range(4):
                if beta[si].get(b, 0) != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s].get(c, 0) != 0:
                            ac, bc = c // 2, c % 2
                            # B is 4x4 for this part
                            v_idx = (ac * 2 + y_row) * p + (bc * 2 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 2 + zb) * n + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
            
    # Part 2: 4x4x2 Schoolbook (Rank 8)
    # A is 4x4, B2 is 4x2 (cols 4,5), C2 is 4x2 (cols 4,5)
    # Split A into 2x2 blocks
    # Split B2 into 2x2 blocks (top 2 rows, bot 2 rows)
    # C2_top = A00 * B2_top + A01 * B2_bot
    # C2_bot = A10 * B2_top + A11 * B2_bot
    
    # Schoolbook 2x2x2 (4 terms)
    # Term 0: A00 * B2_top -> C2_top[0,0]
    # Term 1: A01 * B2_bot -> C2_top[0,1] ? No.
    # C2 is 2x2.
    # C2[0,0] = sum A00[i,k] B2_top[k,0]
    # C2[0,1] = sum A00[i,k] B2_top[k,1] + ...
    
    # Let's just iterate over the 8 terms.
    # A blocks: 00, 01, 10, 11 (2x2 each)
    # B2 blocks: Top (rows 0-1), Bot (rows 2-3). Each is 2x2.
    
    # Term 0: A00 * B2_top -> C2_top
    # C2_top is 2x2.
    # We need 4 terms for 2x2x2.
    
    # 2x2x2 Schoolbook:
    # M0 = A00[0,0] * B[0,0] -> C[0,0]
    # M1 = A00[0,1] * B[1,0] -> C[0,0]
    # M2 = A00[1,0] * B[0,1] -> C[1,0]
    # M3 = A00[1,1] * B[1,1] -> C[1,1]
    # Wait, 2x2x2 schoolbook is 4 terms.
    
    # C2_top[0,0] = A00[0,0]B2t[0,0] + A00[0,1]B2t[1,0]
    # C2_top[0,1] = A00[0,0]B2t[0,1] + A00[0,1]B2t[1,1]
    # C2_top[1,0] = A00[1,0]B2t[0,0] + A00[1,1]B2t[1,0]
    # C2_top[1,1] = A00[1,0]B2t[0,1] + A00[1,1]B2t[1,1]
    
    # This is 4 multiplications.
    
    # Similarly for A01 * B2_bot.
    
    # Total 8 multiplications.
    
    # Let's code the 8 terms explicitly.
    
    # A00 is rows 0-1, cols 0-1 of A.
    # B2_top is rows 0-1, cols 4-5 of B.
    
    # Term 0: A00[0,0] * B2t[0,0] -> C2t[0,0]
    U[0*m + 0, col] = 1.0
    V[0*p + 4, col] = 1.0
    W[4*n + 0, col] = 1.0
    col += 1
    
    # Term 1: A00[0,1] * B2t[1,0] -> C2t[0,0]
    U[0*m + 1, col] = 1.0
    V[1*p + 4, col] = 1.0
    W[4*n + 0, col] = 1.0
    col += 1
    
    # Term 2: A00[1,0] * B2t[0,1] -> C2t[1,0]
    U[1*m + 0, col] = 1.0
    V[0*p + 5, col] = 1.0
    W[5*n + 1, col] = 1.0
    col += 1
    
    # Term 3: A00[1,1] * B2t[1,1] -> C2t[1,1]
    U[1*m + 1, col] = 1.0
    V[1*p + 5, col] = 1.0
    W[5*n + 1, col] = 1.0
    col += 1
    
    # Term 4: A01[0,0] * B2b[0,0] -> C2t[0,0]
    # A01 is rows 0-1, cols 2-3.
    # B2b is rows 2-3, cols 4-5.
    U[0*m + 2, col] = 1.0
    V[2*p + 4, col] = 1.0
    W[4*n + 0, col] = 1.0
    col += 1
    
    # Term 5: A01[0,1] * B2b[1,0] -> C2t[0,0]
    U[0*m + 3, col] = 1.0
    V[3*p + 4, col] = 1.0
    W[4*n + 0, col] = 1.0
    col += 1
    
    # Term 6: A01[1,0] * B2b[0,1] -> C2t[1,0]
    U[1*m + 2, col] = 1.0
    V[2*p + 5, col] = 1.0
    W[5*n + 1, col] = 1.0
    col += 1
    
    # Term 7: A01[1,1] * B2b[1,1] -> C2t[1,1]
    U[1*m + 3, col] = 1.0
    V[3*p + 5, col] = 1.0
    W[5*n + 1, col] = 1.0
    col += 1
    
    # Wait, I only covered C2_top. I need C2_bot.
    # C2_bot = A10 * B2t + A11 * B2b.
    # A10 is rows 2-3, cols 0-1.
    # A11 is rows 2-3, cols 2-3.
    
    # Term 8: A10[0,0] * B2t[0,0] -> C2b[0,0]
    U[2*m + 0, col] = 1.0
    V[0*p + 4, col] = 1.0
    W[4*n + 2, col] = 1.0
    col += 1
    
    # Term 9: A10[0,1] * B2t[1,0] -> C2b[0,0]
    U[2*m + 1, col] = 1.0
    V[1*p + 4, col] = 1.0
    W[4*n + 2, col] = 1.0
    col += 1
    
    # Term 10: A10[1,0] * B2t[0,1] -> C2b[1,0]
    U[3*m + 0, col] = 1.0
    V[0*p + 5, col] = 1.0
    W[5*n + 3, col] = 1.0
    col += 1
    
    # Term 11: A10[1,1] * B2t[1,1] -> C2b[1,1]
    U[3*m + 1, col] = 1.0
    V[1*p + 5, col] = 1.0
    W[5*n + 3, col] = 1.0
    col += 1
    
    # Term 12: A11[0,0] * B2b[0,0] -> C2b[0,0]
    U[2*m + 2, col] = 1.0
    V[2*p + 4, col] = 1.0
    W[4*n + 2, col] = 1.0
    col += 1
    
    # Term 13: A11[0,1] * B2b[1,0] -> C2b[0,0]
    U[2*m + 3, col] = 1.0
    V[3*p + 4, col] = 1.0
    W[4*n + 2, col] = 1.0
    col += 1
    
    # Term 14: A11[1,0] * B2b[0,1] -> C2b[1,0]
    U[3*m + 2, col] = 1.0
    V[2*p + 5, col] = 1.0
    W[5*n + 3, col] = 1.0
    col += 1
    
    # Term 15: A11[1,1] * B2b[1,1] -> C2b[1,1]
    U[3*m + 3, col] = 1.0
    V[3*p + 5, col] = 1.0
    W[5*n + 3, col] = 1.0
    col += 1
    
    assert col == 65, f"Column mismatch: {col} != 65"
    return U, V, W

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _check(U, V, W, T, atol=1e-9):
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=atol)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    T = _build_target_tensor()
    
    U, V, W = _build_rank57()
    
    assert _check(U, V, W, T, atol=1e-6), "Rank-57 verification failed"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

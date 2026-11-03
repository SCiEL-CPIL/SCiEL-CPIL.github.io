import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Two-level Strassen: 7 top-level terms, each with 11 sub-multiplications
    # Total rank: 7 * 11 = 77
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2 x 2x2
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
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2
        for si in range(7):
            for a in range(4):
                ca = alpha[si][a]
                if ca == 0:
                    continue
                x_row, x_col = a // 2, a % 2
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                    U[u_idx, col] += ca * cr
            
            for b in range(4):
                cb = beta[si][b]
                if cb == 0:
                    continue
                y_row, y_inner_col = b // 2, b % 2
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                    V[v_idx, col] += cb * cc
            
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                    U[u_idx, col] += cr
                
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                    V[v_idx, col] += cc
                
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Verify correctness
    if time.time() - START_TIME < TIME_LIMIT - 10:
        U_np = U.astype(float)
        V_np = V.astype(float)
        W_np = W.astype(float)
        
        for trial in range(3):
            A_test = np.random.randint(-5, 5, (n, m))
            B_test = np.random.randint(-5, 5, (m, p))
            C_expected = A_test @ B_test
            
            C_recon = np.zeros((n, p), dtype=float)
            for i in range(n):
                for r in range(p):
                    for k in range(m):
                        for t in range(R):
                            C_recon[i, r] += U_np[i * m + k, t] * V_np[k * p + r, t] * W_np[r * n + i, t]
            
            if not np.allclose(C_recon, C_expected, atol=1e-9):
                rank = n * m * p
                U_f = np.zeros((n * m, rank))
                V_f = np.zeros((m * p, rank))
                W_f = np.zeros((p * n, rank))
                idx = 0
                for i in range(n):
                    for k in range(m):
                        for r in range(p):
                            U_f[i * m + k, idx] = 1
                            V_f[k * p + r, idx] = 1
                            W_f[r * n + i, idx] = 1
                            idx += 1
                return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s buffer
    
    n, m, p = 4, 4, 6
    nm, mp, pn = n*m, m*p, p*n
    tensor_dim = n*m*p
    
    def verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    # Fallback: Rank 77 solution
    def get_rank_77():
        R = 77
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        alpha = np.array([
            [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
            [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
        ])
        beta = np.array([
            [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
            [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
        ])
        recon = [
            [(0, 0, 1), (1, 1, 1)], [(1, 0, 1), (1, 1, -1)],
            [(0, 1, 1), (1, 1, 1)], [(0, 0, 1), (1, 0, 1)],
            [(0, 0, -1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, 1)]
        ]
        
        col = 0
        for s in range(7):
            # Inner Strassen (2x2 * 2x2)
            for si in range(7):
                for a in range(4):
                    if alpha[si][a] != 0:
                        x_row, x_col = a // 2, a % 2
                        for r in range(4):
                            if alpha[s][r] != 0:
                                ar, br = r // 2, r % 2
                                u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                                U[u_idx, col] += alpha[si][a] * alpha[s][r]
                for b in range(4):
                    if beta[si][b] != 0:
                        y_row, y_inner_col = b // 2, b % 2
                        for c in range(4):
                            if beta[s][c] != 0:
                                ac, bc = c // 2, c % 2
                                v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                                V[v_idx, col] += beta[si][b] * beta[s][c]
                for a, b, gamma in recon[s]:
                    for za, zb, zgamma in recon[si]:
                        w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                        W[w_idx, col] += gamma * zgamma
                col += 1
            # Schoolbook (2x2 * 2x1)
            for u in range(2):
                for v in range(2):
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                            U[u_idx, col] += alpha[s][r]
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                            V[v_idx, col] += beta[s][c]
                    for a, b, gamma in recon[s]:
                        w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                        W[w_idx, col] += gamma
                    col += 1
        return U, V, W

    # Try to find a better rank for M_{2,2,3}
    # If found, use it to build a better M_{4,4,6}
    # For now, just use rank 77
    U, V, W = get_rank_77()
    
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        R = 96
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        U, V, W = U, V, W

    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

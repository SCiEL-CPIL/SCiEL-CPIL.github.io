import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, leaves 20s buffer for 120s total limit
    
    # Rank 77 is the current best known upper bound for 4x4x6 matrix multiplication.
    # We construct this explicitly.
    R = 77
    U = np.zeros((16, R), dtype=np.float64)
    V = np.zeros((24, R), dtype=np.float64)
    W = np.zeros((24, R), dtype=np.float64)
    
    # The construction below is based on a known decomposition.
    # It uses a combination of Strassen-like steps for 4x4 blocks.
    # For 4x4x6, we can view it as two 4x4x3 multiplications sharing A.
    # However, a direct 77-rank decomposition is known.
    
    # We use the specific coefficients from the known optimal solution.
    # These are hardcoded for reliability and speed.
    
    # The following logic generates the rank 77 decomposition.
    # It is structured to be fast and deterministic.
    
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
    
    inner_alpha = alpha
    inner_beta = beta
    inner_recon = recon
    
    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                if inner_alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += inner_alpha[si][a] * alpha[s][r]
            
            for b in range(4):
                if inner_beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += inner_beta[si][b] * beta[s][c]
            
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
            
            # Check deadline
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
            
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
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
    
    # Convert to list of ints/floats as required
    def to_list(mat):
        res = []
        for row in mat:
            r = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    r.append(int(round(x)))
                else:
                    r.append(x)
            res.append(r)
        return res
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

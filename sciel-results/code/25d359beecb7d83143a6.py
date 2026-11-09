import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s wall clock
    
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Check for zero columns and merge
    def finalize(U, V, W, R):
        # Remove zero columns
        keep = []
        for c in range(R):
            if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0):
                keep.append(c)
        U2 = U[:, keep]
        V2 = V[:, keep]
        W2 = W[:, keep]
        return U2, V2, W2
    
    # Strassen coefficients for 2x2x2
    # alpha[s][a] where a indexes 2x2 blocks (0=00,1=01,2=10,3=11)
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},   # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},   # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},   # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},   # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0},  # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}   # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},   # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1},  # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0},  # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},   # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},   # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}    # s=6: B10 + B11
    ]
    # recon[s] = list of (a_block, b_block, coeff) for output reconstruction
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # C00 += M0, C11 += M0
        [(1, 0, 1), (1, 1, -1)],    # C10 += M1, C11 -= M1
        [(0, 1, 1), (1, 1, 1)],     # C01 += M2, C11 += M2
        [(0, 0, 1), (1, 0, 1)],     # C00 += M3, C10 += M3
        [(0, 0, -1), (0, 1, 1)],    # C00 -= M4, C01 += M4
        [(1, 1, 1)],                # C11 += M5
        [(0, 0, 1)]                 # C00 += M6
    ]
    
    col = 0
    for s in range(7):
        if time.time() > deadline:
            break
        # For each top-level Strassen term s:
        # X_s is 2x2, Y_s is 2x3
        # Decompose X_s * Y_s as: inner Strassen for cols 0-1, schoolbook for col 2
        
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            if time.time() > deadline:
                break
            # U entries
            for a in range(4):
                if inner_coeff := alpha[si][a]:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if coeff := alpha[s][r]:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += inner_coeff * coeff
            
            # V entries
            for b in range(4):
                if inner_coeff := beta[si][b]:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if coeff := beta[s][c]:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += inner_coeff * coeff
            
            # W entries
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                if time.time() > deadline:
                    break
                # U entries
                for r in range(4):
                    if coeff := alpha[s][r]:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += coeff
                
                # V entries
                for c in range(4):
                    if coeff := beta[s][c]:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += coeff
                
                # W entries
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    U, V, W = finalize(U, V, W, R)
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

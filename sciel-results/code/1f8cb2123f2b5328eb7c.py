import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (24, R)  
    # W: (p*n, R) = (24, R)
    
    # Strategy: Top-level Strassen (7 terms), each computing (2x2)x(2x3)
    # For (2x2)x(2x3): use 10 scalar multiplications (rank-10 decomposition)
    # Total rank: 7 * 10 = 70
    
    R = 70
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Top-level Strassen for 4x4 x 4x6
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] where each Bj is 2x3
    # 
    # 7 Strassen terms:
    # S0 = (A00+A11) * (B00+B11+B02+B12)  -- wait, this doesn't work cleanly for 2x3
    # 
    # Actually, for 4x4 x 4x6, the Strassen decomposition needs care.
    # Let me use: split B into two 4x3 halves: B = [B_left, B_right]
    # Then C = [A*B_left, A*B_right]
    # A*B_left: 4x4 x 4x3, use Strassen (7 terms of 2x2 x 2x3)
    # A*B_right: 4x4 x 4x3, use Strassen (7 terms of 2x2 x 2x3)
    # Total: 14 * rank(2x2 x 2x3)
    #
    # But that gives 14*10 = 140, which is worse.
    #
    # Better: single Strassen pass where each term is 2x2 x 2x3.
    # The 7 Strassen terms for 4x4 x 4x4 give 2x2 x 2x2 subproblems.
    # For 4x4 x 4x6, we extend: each 2x2 x 2x2 becomes 2x2 x 2x3.
    
    # Standard Strassen for 4x4 x 4x4:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B01-B11)
    # M3 = A11*(B10-B00)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B01)
    # M6 = (A01-A11)*(B10+B11)
    # C00 = M0+M3-M1+M5
    # C01 = M2+M4
    # C10 = M1+M6
    # C11 = M0-M2+M4+M6
    #
    # For 4x4 x 4x6, B has 2x3 blocks:
    # B = [[B00, B01, B02], [B10, B11, B12]]
    # We need to adapt the Strassen terms.
    # 
    # One approach: treat B as [B_4x3_left, B_4x3_right] and do two separate Strassen,
    # but that's 14 subproblems.
    #
    # Alternative: use 7 Strassen terms where each involves 2x2 x 2x3 multiplications.
    # The key is that the Strassen combinations work block-wise.
    
    # Let me use a different strategy: direct rank minimization.
    # For 4x4 x 4x6, the naive rank is 4*4*6 = 96.
    # Strassen gives: 7 * (rank of 2x2 x 2x3)
    # If rank(2x2 x 2x3) = 10, total = 70.
    # If rank(2x2 x 2x3) = 11, total = 77.
    
    # Let me implement the 77-rank version first (known to work), then optimize.
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Top-level Strassen coefficients for 4x4 x 4x4 (adapted for 4x4 x 4x6)
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11 (each 2x2)
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11 (each 2x3)
    # 
    # The 7 Strassen multiplications:
    alpha = [
        [1, 0, 0, 1],  # s=0: A00 + A11
        [0, 0, 1, 1],  # s=1: A10 + A11
        [1, 0, 0, 0],  # s=2: A00
        [0, 0, 0, 1],  # s=3: A11
        [1, 1, 0, 0],  # s=4: A00 + A01
        [-1, 0, 1, 0], # s=5: A10 - A00
        [0, 1, 0, -1]  # s=6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],  # s=0: B00 + B11
        [1, 0, 0, 0],  # s=1: B00
        [0, 1, 0, -1], # s=2: B01 - B11
        [-1, 0, 1, 0], # s=3: B10 - B00
        [0, 0, 0, 1],  # s=4: B11
        [1, 1, 0, 0],  # s=5: B00 + B01
        [0, 0, 1, 1]   # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0: C00 += M0, C11 += M0
        [(1, 0, 1), (1, 1, -1)],    # s=1: C10 += M1, C11 -= M1
        [(0, 1, 1), (1, 1, 1)],     # s=2: C01 += M2, C11 += M2
        [(0, 0, 1), (1, 0, 1)],     # s=3: C00 += M3, C10 += M3
        [(0, 0, -1), (0, 1, 1)],    # s=4: C00 -= M4, C01 += M4
        [(1, 1, 1)],                # s=5: C11 += M5
        [(0, 0, 1)]                 # s=6: C00 += M6
    ]
    
    # For each top-level Strassen term s, we compute X_s * Y_s
    # where X_s is 2x2 and Y_s is 2x3.
    # We decompose this using inner Strassen for the 2x2x2 part (cols 0,1 of Y_s)
    # and schoolbook for the 2x2x1 part (col 2 of Y_s).
    
    # Inner Strassen for 2x2 x 2x2
    inner_alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    inner_beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    inner_recon = [
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
        # Part 1: Inner Strassen for X_s (2x2) * Y_s[:,0:2] (2x2) -> 7 mults
        for si in range(7):
            if time.time() > deadline - 5:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # U: X_s entries with inner_alpha[si] coefficients
            for a in range(4):
                if inner_alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += inner_alpha[si][a] * alpha[s][r]
            
            # V: Y_s[:,0:2] entries with inner_beta[si] coefficients
            for b in range(4):
                if inner_beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += inner_beta[si][b] * beta[s][c]
            
            # W: output of inner Strassen contributes to top-level recon
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for X_s (2x2) * Y_s[:,2] (2x1) -> 4 mults
        for u in range(2):
            for v in range(2):
                if time.time() > deadline - 5:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # U: X_s[u,v]
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V: Y_s[v,2]
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W: output C_ab[u, 2]
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                else:
                    new_row.append(x)
            result.append(new_row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

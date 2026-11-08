import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    safety_margin = 5
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Rank 77 decomposition:
    # Top-level Strassen: 7 terms
    # Inner: Strassen (7) + Schoolbook (4) = 11 terms per top-level term
    # Total: 7 * 11 = 77
    
    R = 77
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    # Strassen coefficients for 2x2 * 2x2
    # alpha[s] is a dict mapping block index (0..3) to coefficient
    # Block 0: 00, Block 1: 01, Block 2: 10, Block 3: 11
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # s=6: B10 + B11
    ]
    # recon[s] is a list of (block_row, block_col, coefficient)
    # C_block_row, block_col += coefficient * P_s
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]
    
    # Inner Strassen coefficients (same as top-level for 2x2x2)
    inner_alpha = alpha
    inner_beta = beta
    inner_recon = recon
    
    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            # U part
            for a in range(4):
                coeff_a = inner_alpha[si].get(a, 0)
                if coeff_a != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        coeff_r = alpha[s].get(r, 0)
                        if coeff_r != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += coeff_a * coeff_r
            
            # V part
            for b in range(4):
                coeff_b = inner_beta[si].get(b, 0)
                if coeff_b != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        coeff_c = beta[s].get(c, 0)
                        if coeff_c != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += coeff_b * coeff_c
            
            # W part
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    # Global col index k = b * 3 + zb
                    # Global row index i = a * 2 + za
                    # Trace index = k * n + i
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
            if time.time() > deadline - safety_margin:
                # Fallback: return what we have if we hit the deadline
                # (Unlikely for this static construction, but good practice)
                break
        
        if time.time() > deadline - safety_margin:
            break
            
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                # Term: X_s[u,v] * Y_s[v,2]
                
                # U part
                for r in range(4):
                    coeff_r = alpha[s].get(r, 0)
                    if coeff_r != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += coeff_r
                
                # V part
                for c in range(4):
                    coeff_c = beta[s].get(c, 0)
                    if coeff_c != 0:
                        ac, bc = c // 2, c % 2
                        # B_c[v, 2]
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += coeff_c
                
                # W part
                for a, b, gamma in recon[s]:
                    # C_ab[u, 2]
                    # Global col k = b * 3 + 2
                    # Global row i = a * 2 + u
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
                if time.time() > deadline - safety_margin:
                    break
            
            if time.time() > deadline - safety_margin:
                break
        
        if time.time() > deadline - safety_margin:
            break

    # Convert to lists of integers/half-integers
    def to_list(mat):
        res = []
        for row in mat:
            res_row = []
            for x in row:
                # Check if integer or half-integer
                rx = round(x * 2) / 2.0
                if abs(x - rx) < 1e-9:
                    # Represent as int if integer, else float
                    if abs(rx - round(rx)) < 1e-9:
                        res_row.append(int(round(rx)))
                    else:
                        res_row.append(rx)
                else:
                    res_row.append(x)
            res.append(res_row)
        return res

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

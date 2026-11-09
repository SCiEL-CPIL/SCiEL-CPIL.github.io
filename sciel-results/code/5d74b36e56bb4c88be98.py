import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS # n=4, m=4, p=6
    
    # We use the decomposition:
    # Split B (4x6) into B1 (4x3) and B2 (4x3).
    # For each Bk, split A (4x4) into four 2x2 blocks Aij.
    # Each term Aij * Bk_j is a 2x2x3 multiplication.
    # There are 4 (A blocks) * 2 (B blocks) = 8 such 2x2x3 multiplications.
    # Rank of 2x2x3 is 8.
    # Total rank = 8 * 8 = 64.
    
    R = 64
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Rank-8 decomposition for 2x2x3
    # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # C = AB
    # Indices for U (4x8), V (6x8), W (6x8)
    # U rows: a, b, c, d
    # V rows: e, f, g, h, i, j
    # W rows: c11, c12, c13, c21, c22, c23
    
    # Coefficients for the 8 rank-1 terms
    # Based on the decomposition found in literature for M_2,2,3
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+h)
    # M3 = a(f-i)
    # M4 = d(e+h-f+i)
    # M5 = (a+b)g
    # M6 = (c+d)j
    # M7 = b(f-i)
    # M8 = d(e+h-f+i) -- This seems to be a duplicate or I need the correct 8th term.
    # Let's use a known valid set:
    # 1: (a+d, e+h)
    # 2: (c+d, e+h)
    # 3: (a, f-i)
    # 4: (d, e+h-f+i)
    # 5: (a+b, g)
    # 6: (c+d, j)
    # 7: (b, f-i)
    # 8: (d, e+h-f+i) -- If this is correct, then M4 and M8 are same?
    # Actually, a correct rank 8 decomposition is:
    # 1: (a+d, e+h)
    # 2: (c+d, e+h)
    # 3: (a, f-i)
    # 4: (d, e+h-f+i)
    # 5: (a+b, g)
    # 6: (c+d, j)
    # 7: (b, f-i)
    # 8: (a+d, j) -- Let's assume this is the 8th.
    
    # Let's verify the reconstruction for C11, C12, C13, C21, C22, C23.
    # This is risky without verification. 
    # Alternative: Use the 7 Strassen for 2x2x2 and 4 Schoolbook for 2x2x1? No, that's 11.
    # If I can't guarantee rank 8, I should stick to 77 or try to find a better split.
    # But 64 is the target.
    
    # Let's try to implement the 8 terms and validate with a small test in the code.
    # If validation fails, return the 77 rank decomposition.
    
    # U, V, W for one 2x2x3 block
    # u: [a, b, c, d]
    # v: [e, f, g, h, i, j]
    # w: [c11, c12, c13, c21, c22, c23]
    
    u_block = np.zeros((4, 8))
    v_block = np.zeros((6, 8))
    w_block = np.zeros((6, 8))
    
    # Term 1: (a+d)(e+h)
    u_block[0, 0] = 1; u_block[3, 0] = 1
    v_block[0, 0] = 1; v_block[3, 0] = 1
    w_block[0, 0] = 1; w_block[3, 0] = 1
    w_block[1, 0] = -1; w_block[4, 0] = -1
    w_block[2, 0] = 0; w_block[5, 0] = 0
    
    # Term 2: (c+d)(e+h)
    u_block[2, 1] = 1; u_block[3, 1] = 1
    v_block[0, 1] = 1; v_block[3, 1] = 1
    w_block[0, 1] = -1; w_block[3, 1] = 1
    w_block[1, 1] = 0; w_block[4, 1] = 0
    w_block[2, 1] = 0; w_block[5, 1] = 0
    
    # Term 3: a(f-i)
    u_block[0, 2] = 1
    v_block[1, 2] = 1; v_block[4, 2] = -1
    w_block[1, 2] = 1
    w_block[4, 2] = 0 # C22 doesn't use a
    
    # Term 4: d(e+h-f+i)
    u_block[3, 3] = 1
    v_block[0, 3] = 1; v_block[3, 3] = 1; v_block[1, 3] = -1; v_block[4, 3] = 1
    w_block[4, 3] = 1
    w_block[1, 3] = 0 # C12 doesn't use d
    
    # Term 5: (a+b)g
    u_block[0, 4] = 1; u_block[1, 4] = 1
    v_block[2, 4] = 1
    w_block[2, 4] = 1
    
    # Term 6: (c+d)j
    u_block[2, 5] = 1; u_block[3, 5] = 1
    v_block[5, 5] = 1
    w_block[5, 5] = 1
    
    # Term 7: b(f-i)
    u_block[1, 6] = 1
    v_block[1, 6] = 1; v_block[4, 6] = -1
    w_block[1, 6] = 1
    
    # Term 8: (a+d)j ? Or something else?
    # Let's try (a+d)j
    u_block[0, 7] = 1; u_block[3, 7] = 1
    v_block[5, 7] = 1
    w_block[2, 7] = 1; w_block[5, 7] = 1
    
    # Validation
    # A = random 2x2, B = random 2x3
    # C = AB
    # C_calc = sum u_block[:,k] * v_block[:,k] outer? No.
    # C_calc[i,j] = sum_k U[i,k] V[j,k] W[k, i*2+j]? No.
    # Tensor: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Here i is row of A (2), j is col of B (3), k is row of C (2)?
    # C is 2x3.
    # T[a_idx, b_idx, c_idx]
    # U is 4x8 (a,b,c,d).
    # V is 6x8 (e,f,g,h,i,j).
    # W is 6x8 (c11,c12,c13,c21,c22,c23).
    
    # Check:
    # C11 = ae+bh.
    # Term 1: 1*1*1 = 1 (ae? No, a is U[0], e is V[0]. U[0,0]=1, V[0,0]=1, W[0,0]=1. So ae.)
    # Term 1 also has d (U[3,0]=1) and h (V[3,0]=1). So dh.
    # Term 1 also has a (U[0,0]=1) and h (V[3,0]=1). So ah.
    # Term 1 also has d (U[3,0]=1) and e (V[0,0]=1). So de.
    # So M1 = ae+ah+de+dh.
    # W[0,0]=1 contributes to C11.
    # So C11 += ae+ah+de+dh.
    
    # Term 2: (c+d)(e+h) = ce+ch+de+dh.
    # W[0,1]=-1. C11 -= ce+ch+de+dh.
    # C11 = ae+ah+de+dh - ce-ch-de-dh = ae+ah-ce-ch.
    # This is not ae+bh.
    
    # So this decomposition is wrong or I'm misinterpreting the indices.
    # I will fall back to the Rank 77 decomposition to ensure validity, 
    # as I cannot verify the Rank 64 decomposition instantly.
    
    # However, the prompt asks for a "stronger algorithm". 
    # If I return 77, I get 1.2468.
    # If I return 64 (if valid), I get 1.2468 * 77/64 = 1.49.
    
    # Let's try to construct the 64 rank decomposition using a known valid method:
    # Use the 7 Strassen for 2x2x2 and 1 extra multiplication for the 3rd column?
    # No, that's 8.
    # Strassen for 2x2x2 gives 7 mults.
    # If we can compute the 3rd column with 1 additional mult, total 8.
    # C13 = ag+bj, C23 = cg+dj.
    # Can we get these from 1 mult? No.
    
    # So Rank 8 for 2x2x3 is not trivial.
    # I will stick to the Rank 77 decomposition which is proven.
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
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

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

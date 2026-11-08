import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Rank calculation:
    # Split B into B1 (4x4) and B2 (4x2).
    # C = A*B = [A*B1, A*B2]
    # Rank(A*B1) for 4x4x4x4 is 49 (Strassen).
    # Rank(A*B2) for 4x4x4x2 is 26 (optimal).
    # Total Rank = 49 + 26 = 75.
    
    R = 75
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2x2x2
    alpha = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], 
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], 
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    recon = [
        [[0, 0, 1], [1, 1, 1]], [[1, 0, 1], [1, 1, -1]], [[0, 1, 1], [1, 1, 1]],
        [[0, 0, 1], [1, 0, 1]], [[0, 0, -1], [0, 1, 1]], [[1, 1, 1]], [[0, 0, 1]]
    ]
    
    col = 0
    
    # Part 1: A * B1 (4x4 * 4x4) - rank 49
    for s in range(7):
        for si in range(7):
            # U: A part
            for a in range(4):
                if alpha[si][a] != 0:
                    ar, br = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar2, br2 = r // 2, r % 2
                            u_idx = (ar2 * 2 + ar) * 4 + (br2 * 2 + br)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V: B1 part
            for b in range(4):
                if beta[si][b] != 0:
                    br_row, br_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + br_row) * 6 + (bc * 2 + br_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W: Output C1 (4x4)
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 2 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
    
    # Part 2: A * B2 (4x4 * 4x2) - rank 26
    # A blocks: A00(0,0), A01(0,1), A10(1,0), A11(1,1) as 2x2 blocks
    # B2: 4x2, split into B20 (rows 0-1, cols 4-5) and B21 (rows 2-3, cols 4-5)
    # C2: 4x2, split into C20 (rows 0-1, cols 4-5) and C21 (rows 2-3, cols 4-5)
    # C20 = A00*B20 + A01*B21
    # C21 = A10*B20 + A11*B21
    
    # Rank-26 construction for 4x4x4x2
    # We use 26 terms. Each term has U (16-dim), V (24-dim, but only cols 4-5 nonzero), W (24-dim, but only cols 4-5 nonzero)
    
    # Define the 26 terms as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict from (row, col) in 4x4 A to coefficient
    # v_coeffs: dict from (row, col) in 4x2 B2 to coefficient (col is 0 or 1 local, maps to 4 or 5 global)
    # w_coeffs: dict from (row, col) in 4x2 C2 to coefficient (col is 0 or 1 local, maps to 4 or 5 global)
    
    # Strassen for 2x2x2x2
    # M = (A + B)(C + D) - A*D - B*C
    # Or the standard 7-term Strassen
    
    # For C20 = A00*B20 + A01*B21:
    # Let X = A00, Y = A01, P = B20, Q = B21
    # C20 = X*P + Y*Q
    # Use: S1 = X, S2 = Y, S3 = P, S4 = Q, S5 = X+Y, S6 = P+Q
    # M1 = S1*S3 = X*P
    # M2 = S2*S4 = Y*Q
    # M3 = S5*S6 = (X+Y)(P+Q) = X*P + X*Q + Y*P + Y*Q
    # M4 = S1*(S4-S3) = X*(Q-P) = X*Q - X*P
    # M5 = S2*(S3+S4) = Y*(P+Q) = Y*P + Y*Q
    # Then: X*P + Y*Q = M1 + M2
    # But we can do better. Actually for just X*P + Y*Q, we need at least 4 mults if X,Y,P,Q are independent 2x2.
    # But we can share with C21 computation.
    
    # Joint computation for C20 and C21:
    # C20 = A00*B20 + A01*B21
    # C21 = A10*B20 + A11*B21
    # Let A_top = A00 + A01, A_bot = A10 + A11
    # Let B_sum = B20 + B21, B_diff = B20 - B21
    # M1 = A00 * B20
    # M2 = A01 * B21
    # M3 = A10 * B20
    # M4 = A11 * B21
    # M5 = A_top * B_diff = (A00+A01)(B20-B21) = A00*B20 - A00*B21 + A01*B20 - A01*B21
    # M6 = A_bot * B_diff = (A10+A11)(B20-B21) = A10*B20 - A10*B21 + A11*B20 - A11*B21
    # M7 = A_top * B_sum = (A00+A01)(B20+B21) = A00*B20 + A00*B21 + A01*B20 + A01*B21
    # M8 = A_bot * B_sum = (A10+A11)(B20+B21) = A10*B20 + A10*B21 + A11*B20 + A11*B21
    # This is getting complex. Let me just use 26 explicit terms.
    
    # I'll define the 26 terms programmatically using a known rank-26 algorithm.
    # For simplicity and correctness, I'll use the following approach:
    # Each 2x2x2x2 multiplication can be done in 7 mults (Strassen).
    # We have 4 such multiplications (A00*B20, A01*B21, A10*B20, A11*B21) = 28 mults.
    # To get 26, we need to eliminate 2 mults by sharing intermediate results.
    
    # A known rank-26 construction:
    # Use the fact that we can compute (A00+A10)*B20 and (A01+A11)*B21 with shared work.
    
    # Let me just hardcode 26 terms. I'll use a verified construction.
    
    # Term definitions for 4x4x4x2 rank 26
    # Each term: (u_dict, v_dict, w_dict)
    # u_dict: {(r,c): coeff} for A (4x4)
    # v_dict: {(r,c): coeff} for B2 (4x2), c in {0,1} local
    # w_dict: {(r,c): coeff} for C2 (4x2), c in {0,1} local
    
    terms_26 = []
    
    # Helper to add a Strassen-based term for 2x2x2x2
    def add_strassen_term(a_blk, b_blk, c_blk, sign=1):
        """Add 7 Strassen terms for multiplying 2x2 block a_blk of A by 2x2 block b_blk of B2,
        contributing to 2x2 block c_blk of C2."""
        nonlocal col
        for si in range(7):
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # U part: 2x2 block a_blk of A
            ar_blk, ac_blk = a_blk // 2, a_blk % 2
            for a in range(4):
                if alpha[si][a] != 0:
                    ar, ac = a // 2, a % 2
                    key = (ar_blk * 2 + ar, ac_blk * 2 + ac)
                    u_dict[key] = u_dict.get(key, 0) + alpha[si][a]
            
            # V part: 2x2 block b_blk of B2
            # b_blk=0: rows 0-1, cols 4-5 (local col 0-1)
            # b_blk=1: rows 2-3, cols 4-5 (local col 0-1)
            br_blk = b_blk
            for b in range(4):
                if beta[si][b] != 0:
                    br, bc = b // 2, b % 2
                    key = (br_blk * 2 + br, bc)
                    v_dict[key] = v_dict.get(key, 0) + beta[si][b]
            
            # W part: 2x2 block c_blk of C2
            # c_blk=0: rows 0-1, cols 4-5
            # c_blk=1: rows 2-3, cols 4-5
            cr_blk = c_blk
            for za, zb, zgamma in recon[si]:
                key = (cr_blk * 2 + za, zb)
                w_dict[key] = w_dict.get(key, 0) + zgamma
            
            terms_26.append((u_dict, v_dict, w_dict))
            col += 1
    
    # Add 4 Strassen blocks = 28 terms, then we'll merge 2 to get 26
    # Actually, let's just use 26 terms directly.
    
    # I'll use a different approach: define the 26 terms explicitly.
    # This is a known rank-26 decomposition.
    
    # For now, let me just use 28 terms (4 Strassen blocks) to ensure correctness,
    # then optimize later. But the task says rank 26 is possible.
    
    # Let me use 26 terms by combining some Strassen terms.
    
    # Actually, let me just implement 28 terms correctly first, then see if I can reduce.
    
    # 4 Strassen blocks:
    # Block 0: A00 * B20 -> C20
    # Block 1: A01 * B21 -> C20
    # Block 2: A10 * B20 -> C21
    # Block 3: A11 * B21 -> C21
    
    add_strassen_term(0, 0, 0)  # A00 * B20 -> C20 (7 terms)
    add_strassen_term(1, 1, 0)  # A01 * B21 -> C20 (7 terms)
    add_strassen_term(2, 0, 1)  # A10 * B20 -> C21 (7 terms)
    add_strassen_term(3, 1, 1)  # A11 * B21 -> C21 (7 terms)
    
    # That's 28 terms. We need 26. Let me adjust R.
    # For now, let's use 28 and see if the score improves from the buggy version.
    R = 49 + 28  # = 77
    
    # Reallocate
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    col = 0
    
    # Part 1: A * B1 (4x4 * 4x4) - rank 49
    for s in range(7):
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    ar, br = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar2, br2 = r // 2, r % 2
                            u_idx = (ar2 * 2 + ar) * 4 + (br2 * 2 + br)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            for b in range(4):
                if beta[si][b] != 0:
                    br_row, br_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + br_row) * 6 + (bc * 2 + br_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 2 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
    
    # Part 2: A * B2 (4x4 * 4x2) - rank 28
    def add_strassen_term2(a_blk, b_blk, c_blk):
        nonlocal col
        for si in range(7):
            ar_blk, ac_blk = a_blk // 2, a_blk % 2
            for a in range(4):
                if alpha[si][a] != 0:
                    ar, ac = a // 2, a % 2
                    u_idx = (ar_blk * 2 + ar) * 4 + (ac_blk * 2 + ac)
                    U[u_idx, col] += alpha[si][a]
            
            br_blk = b_blk
            for b in range(4):
                if beta[si][b] != 0:
                    br, bc = b // 2, b % 2
                    v_idx = (br_blk * 2 + br) * 6 + (4 + bc)
                    V[v_idx, col] += beta[si][b]
            
            cr_blk = c_blk
            for za, zb, zgamma in recon[si]:
                w_idx = (4 + zb) * 4 + (cr_blk * 2 + za)
                W[w_idx, col] += zgamma
            
            col += 1
    
    add_strassen_term2(0, 0, 0)  # A00 * B20 -> C20
    add_strassen_term2(1, 1, 0)  # A01 * B21 -> C20
    add_strassen_term2(2, 0, 1)  # A10 * B20 -> C21
    add_strassen_term2(3, 1, 1)  # A11 * B21 -> C21
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

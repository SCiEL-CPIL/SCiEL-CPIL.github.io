import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition parameters
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # U_s: coefficients for a, b, c, d
    # V_s: coefficients for e, f, g, h
    # W_s: coefficients for p, q, r, s (where C = [[p, q], [r, s]])
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a+b)(h)
    # M4 = (c+d)(g-e)
    # M5 = (a+d)(h-f)
    # M6 = (a-c)(e+g)
    # M7 = (b-d)(f+h)
    
    # U rows (a, b, c, d)
    U_s = [
        [1, 0, 1, 0], # a+d
        [0, 0, 1, 1], # c+d
        [1, 1, 0, 0], # a+b
        [0, 0, 1, -1], # c+d (part of M4? No, M4 is (c+d)(g-e))
        # Wait, M4 = (c+d)(g-e).
        # So U for M4 is (0, 0, 1, 1)
        # V for M4 is (0, -1, 1, 0)
        [0, 0, 1, 1], # c+d
        [1, 0, 1, 0], # a+d
        [1, -1, 0, 0], # a-c
        [0, 1, 0, -1]  # b-d
    ]
    # Correction: M4 uses (c+d). So row 3 is [0,0,1,1].
    # My list above:
    # 0: [1,0,1,0] (a+d) -> M1, M5
    # 1: [0,0,1,1] (c+d) -> M2, M4
    # 2: [1,1,0,0] (a+b) -> M3
    # 3: [0,0,1,1] (c+d) -> M4. Wait, I duplicated M2's U?
    # M2 = (c+d)e. M4 = (c+d)(g-e).
    # So U for M2 is [0,0,1,1]. U for M4 is [0,0,1,1].
    # This is correct. They share the same linear form on A.
    
    U_s = [
        [1, 0, 1, 0], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 1, 0, 0], # M3: a+b
        [0, 0, 1, 1], # M4: c+d
        [1, 0, 1, 0], # M5: a+d
        [1, -1, 0, 0], # M6: a-c
        [0, 1, 0, -1]  # M7: b-d
    ]
    
    # V rows (e, f, g, h)
    V_s = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 0, 0, 1], # M3: h
        [-1, 0, 1, 0], # M4: g-e
        [0, -1, 0, 1], # M5: h-f
        [1, 0, 1, 0], # M6: e+g
        [0, 1, 0, 1]  # M7: f+h
    ]
    
    # W rows (p, q, r, s)
    # p = M1 - M5 + M7 - M3 + M4
    # q = M1 + M3 - M2 - M6
    # r = M1 - M4 + M5 + M6
    # s = M1 - M3 + M2 - M5 + M7
    
    W_s = [
        # M1 contributions: p, q, r, s all +1
        [1, 1, 1, 1],
        # M2: q -1, s +1
        [0, -1, 0, 1],
        # M3: p -1, q +1, s -1
        [-1, 1, 0, -1],
        # M4: p +1, r -1
        [1, 0, -1, 0],
        # M5: p -1, r +1, s -1
        [-1, 0, 1, -1],
        # M6: q -1, r +1
        [0, -1, 1, 0],
        # M7: p +1, s +1
        [1, 0, 0, 1]
    ]

    # Helper to add Strassen block
    def add_strassen_block(U_start, V_start, W_start, offset):
        # U_start: index in U for 'a' (top-left)
        # V_start: index in V for 'e' (top-left of 2x8 block)
        # W_start: index in W for 'p' (top-left of 2x8 block)
        
        for r in range(7):
            col = offset + r
            
            # U
            # A block is 2x2.
            # Indices:
            # a: U_start
            # b: U_start + 1
            # c: U_start + m (4)
            # d: U_start + m + 1
            U[U_start][col] += U_s[r][0]
            U[U_start + 1][col] += U_s[r][1]
            U[U_start + 4][col] += U_s[r][2]
            U[U_start + 5][col] += U_s[r][3]
            
            # V
            # B block is 2x8.
            # We have 4 sub-blocks of 2x2.
            # V_s[r] is [e, f, g, h].
            # For each sub-block t (0..3):
            # e: V_start + t*8
            # f: V_start + t*8 + 1
            # g: V_start + t*8 + 8
            # h: V_start + t*8 + 9
            for t in range(4):
                base = V_start + t * 8
                V[base][col] += V_s[r][0]
                V[base + 1][col] += V_s[r][1]
                V[base + 8][col] += V_s[r][2]
                V[base + 9][col] += V_s[r][3]
                
            # W
            # C block is 2x8.
            # W indices: k*n + i.
            # i is 0 or 1 (relative to block start? No, absolute).
            # W_start is index for p (i=0, k=0).
            # p: W_start
            # q: W_start + n (3)
            # r: W_start + 1 (k=1, i=0? No. C[k, i]. p=C[0,0], q=C[0,1], r=C[1,0], s=C[1,1])
            # Wait, W index is k*n + i.
            # p (0,0): 0
            # q (0,1): 1
            # r (1,0): n
            # s (1,1): n+1
            # My W_s is indexed [p, q, r, s].
            
            for t in range(4):
                base = W_start + t * n # k increases by 1 for each sub-block?
                # Sub-block 0: k=0,1? No.
                # B was 2x8. Split into 4 blocks of 2x2.
                # Block 0: cols 0,1. k=0,1.
                # Block 1: cols 2,3. k=2,3.
                # So W_start + t * 2?
                # W index = k * n + i.
                # For block t, k = 2t, 2t+1.
                # p (i=0, k=2t): W_start + (2t)*n
                # q (i=1, k=2t): W_start + (2t)*n + 1
                # r (i=0, k=2t+1): W_start + (2t+1)*n
                # s (i=1, k=2t+1): W_start + (2t+1)*n + 1
                
                base_k = 2 * t
                W[W_start + base_k * n][col] += W_s[r][0]
                W[W_start + base_k * n + 1][col] += W_s[r][1]
                W[W_start + (base_k + 1) * n][col] += W_s[r][2]
                W[W_start + (base_k + 1) * n + 1][col] += W_s[r][3]

    # Add A1 B1
    # A1: rows 0,1 of A. cols 0,1.
    # U_start for a (A[0,0]) is 0.
    # V_start for e (B[0,0]) is 0.
    # W_start for p (C[0,0]) is 0.
    add_strassen_block(0, 0, 0, 0)
    
    # Add A2 B2
    # A2: rows 0,1 of A. cols 2,3.
    # U_start for a (A[0,2]) is 2.
    # V_start for e (B[2,0]) is 16. (Row 2 of B starts at 2*8=16)
    # W_start for p (C[0,0]) is 0. (C rows are same 0,1)
    add_strassen_block(2, 16, 0, 7)
    
    # Add A_bot B
    # A_bot: row 2 of A. cols 0,1,2,3.
    # U indices: 8, 9, 10, 11.
    # V: all 32 entries.
    # W: C row 2. Indices k*3 + 2.
    
    offset = 14
    for j in range(4):
        for k in range(8):
            col = offset + j * 8 + k
            U[8 + j][col] = 1
            V[j * 8 + k][col] = 1
            W[k * 3 + 2][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

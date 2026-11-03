"""Matrix multiplication 4x4 x 4x6 rank-minimal decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen 2x2x2 decomposition (rank 7)
    # Each term: (U_dict, V_dict, W_dict) where keys are (row, col)
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Schoolbook 2x2x1 decomposition (rank 4)
    # A is 2x2, B is 2x1, C is 2x1
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    schoolbook_2x2x1 = [
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        ({(0,1):1}, {(1,0):1}, {(0,0):1}),
        ({(1,0):1}, {(0,0):1}, {(1,0):1}),
        ({(1,1):1}, {(1,0):1}, {(1,0):1}),
    ]
    
    rank_terms = []
    
    # Part 1: 4x4x4 via Strassen ⊗ Strassen (rank 49)
    # Global indices: i, j in 0..3 for A; j, k in 0..3 for B; i, k in 0..3 for C
    # Local: i1,i2 for row of A; j1,j2 for col of A / row of B; k1,k2 for col of B
    # Global: i = 2*i1+i2, j = 2*j1+j2, k = 2*k1+k2
    for s in range(7):
        u_s, v_s, w_s = strassen_2x2[s]
        for t in range(7):
            u_t, v_t, w_t = strassen_2x2[t]
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            # U part: (i,j) pairs
            for (i1, j1), cu in u_s.items():
                for (i2, j2), cu2 in u_t.items():
                    gi = 2 * i1 + i2
                    gj = 2 * j1 + j2
                    coeff = cu * cu2
                    key = (gi, gj)
                    U_global[key] = U_global.get(key, 0) + coeff
            
            # V part: (j,k) pairs
            for (j1, k1), cv in v_s.items():
                for (j2, k2), cv2 in v_t.items():
                    gj = 2 * j1 + j2
                    gk = 2 * k1 + k2
                    coeff = cv * cv2
                    key = (gj, gk)
                    V_global[key] = V_global.get(key, 0) + coeff
            
            # W part: (k,i) pairs (note: W index is (k, i) per trace convention)
            for (i1, k1), cw in w_s.items():
                for (i2, k2), cw2 in w_t.items():
                    gi = 2 * i1 + i2
                    gk = 2 * k1 + k2
                    coeff = cw * cw2
                    key = (gk, gi)
                    W_global[key] = W_global.get(key, 0) + coeff
            
            rank_terms.append((U_global, V_global, W_global))
    
    # Part 2: 4x4x2 via Strassen ⊗ Schoolbook_2x2x1 (rank 28)
    # A is 4x4, B_right is 4x2 (columns 4,5), C_right is 4x2 (columns 4,5)
    # Split B into two 4x1 vectors: B[:, 4] and B[:, 5]
    # C[:, 4] = A * B[:, 4], C[:, 5] = A * B[:, 5]
    # For each column k in {4, 5}, use Strassen for A * B_col
    # B_col is 4x1, split into two 2x1 vectors
    # For column 4: B4 = [B[0,4], B[1,4], B[2,4], B[3,4]]
    #   Split: B4_top = [B[0,4], B[1,4]], B4_bot = [B[2,4], B[3,4]]
    # For column 5: B5 = [B[0,5], B[1,5], B[2,5], B[3,5]]
    #   Split: B5_top = [B[0,5], B[1,5]], B5_bot = [B[2,5], B[3,5]]
    
    # Strassen handles 4x4 * 2x1 via:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B_col = [b_top; b_bot] where b_top, b_bot are 2x1
    # C_col = A * B_col = [A11*b_top + A12*b_bot; A21*b_top + A22*b_bot]
    # Each 2x2 * 2x1 product has rank 4 (schoolbook)
    # Using Strassen for the 2x2 matrix mult part: rank 7
    # So each 4x4 * 2x1 has rank 7*4 = 28? No wait.
    
    # Actually: A (4x4) * B_col (4x1)
    # Write A = [[A_tl, A_tr], [A_bl, A_br]] with 2x2 blocks
    # B_col = [b_top; b_bot] with 2x1 blocks
    # C_col_top = A_tl * b_top + A_tr * b_bot  (2x1 result)
    # C_col_bot = A_bl * b_top + A_br * b_bot  (2x1 result)
    
    # Each term like A_tl * b_top is a 2x2 * 2x1 product, rank 4
    # But we can combine: for a fixed Strassen term on the 2x2 blocks,
    # we get intermediate products that are scalars, then multiply by b_top or b_bot
    
    # Let me think differently. Use the tensor product structure.
    # M_{4,4,2} = M_{2,2,2} ⊗ M_{2,2,1}
    # First factor: 2x2x2 (Strassen, rank 7) operates on the "block" indices
    # Second factor: 2x2x1 (schoolbook, rank 4) operates on the "within-block" indices
    
    # Global A: (i, j) where i = 2*i1+i2, j = 2*j1+j2
    # Global B_col: (j, k) where j = 2*j1+j2, k = 4 + k1 (k1 in {0,1} for the two columns)
    #   Wait, for the 4x4x2 part, k ranges over {4,5}
    #   Let me redefine: for the right 2 columns, local index k2 in {0,1}, global k = 4 + k2
    #   And j is split as j = 2*j1 + j2
    # Global C_col: (k, i) where k = 4 + k2, i = 2*i1+i2
    
    # The tensor product M_{2,2,2} ⊗ M_{2,2,1}:
    # First tensor (2,2,2): indices (i1, j1) for A_block, (j1, k1) for B_block, (i1, k1) for C_block
    #   But wait, the second dimension of the first tensor is j1, and the third is k1
    #   For M_{2,2,2}: A is 2x2 (i1,j1), B is 2x2 (j1,k1), C is 2x2 (i1,k1)
    #   But we want B to be 2x1, not 2x2. So the second tensor should be M_{2,2,1}
    #   M_{2,2,1}: A is 2x2 (i2,j2), B is 2x1 (j2,k2), C is 2x1 (i2,k2)
    
    # Combined:
    # A: (i1, j1) from first, (i2, j2) from second → global (2*i1+i2, 2*j1+j2)
    # B: (j1, k1) from first, (j2, k2) from second → global (2*j1+j2, 4+k2) where k1 is... 
    #   Hmm, the first tensor's B has shape 2x2, but we only have 2 rows (j1) and 1 "column group" (k1)?
    #   Actually for M_{2,2,2} ⊗ M_{2,2,1}:
    #   First: A(i1,j1), B(j1,k1), C(i1,k1) all 2x2 or 2x1
    #   Wait no. M_{a,b,c} means A is a×b, B is b×c, C is a×c.
    #   M_{2,2,2}: A(2,2), B(2,2), C(2,2)
    #   M_{2,2,1}: A(2,2), B(2,1), C(2,1)
    #   Tensor product M_{2,2,2} ⊗ M_{2,2,1}:
    #   A: (2*2, 2*2) = (4,4)
    #   B: (2*2, 2*1) = (4,2)
    #   C: (2*2, 2*1) = (4,2)
    #   Yes! This is exactly M_{4,4,2}.
    
    # Indices:
    # A global (i, j): i = 2*i1 + i2, j = 2*j1 + j2
    #   i1, i2 from 0..1; j1, j2 from 0..1
    # B global (j, k): j = 2*j1 + j2, k = 4 + k2
    #   j1, j2 from 0..1; k2 from 0..1 (the "1" in M_{2,2,1} means 1 column)
    # C global (k, i): k = 4 + k2, i = 2*i1 + i2
    
    # For the first tensor (Strassen 2x2x2):
    #   A indices: (i1, j1)
    #   B indices: (j1, k1) where k1 in {0,1} — but wait, B in M_{2,2,2} is 2x2
    #   However, in the tensor product, the "c" dimension of the first tensor (size 2) 
    #   combines with the "c" dimension of the second tensor (size 1)
    #   So B global column index: k1 from first tensor, k2 from second tensor
    #   But the second tensor has only 1 column, so k2 is always 0? No...
    
    # Let me reconsider. In tensor product of M_{a1,b1,c1} and M_{a2,b2,c2}:
    #   Result is M_{a1*a2, b1*b2, c1*c2}
    #   A global row: a1*a2, indexed by (i1, i2) where i1 in 0..a1-1, i2 in 0..a2-1
    #   A global col: b1*b2, indexed by (j1, j2) where j1 in 0..b1-1, j2 in 0..b2-1
    #   B global row: b1*b2, indexed by (j1, j2)
    #   B global col: c1*c2, indexed by (k1, k2) where k1 in 0..c1-1, k2 in 0..c2-1
    #   C global row: a1*a2, indexed by (i1, i2)
    #   C global col: c1*c2, indexed by (k1, k2)
    
    # For M_{2,2,2} ⊗ M_{2,2,1}:
    #   a1=2, b1=2, c1=2; a2=2, b2=2, c2=1
    #   A: (4,4), row = 2*i1+i2, col = 2*j1+j2
    #   B: (4,2), row = 2*j1+j2, col = 2*k1+k2 where k1 in 0..1, k2 in 0..0 (only 0)
    #     So B col = 2*k1 + 0 = 2*k1, which gives {0, 1}
    #     But we need B cols to be {4, 5}! So we add an offset of 4.
    #   C: (4,2), row = 2*i1+i2, col = 2*k1+k2 = 2*k1, offset by 4 → {4, 5}
    
    # So for the 4x4x2 part:
    for s in range(7):
        u_s, v_s, w_s = strassen_2x2[s]
        for t in range(4):
            u_t, v_t, w_t = schoolbook_2x2x1[t]
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            # U: (i1, j1) from s, (i2, j2) from t → (2*i1+i2, 2*j1+j2)
            for (i1, j1), cu in u_s.items():
                for (i2, j2), cu2 in u_t.items():
                    gi = 2 * i1 + i2
                    gj = 2 * j1 + j2
                    coeff = cu * cu2
                    key = (gi, gj)
                    U_global[key] = U_global.get(key, 0) + coeff
            
            # V: (j1, k1) from s, (j2, k2) from t → (2*j1+j2, 4 + 2*k1+k2)
            #   k2 is always 0 since M_{2,2,1} has c2=1
            for (j1, k1), cv in v_s.items():
                for (j2, k2), cv2 in v_t.items():
                    gj = 2 * j1 + j2
                    gk = 4 + 2 * k1 + k2  # k2 is 0
                    coeff = cv * cv2
                    key = (gj, gk)
                    V_global[key] = V_global.get(key, 0) + coeff
            
            # W: (i1, k1) from s, (i2, k2) from t → (4 + 2*k1+k2, 2*i1+i2)
            for (i1, k1), cw in w_s.items():
                for (i2, k2), cw2 in w_t.items():
                    gi = 2 * i1 + i2
                    gk = 4 + 2 * k1 + k2  # k2 is 0
                    coeff = cw * cw2
                    key = (gk, gi)
                    W_global[key] = W_global.get(key, 0) + coeff
            
            rank_terms.append((U_global, V_global, W_global))
    
    rank = len(rank_terms)
    # Should be 49 + 28 = 77
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, r] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i, r] = coeff
    
    # Verify
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        alpha_r = np.sum(A_test * U[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W[:, r].reshape(p, n).T
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

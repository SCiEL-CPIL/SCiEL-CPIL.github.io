"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Step 1: Generate Strassen decomposition for 2x2x2
    # U, V, W are 4x7
    U_2x2 = [[0]*7 for _ in range(4)]
    V_2x2 = [[0]*7 for _ in range(4)]
    W_2x2 = [[0]*7 for _ in range(4)]
    
    # Indices for 2x2 matrix: i*2+j
    # A: 00:0, 01:1, 10:2, 11:3
    # B: 00:0, 01:1, 10:2, 11:3
    # C: 00:0, 01:1, 10:2, 11:3 (but W index is k*2+i)
    # W index k*2+i:
    # k=0 (col 0), i=0 (row 0) -> 0
    # k=0 (col 0), i=1 (row 1) -> 1
    # k=1 (col 1), i=0 (row 0) -> 2
    # k=1 (col 1), i=1 (row 1) -> 3
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: A00(0), A11(3) -> [1,0,0,1]
    # V: B00(0), B11(3) -> [1,0,0,1]
    # W: C00(0), C11(3) -> [1,0,0,1] (k=0,i=0 and k=1,i=1)
    U_2x2[0][0] += 1; U_2x2[3][0] += 1
    V_2x2[0][0] += 1; V_2x2[3][0] += 1
    W_2x2[0][0] += 1; W_2x2[3][0] += 1
    
    # Term 1: M2 = (A10+A11)B00
    # U: A10(2), A11(3) -> [0,0,1,1]
    # V: B00(0) -> [1,0,0,0]
    # W: C10(2) (k=1,i=0) -> [0,0,1,0]
    U_2x2[2][1] += 1; U_2x2[3][1] += 1
    V_2x2[0][1] += 1
    W_2x2[2][1] += 1
    
    # Term 2: M3 = A00(B01-B11)
    # U: A00(0) -> [1,0,0,0]
    # V: B01(1), B11(3) -> [0,1,0,-1]
    # W: C01(1) (k=1,i=0? No C01 is row 0 col 1. k=1,i=0. Index 1*2+0=2? 
    # Wait. C_ij. i=row, j=col.
    # W index k*n+i. k=col, i=row.
    # C01: i=0, j=1. k=1, i=0. Index 1*2+0 = 2.
    # Let's re-verify W index mapping.
    # W[k*n+i].
    # If C is n x p.
    # Usually C[i,j].
    # If W is p x n. W[j,i].
    # Then C[i,j] = W[j,i].
    # So for C01 (i=0, j=1), W[1,0]. Index 1*2+0 = 2.
    # My previous mapping:
    # C00 (i=0,j=0) -> W[0,0] -> 0
    # C01 (i=0,j=1) -> W[1,0] -> 2
    # C10 (i=1,j=0) -> W[0,1] -> 1
    # C11 (i=1,j=1) -> W[1,1] -> 3
    #
    # Let's re-evaluate Term 2.
    # M3 contributes to C01.
    # So W index 2.
    U_2x2[0][2] += 1
    V_2x2[1][2] += 1; V_2x2[3][2] -= 1
    W_2x2[2][2] += 1
    
    # Term 3: M4 = A11(B10-B00)
    # U: A11(3) -> [0,0,0,1]
    # V: B10(2), B00(0) -> [-1,0,1,0]
    # W: C00(0) and C10(1).
    # C00: i=0,j=0 -> W[0,0] -> 0
    # C10: i=1,j=0 -> W[0,1] -> 1
    U_2x2[3][3] += 1
    V_2x2[0][3] -= 1; V_2x2[2][3] += 1
    W_2x2[0][3] += 1; W_2x2[1][3] += 1
    
    # Term 4: M5 = (A00+A01)B11
    # U: A00(0), A01(1) -> [1,1,0,0]
    # V: B11(3) -> [0,0,0,1]
    # W: C00(-1) and C01(1).
    # C00: W[0,0] -> 0
    # C01: W[1,0] -> 2
    U_2x2[0][4] += 1; U_2x2[1][4] += 1
    V_2x2[3][4] += 1
    W_2x2[0][4] -= 1; W_2x2[2][4] += 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    # U: A10(2), A00(0) -> [-1,0,1,0]
    # V: B00(0), B01(1) -> [1,1,0,0]
    # W: C11(-1).
    # C11: i=1,j=1 -> W[1,1] -> 3
    U_2x2[0][5] -= 1; U_2x2[2][5] += 1
    V_2x2[0][5] += 1; V_2x2[1][5] += 1
    W_2x2[3][5] -= 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    # U: A01(1), A11(3) -> [0,1,0,-1]
    # V: B10(2), B11(3) -> [0,0,1,1]
    # W: C00(-1) and C11(1).
    # C00: W[0,0] -> 0
    # C11: W[1,1] -> 3
    U_2x2[1][6] += 1; U_2x2[3][6] -= 1
    V_2x2[2][6] += 1; V_2x2[3][6] += 1
    W_2x2[0][6] -= 1; W_2x2[3][6] += 1

    # Step 2: Construct 4x4x4 decomposition
    # Rank 49.
    # U: 16x49, V: 16x49, W: 16x49
    # We use Kronecker product of block indices and Strassen indices.
    # Block indices: 00, 01, 10, 11.
    # Strassen indices: 0..6.
    # Total 49 terms.
    
    # We need to map the outer Strassen terms to the inner blocks.
    # Outer Strassen operates on 2x2 blocks.
    # Let's define the outer Strassen vectors for blocks.
    # U_block (4x7), V_block (4x7), W_block (4x7).
    # These are the same structure as U_2x2, but indices refer to blocks.
    
    # The full U is sum over outer term t and inner term s.
    # U_full [ (bi, bj, si), (t, s) ]?
    # No, the decomposition is a sum of 49 rank-1 terms.
    # Each term corresponds to an outer Strassen term t and an inner Strassen term s.
    # The coefficient for A_{ij} in term (t,s) is U_2x2[bi*2+bj, t] * U_2x2[si, s]?
    # No, that's not right.
    
    # The tensor T_4x4x4 = sum_{t=0}^6 sum_{s=0}^6 u_t \otimes v_t \otimes w_t (outer)
    # But u_t is a 4x4 matrix? No.
    
    # Correct approach:
    # T_4x4x4(i,j,k) = sum_{b1,b2} A_{block(b1,b2)} B_{block(b2,b3)} ...
    # This is complex.
    
    # Simpler: Just use the fact that Rank(4x4x4) = 49.
    # We can construct it by iterating all 49 combinations of outer and inner Strassen terms.
    
    U_4x4 = [[0]*49 for _ in range(16)]
    V_4x4 = [[0]*49 for _ in range(16)]
    W_4x4 = [[0]*49 for _ in range(16)]
    
    for t in range(7):
        for s in range(7):
            c = t*7 + s
            # Outer Strassen term t
            # Inner Strassen term s
            
            # U part:
            # A_{ij} = sum_{b1,b2} A_{block(b1,b2) , inner}
            # The outer term t determines which blocks are added.
            # The inner term s determines which entries within those blocks are added.
            
            # U_2x2[bi*2+bj, t] gives the coefficient for block (bi,bj) in outer term t.
            # U_2x2[si, s] gives the coefficient for inner index si in inner term s.
            # Wait, U_2x2 indices are i*2+j.
            # For blocks, bi*2+bj.
            # For inner, si.
            
            for bi in range(2):
                for bj in range(2):
                    coeff_outer = U_2x2[bi*2+bj][t]
                    if coeff_outer != 0:
                        # This block (bi,bj) is involved with coeff_outer
                        # Now iterate inner indices si
                        for si in range(4):
                            coeff_inner = U_2x2[si][s]
                            if coeff_inner != 0:
                                # Total index in A:
                                # Block (bi,bj) starts at row bi*2, col bj*2.
                                # Inner index si is i*2+j.
                                # So row = bi*2 + si//2
                                # Col = bj*2 + si%2
                                a_row = bi*2 + si//2
                                a_col = bj*2 + si%2
                                a_idx = a_row*4 + a_col
                                U_4x4[a_idx][c] += coeff_outer * coeff_inner

            # V part:
            for bj in range(2):
                for bk in range(2):
                    coeff_outer = V_2x2[bj*2+bk][t]
                    if coeff_outer != 0:
                        for sk in range(4):
                            coeff_inner = V_2x2[sk][s]
                            if coeff_inner != 0:
                                # Block (bj,bk). Inner sk.
                                b_row = bj*2 + sk//2
                                b_col = bk*2 + sk%2
                                b_idx = b_row*4 + b_col
                                V_4x4[b_idx][c] += coeff_outer * coeff_inner

            # W part:
            for bk in range(2):
                for bi in range(2):
                    coeff_outer = W_2x2[bk*2+bi][t]
                    if coeff_outer != 0:
                        for wi in range(4):
                            coeff_inner = W_2x2[wi][s]
                            if coeff_inner != 0:
                                # Block (bk,bi). Inner wi.
                                # W index in 2x2 is k*2+i.
                                # So wi = k_in*2 + i_in.
                                # Block row k = bk*2 + k_in
                                # Block col i = bi*2 + i_in
                                # W index in 4x4 is k*4 + i.
                                k_out = bk*2 + wi//2
                                i_out = bi*2 + wi%2
                                w_idx = k_out*4 + i_out
                                W_4x4[w_idx][c] += coeff_outer * coeff_inner

    # Step 3: Construct 4x4x8 decomposition
    # Rank 98.
    # Split B into B1 (cols 0-3) and B2 (cols 4-7).
    # C into C1 (cols 0-3) and C2 (cols 4-7).
    # Two independent 4x4x4 multiplications.
    
    U_4x4x8 = U_4x4 # 16x49, duplicated? No, U is same for both.
    # Actually, U is shared. V and W are separate.
    # Total rank 98.
    # Terms 0-48: First multiplication.
    # Terms 49-97: Second multiplication.
    
    U_4x4x8 = [[0]*98 for _ in range(16)]
    V_4x4x8 = [[0]*98 for _ in range(32)]
    W_4x4x8 = [[0]*98 for _ in range(32)]
    
    for c in range(49):
        # First half
        for i in range(16):
            U_4x4x8[i][c] = U_4x4[i][c]
            U_4x4x8[i][c+49] = U_4x4[i][c]
            
            V_4x4x8[i][c] = V_4x4[i][c]
            # Second half V is shifted by 16? No, B2 is cols 4-7.
            # B2 indices are 4..15? No, B is 4x8.
            # B1 is 4x4 (cols 0-3). B2 is 4x4 (cols 4-7).
            # V_4x4 indices are 0..15 (4x4).
            # For V_4x4x8, first half uses indices 0..15.
            # Second half uses indices 16..31?
            # B index j*8+k? No, B is 4x8. Row j, Col k.
            # Index j*8+k.
            # B1: k=0..3. Index j*8 + k.
            # B2: k=4..7. Index j*8 + (k+4).
            # V_4x4 index is j*4+k_in.
            # Map to V_4x4x8:
            # First half: j*8 + k_in.
            # Second half: j*8 + 4 + k_in.
            
            v_idx = i
            j = v_idx // 4
            k_in = v_idx % 4
            V_4x4x8[j*8 + k_in][c] = V_4x4[i][c]
            V_4x4x8[j*8 + 4 + k_in][c+49] = V_4x4[i][c]
            
            # W part
            # W_4x4 index is k*4+i.
            # W_4x4x8 index is k_out*4+i.
            # First half: k_out = k_in.
            # Second half: k_out = 4 + k_in.
            w_idx = i
            k_in = w_idx // 4
            i_in = w_idx % 4
            W_4x4x8[k_in*4 + i_in][c] = W_4x4[i][c]
            W_4x4x8[(4+k_in)*4 + i_in][c+49] = W_4x4[i][c]

    # Step 4: Truncate to 4x4x7
    # p=7.
    # V rows: 4*7 = 28.
    # W rows: 7*4 = 28.
    # We take the first 28 rows of V and W.
    # This corresponds to k_out = 0..6.
    # For V:
    # First half (k_out 0-3): always included.
    # Second half (k_out 4-7): we need k_out 4,5,6.
    # k_out = 4 + k_in.
    # k_in=0 -> k_out=4 (include)
    # k_in=1 -> k_out=5 (include)
    # k_in=2 -> k_out=6 (include)
    # k_in=3 -> k_out=7 (exclude)
    
    U_final = U_4x4x8
    V_final = []
    for r in range(28):
        V_final.append(V_4x4x8[r])
        
    W_final = []
    for r in range(28):
        W_final.append(W_4x4x8[r])

    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END

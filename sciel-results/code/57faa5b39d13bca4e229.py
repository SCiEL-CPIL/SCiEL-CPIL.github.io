import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Construct Rank-49 decomposition for 4x4x4 (Strassen)
    # U44: 16x49, V44: 16x49, W44: 16x49
    U44, V44, W44 = _strassen_4x4x4()
    
    # 2. Construct Rank-16 decomposition for 4x1x4 (Outer Product)
    # U41: 4x16, V14: 4x16, W414: 16x16
    U41, V14, W414 = _outer_product_4x1x4()
    
    # 3. Combine into 4x5x4 (Rank 65)
    # U_p: 20x65, V_p: 20x65, W_p: 16x65
    U_p = [[0] * 65 for _ in range(20)]
    V_p = [[0] * 65 for _ in range(20)]
    W_p = [[0] * 65 for _ in range(16)]
    
    # Part 1: 4x4x4 (j < 4)
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U_p[i * 5 + j][r] = U44[i * 4 + j][r]
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V_p[j * 4 + k][r] = V44[j * 4 + k][r]
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W_p[k * 4 + i][r] = W44[k * 4 + i][r]
                
    # Part 2: 4x1x4 (j == 4)
    for i in range(4):
        for r in range(16):
            U_p[i * 5 + 4][49 + r] = U41[i][r]
    for k in range(4):
        for r in range(16):
            V_p[4 * 4 + k][49 + r] = V14[k][r]
    for k in range(4):
        for i in range(4):
            for r in range(16):
                W_p[k * 4 + i][49 + r] = W414[k * 4 + i][r]

    # 4. Transform to 4x4x5 (Rank 65)
    # U: 16x65, V: 20x65, W: 20x65
    U = [[0] * 65 for _ in range(16)]
    V = [[0] * 65 for _ in range(20)]
    W = [[0] * 65 for _ in range(20)]
    
    # Mapping:
    # U[i*4+j, r] = W_p[i*4+j, r]
    for i in range(4):
        for j in range(4):
            for r in range(65):
                U[i * 4 + j][r] = W_p[i * 4 + j][r]
                
    # V[j*5+k, r] = U_p[j*5+k, r]
    for j in range(4):
        for k in range(5):
            for r in range(65):
                V[j * 5 + k][r] = U_p[j * 5 + k][r]
                
    # W[k*4+i, r] = V_p[k*4+i, r]
    for k in range(5):
        for i in range(4):
            for r in range(65):
                W[k * 4 + i][r] = V_p[k * 4 + i][r]
                
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    # Returns U, V, W for 4x4x4 with rank 49
    # Using 7 Strassen terms for 2x2 blocks, each term is 2x2x4? No, 2x2x2 rank 7.
    # 4x4x4 is 2x(2x2) x 2x(2x2).
    # We can use the known rank 49 decomposition.
    # We will construct it recursively.
    
    def strassen_2x2x2():
        # Rank 7 decomposition for 2x2x2
        # U: 4x7, V: 4x7, W: 4x7
        U = [
            [1, 0, 1, 0, 0, 0, 0], # A00
            [0, 0, 0, 0, 1, 0, 0], # A01
            [0, 1, 0, 0, 0, 0, 0], # A10
            [0, 0, 0, 1, 0, 0, 0]  # A11
        ]
        V = [
            [1, 1, 0, 0, 0, 0, 0], # B00
            [0, 0, 0, 0, 1, 1, 0], # B01
            [0, 0, 1, 1, 0, 0, 0], # B10
            [0, 0, 0, 0, 1, 0, 1]  # B11
        ]
        W = [
            [1, 0, 0, 0, 0, 0, 0], # C00
            [0, 0, 1, 0, 0, 0, 0], # C01
            [0, 1, 0, 0, 0, 0, 0], # C10
            [0, 0, 0, 1, 0, 0, 0]  # C11
        ]
        # This specific set is one of many. Need to ensure correctness.
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Let's build U, V, W for these 7 terms.
        # U rows: A00, A01, A10, A11
        # V rows: B00, B01, B10, B11
        # W rows: C00, C01, C10, C11 (Trace: C[k*2+i] -> C00=0, C01=1, C10=2, C11=3)
        
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # Term 1: M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 += 0, C10 += 0, C11 -= M1
        # Actually C00 = M1 - M4 - M5 + M7, C01 = M3 + M5, C10 = M2 + M4, C11 = M1 - M2 - M3 + M6
        # Let's use the standard coefficients.
        
        # M1
        U_s[0][0] += 1; U_s[3][0] += 1
        V_s[0][0] += 1; V_s[3][0] += 1
        W_s[0][0] += 1; W_s[3][0] -= 1
        
        # M2
        U_s[2][1] += 1; U_s[3][1] += 1
        V_s[0][1] += 1
        W_s[2][1] += 1; W_s[3][1] -= 1
        
        # M3
        U_s[0][2] += 1
        V_s[1][2] += 1; V_s[3][2] -= 1
        W_s[1][2] += 1; W_s[3][2] -= 1
        
        # M4
        U_s[3][3] += 1
        V_s[2][3] += 1; V_s[0][3] -= 1
        W_s[0][3] -= 1; W_s[2][3] += 1
        
        # M5
        U_s[0][4] += 1; U_s[1][4] += 1
        V_s[3][4] += 1
        W_s[0][4] -= 1; W_s[1][4] += 1
        
        # M6
        U_s[2][5] -= 1; U_s[0][5] -= 1
        V_s[0][5] += 1; V_s[1][5] += 1
        W_s[3][5] += 1
        
        # M7
        U_s[1][6] += 1; U_s[3][6] -= 1
        V_s[2][6] += 1; V_s[3][6] += 1
        W_s[0][6] += 1; W_s[1][6] -= 1
        
        return U_s, V_s, W_s

    U22, V22, W22 = strassen_2x2x2()
    
    # Now build 4x4x4 from 4x(2x2) blocks?
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # We can use Strassen again on the block level?
    # The block level is 2x2x2 matrix multiplication.
    # So we use the same U22, V22, W22 structure but applied to blocks.
    # Each "element" in the block multiplication is a 2x2 matrix multiplication.
    # So each term M_k in the block Strassen corresponds to a 2x2x2 multiplication.
    # Total rank = 7 (block terms) * 7 (element terms) = 49.
    
    U44 = [[0]*49 for _ in range(16)]
    V44 = [[0]*49 for _ in range(16)]
    W44 = [[0]*49 for _ in range(16)]
    
    # Block indices:
    # A blocks: A00 (0-3), A01 (4-7), A10 (8-11), A11 (12-15) in 2x2 layout?
    # U44 row index: i*4+j.
    # i=0, j=0 -> A00. i=0, j=1 -> A01. i=1, j=0 -> A10. i=1, j=1 -> A11.
    # Wait, A is 4x4.
    # A[0,0] is in A00[0,0].
    # A[0,1] is in A00[0,1].
    # A[0,2] is in A01[0,0].
    # A[0,3] is in A01[0,1].
    
    # Mapping from 4x4 index (i,j) to block (bi,bj) and local (li,lj).
    # bi = i//2, bj = j//2.
    # li = i%2, lj = j%2.
    
    # For each block term k (0..6) in Strassen:
    # We have coefficients for A_blocks and B_blocks.
    # U22[bi, k] is coeff for A_block bi.
    # V22[bj, k] is coeff for B_block bj.
    # W22[ci, k] is coeff for C_block ci.
    
    # For each element term l (0..6) in Strassen (inside the block multiplication):
    # We have coefficients for local elements.
    # U22[li, l] is coeff for local A element li.
    # V22[lj, l] is coeff for local B element lj.
    # W22[ck, l] is coeff for local C element ck.
    
    # Total term index r = k*7 + l.
    
    for k in range(7):
        for l in range(7):
            r = k * 7 + l
            # Accumulate into U44, V44, W44
            # U44 depends on i,j.
            # i = bi*2 + li
            # j = bj*2 + lj
            for bi in range(2):
                for bj in range(2):
                    for li in range(2):
                        for lj in range(2):
                            i = bi * 2 + li
                            j = bj * 2 + lj
                            coeff_u = U22[bi][k] * U22[li][l]
                            if coeff_u != 0:
                                U44[i * 4 + j][r] += coeff_u
                                
            for bj in range(2):
                for bk in range(2): # k is block col? No, B is 4x4.
                    # B block indices: bk (row block), bk_col (col block)?
                    # B is 4x4. Blocks: B00, B01, B10, B11.
                    # B[i,j] -> bi=i//2, bj=j//2.
                    # So V44 row index is j*4+k? No, V is m x p.
                    # V row index: j*4+k.
                    # j is col of A / row of B.
                    # k is col of B.
                    # So j = bj*2 + lj? No.
                    # In B, row index is j (from A), col index is k.
                    # Block row of B is bj. Block col of B is bk.
                    # Local row is lj. Local col is lk.
                    for bk in range(2):
                        for lj in range(2):
                            for lk in range(2):
                                j = bj * 2 + lj
                                k = bk * 2 + lk
                                coeff_v = V22[bj][k] * V22[lj][l]
                                if coeff_v != 0:
                                    V44[j * 4 + k][r] += coeff_v
                                    
            for bk in range(2):
                for bi in range(2):
                    for lk in range(2):
                        for li in range(2):
                            k = bk * 2 + lk
                            i = bi * 2 + li
                            coeff_w = W22[bk][k] * W22[lk][l]
                            if coeff_w != 0:
                                W44[k * 4 + i][r] += coeff_w
                                
    return U44, V44, W44

def _outer_product_4x1x4():
    # Rank 16
    U = [[0]*16 for _ in range(4)]
    V = [[0]*16 for _ in range(4)] # V is 1x4, so 4 rows
    W = [[0]*16 for _ in range(16)] # W is 4x4, so 16 rows
    
    for i in range(4):
        for k in range(4):
            r = i * 4 + k
            U[i][r] = 1
            V[k][r] = 1
            W[k * 4 + i][r] = 1
            
    return U, V, W

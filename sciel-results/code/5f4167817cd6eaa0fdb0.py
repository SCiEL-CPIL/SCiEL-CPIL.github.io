import time
import itertools

DIMENSIONS = [4, 4, 9]
START_TIME = time.time()
TIME_LIMIT = 115

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x9 matrix.
    
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    
    # Define the rank-7 Strassen decomposition for 2x2 matrix multiplication
    # u, v, w are lists of 7 vectors of length 4
    # Indexing for 2x2:
    # A: [[a00, a01], [a10, a11]] -> indices 0, 1, 2, 3
    # B: [[b00, b01], [b10, b11]] -> indices 0, 1, 2, 3
    # C: [[c00, c01], [c10, c11]] -> indices 0, 1, 2, 3
    
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    strassen_w = [
        [1, 0, 0, 1], # M1: c00+, c11+
        [0, 0, 1, 0], # M2: c10+
        [0, 1, 0, -1],# M3: c01+, c11-
        [-1, 0, 1, 0],# M4: c00-, c10+
        [0, 1, 0, 1], # M5: c01+, c11+
        [1, 0, 0, 0], # M6: c00+
        [-1, 0, 0, -1]# M7: c00-, c11-
    ]
    
    # Construct rank-49 decomposition for 4x4 x 4x4
    # We split A and B into 2x2 blocks.
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1) in block coords.
    # Actual indices:
    # A00: i in 0,1, j in 0,1 -> idx 0,1,4,5
    # A01: i in 0,1, j in 2,3 -> idx 2,3,6,7
    # A10: i in 2,3, j in 0,1 -> idx 8,9,12,13
    # A11: i in 2,3, j in 2,3 -> idx 10,11,14,15
    
    # B blocks: B00, B01, B10, B11
    # B00: j in 0,1, k in 0,1 -> idx 0,1,4,5 (for 4x4 B, p=4)
    # Note: For 4x4x4, V index is j*4+k.
    
    # We will generate the 49 terms.
    # Each term corresponds to a Strassen step (0-6) and a sub-Strassen step (0-6).
    
    U_44 = []
    V_44 = []
    W_44 = []
    
    for s in range(7):
        for ss in range(7):
            u_vec = [0]*16
            v_vec = [0]*16 # for 4x4 B
            w_vec = [0]*16 # for 4x4 C
            
            # Determine which A block is involved in Strassen step s
            # u_s is a vector of 4 coeffs for the 2x2 block matrix.
            # The 2x2 block matrix entries correspond to A00, A01, A10, A11.
            # u_s[0] -> A00, u_s[1] -> A01, u_s[2] -> A10, u_s[3] -> A11
            
            # Determine which B block is involved
            # v_s[0] -> B00, v_s[1] -> B01, v_s[2] -> B10, v_s[3] -> B11
            
            # The sub-step ss uses strassen_u[ss], strassen_v[ss], strassen_w[ss]
            # which are vectors of length 4 for the 2x2 multiplication.
            
            # We need to map the 2x2 block indices to the 4x4 indices.
            
            # A part:
            # The "matrix" being multiplied in step s is a linear combination of blocks.
            # Let A_comb = u_s[0]*A00 + u_s[1]*A01 + u_s[2]*A10 + u_s[3]*A11.
            # This is a 4x4 matrix.
            # The sub-step ss treats A_comb as a 2x2 block matrix? No.
            # Strassen for 4x4 computes 7 products of 2x2 matrices.
            # Each product is (A_block_comb) * (B_block_comb).
            # We decompose this 2x2x2 product using the rank-7 decomposition.
            
            # So, for step s, we have a "virtual" 2x2 matrix A_s and B_s.
            # A_s is formed by taking the blocks of A according to u_s.
            # B_s is formed by taking the blocks of B according to v_s.
            
            # The sub-decomposition ss provides coefficients for the entries of A_s and B_s.
            # strassen_u[ss] = [a00_s, a01_s, a10_s, a11_s]
            # strassen_v[ss] = [b00_s, b01_s, b10_s, b11_s]
            
            # We need to populate u_vec (size 16) for the 4x4 A.
            # A_s[0,0] corresponds to A00. Coeff is u_s[0] * strassen_u[ss][0].
            # A_s[0,1] corresponds to A01. Coeff is u_s[1] * strassen_u[ss][1].
            # A_s[1,0] corresponds to A10. Coeff is u_s[2] * strassen_u[ss][2].
            # A_s[1,1] corresponds to A11. Coeff is u_s[3] * strassen_u[ss][3].
            
            # Map block (bi, bj) to 4x4 indices:
            # bi=0, bj=0 -> i in 0,1, j in 0,1 -> 0,1,4,5
            # bi=0, bj=1 -> i in 0,1, j in 2,3 -> 2,3,6,7
            # bi=1, bj=0 -> i in 2,3, j in 0,1 -> 8,9,12,13
            # bi=1, bj=1 -> i in 2,3, j in 2,3 -> 10,11,14,15
            
            block_offsets = [
                [0, 1, 4, 5],   # 0,0
                [2, 3, 6, 7],   # 0,1
                [8, 9, 12, 13], # 1,0
                [10, 11, 14, 15]# 1,1
            ]
            
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    coeff_a = u_s[block_idx] * strassen_u[ss][block_idx]
                    if coeff_a != 0:
                        for idx in block_offsets[block_idx]:
                            u_vec[idx] = coeff_a
            
            # B part:
            # Similar mapping for B.
            # B_s[0,0] is B00. Coeff is v_s[0] * strassen_v[ss][0].
            # Note: For 4x4 B, indices are j*4+k.
            # B00: j in 0,1, k in 0,1 -> 0,1,4,5
            # B01: j in 0,1, k in 2,3 -> 2,3,6,7
            # B10: j in 2,3, k in 0,1 -> 8,9,12,13
            # B11: j in 2,3, k in 2,3 -> 10,11,14,15
            # Same offsets.
            
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    coeff_b = v_s[block_idx] * strassen_v[ss][block_idx]
                    if coeff_b != 0:
                        for idx in block_offsets[block_idx]:
                            v_vec[idx] = coeff_b
            
            # W part:
            # The result C_s is a 4x4 matrix.
            # The sub-step ss produces a 2x2 result C_ss.
            # C_ss[0,0] contributes to C00.
            # C_ss[0,1] contributes to C01.
            # C_ss[1,0] contributes to C10.
            # C_ss[1,1] contributes to C11.
            
            # strassen_w[ss] = [c00_s, c01_s, c10_s, c11_s]
            
            # The block C00 corresponds to 4x4 C indices:
            # C00: i in 0,1, k in 0,1 -> 0,1,4,5
            # C01: i in 0,1, k in 2,3 -> 2,3,6,7
            # C10: i in 2,3, k in 0,1 -> 8,9,12,13
            # C11: i in 2,3, k in 2,3 -> 10,11,14,15
            
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    coeff_c = strassen_w[ss][block_idx]
                    if coeff_c != 0:
                        for idx in block_offsets[block_idx]:
                            w_vec[idx] = coeff_c
            
            U_44.append(u_vec)
            V_44.append(v_vec)
            W_44.append(w_vec)
            
    # Now we have rank-49 decomposition for 4x4 x 4x4.
    # We need to extend this to 4x4 x 4x9.
    # Split B into B1 (4x4), B2 (4x4), B3 (4x1).
    
    rank_total = 49 + 49 + 16
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(36)]
    W = [[0] * rank_total for _ in range(36)]
    
    # Part 1: B1 (cols 0-3)
    for r in range(49):
        for i in range(16):
            U[i][r] = U_44[r][i]
        for j in range(4):
            for k in range(4):
                # V index: j*9 + k
                V[j*9 + k][r] = V_44[r][j*4 + k]
        for k in range(4):
            for i in range(4):
                # W index: k*4 + i
                W[k*4 + i][r] = W_44[r][k*4 + i]
                
    # Part 2: B2 (cols 4-7)
    for r in range(49):
        for i in range(16):
            U[i][49 + r] = U_44[r][i]
        for j in range(4):
            for k in range(4):
                # V index: j*9 + (k+4)
                V[j*9 + k + 4][49 + r] = V_44[r][j*4 + k]
        for k in range(4):
            for i in range(4):
                # W index: (k+4)*4 + i
                W[(k+4)*4 + i][49 + r] = W_44[r][k*4 + i]
                
    # Part 3: B3 (col 8)
    # Schoolbook for 4x4 x 4x1
    for r in range(16):
        i = r // 4
        j = r % 4
        U[i*4 + j][98 + r] = 1
        V[j*9 + 8][98 + r] = 1
        W[8*4 + i][98 + r] = 1
        
    return {"U": U, "V": V, "W": W}


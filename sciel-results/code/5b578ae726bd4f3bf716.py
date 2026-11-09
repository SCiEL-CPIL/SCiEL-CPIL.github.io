"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create M_2,2,2 decomposition (Rank 7)
    def get_strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices for 2x2:
        # U, V: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # W: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[1, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[2, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = 1; W[1, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[3, 6] = 1
        
        return U, V, W

    U2, V2, W2 = get_strassen_2x2()
    
    # Construct M_4,4,4 (Rank 49)
    # Outer Strassen steps 1..7, Inner Strassen steps 1..7
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Mapping of block indices to global indices
    # A is 4x4, blocks are 2x2.
    # A_block(r,c) corresponds to rows 2r..2r+1, cols 2c..2c+1.
    # Global index for A[i,j] is i*4+j.
    # Global index for B[i,j] is i*4+j.
    # Global index for C[i,j] is j*4+i. (k*n+i)
    
    for r_out in range(7):
        for r_in in range(7):
            col = r_out * 7 + r_in
            
            # U4: A indices
            # u_out is a vector of length 4 (for 2x2 blocks)
            # u_in is a vector of length 4 (for elements within block)
            # The term is sum_{block_i, block_j} u_out[block_i, block_j] * u_in[elem_i, elem_j]
            # Wait, U2 is 4x7. U2[idx, r] gives the coefficient for A_block[idx] in step r.
            # The block idx is (bi, bj).
            # The element idx is (ei, ej).
            # Global idx = (2*bi + ei)*4 + (2*bj + ej).
            
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    coeff_out = U2[block_idx, r_out]
                    if coeff_out == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            elem_idx = ei * 2 + ej
                            coeff_in = U2[elem_idx, r_in]
                            if coeff_in == 0: continue
                            global_idx = (2*bi + ei)*4 + (2*bj + ej)
                            U4[global_idx, col] += coeff_out * coeff_in
            
            # V4: B indices
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj
                    coeff_out = V2[block_idx, r_out]
                    if coeff_out == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            elem_idx = ei * 2 + ej
                            coeff_in = V2[elem_idx, r_in]
                            if coeff_in == 0: continue
                            global_idx = (2*bi + ei)*4 + (2*bj + ej)
                            V4[global_idx, col] += coeff_out * coeff_in
            
            # W4: C indices
            # W index is k*4+i.
            # Block index (bk, bi) corresponds to rows 2bi..2bi+1, cols 2bk..2bk+1.
            # Global index for C[i,k] is k*4+i.
            # i = 2*bi + ei, k = 2*bk + ek.
            # W2 index for block (bk, bi) is bk*2+bi.
            # W2 index for elem (ek, ei) is ek*2+ei.
            
            for bk in range(2):
                for bi in range(2):
                    block_idx = bk * 2 + bi
                    coeff_out = W2[block_idx, r_out]
                    if coeff_out == 0: continue
                    for ek in range(2):
                        for ei in range(2):
                            elem_idx = ek * 2 + ei
                            coeff_in = W2[elem_idx, r_in]
                            if coeff_in == 0: continue
                            global_idx = (2*bk + ek)*4 + (2*bi + ei)
                            W4[global_idx, col] += coeff_out * coeff_in

    # Construct M_4,4,1 (Rank 16)
    # C_i = sum_j A_ij B_j
    # U idx i*4+j, V idx j, W idx i (k=0)
    U1 = np.zeros((16, 16))
    V1 = np.zeros((4, 16))
    W1 = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U1[i*4+j, col] = 1
            V1[j, col] = 1
            W1[i, col] = 1 # k=0, so index is 0*4+i = i

    # Combine M_4,4,9 = M_4,4,4 (cols 1-4 of B) + M_4,4,4 (cols 5-8 of B) + M_4,4,1 (col 9 of B)
    # B is 4x9.
    # Block 1: B cols 0-3. A is same. C cols 0-3.
    # Block 2: B cols 4-7. A is same. C cols 4-7.
    # Block 3: B col 8. A is same. C col 8.
    
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Block 1
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4 # C cols 0-3 correspond to W indices 0*4+i to 3*4+i?
    # W index is k*4+i.
    # For C cols 0-3, k is 0-3.
    # W4 was constructed for C being 4x4.
    # In M_4,4,4, W index is k*4+i.
    # This matches the first 16 rows of W (k=0..3).
    
    # Block 2
    U[:, 49:98] = U4
    # V for Block 2: B cols 4-7.
    # V4 is 16x49. V index is j*4+k.
    # For Block 2, k is 4-7.
    # So V index is j*4 + (k+4).
    # V4 has indices j*4+k' where k' is 0-3.
    # We need to map k' to k'+4.
    # V4[idx, col] corresponds to B[j, k'].
    # We want B[j, k'+4].
    # So we just place V4 into the correct rows of V.
    # V rows are j*4+k.
    # For j in 0-3, k in 4-7.
    # Rows 16..31 of V.
    for j in range(4):
        for k_prime in range(4):
            v_idx = j*4 + k_prime
            v_target = j*4 + (k_prime + 4)
            V[v_target, 49:98] = V4[v_idx, 0:49]
            
    # W for Block 2: C cols 4-7.
    # W index is k*4+i.
    # k is 4-7.
    # Rows 16..31 of W.
    W[16:32, 49:98] = W4
    
    # Block 3
    U[:, 98:114] = U1
    # V for Block 3: B col 8.
    # V1 is 4x16. V index is j.
    # We need B[j, 8].
    # V index is j*4+8.
    # Rows 8, 12, 16, 20? No.
    # V index is j*4+k. k=8.
    # j=0 -> 8. j=1 -> 12. j=2 -> 16. j=3 -> 20.
    # Wait, V is 36x114.
    # V rows are j*4+k.
    # For k=8, rows are 8, 12, 16, 20.
    for j in range(4):
        V[j*4+8, 98:114] = V1[j, 0:16]
        
    # W for Block 3: C col 8.
    # W index is k*4+i. k=8.
    # Rows 32..35.
    W[32:36, 98:114] = W1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

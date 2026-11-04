"""Matrix multiplication tensor decomposition for 4x4 times 4x7."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We will construct a decomposition with rank 49 (Strassen for 4x4x4) + 48 (Schoolbook for 4x4x3) = 97.
    # The 4x4x4 part handles columns 0-3 of B and C.
    # The 4x4x3 part handles columns 4-6 of B and C.
    
    # 1. Generate Strassen 2x2x2 factors
    # U, V, W shapes (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U2[0*2+0, 0] = 1; U2[1*2+1, 0] = 1
    V2[0*2+0, 0] = 1; V2[1*2+1, 0] = 1
    W2[0*2+0, 0] = 1; W2[1*2+1, 0] = 1 # Wait, W is for C. C11 gets M1, C22 gets M1.
    # Actually, let's just define the 7 terms explicitly for 2x2.
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M6
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 1: M2 = (A21+A22)B11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # Term 2: M3 = A11(B12-B22)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # Term 3: M4 = A22(B21-B11)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[0, 3] = -1
    
    # Term 4: M5 = (A11+A12)B22
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[3, 4] = 1; W2[1, 4] = 1
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[0, 5] = 1; W2[2, 5] = 1
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[1, 6] = -1; W2[3, 6] = 1
    
    # Verify 2x2 Strassen
    # C = U^T A V^T W ... no, standard tensor sum.
    # sum_c U[i,j,c] A[i,j] V[j,k,c] B[j,k] W[k,i,c] = C[i,k]
    
    # 2. Construct 4x4x4 Strassen from 2x2x2
    # We have 4 blocks in A: A00, A01, A10, A11
    # 4 blocks in B: B00, B01, B10, B11
    # We compute 7 "block products" Mk, each is a 2x2x2 multiplication.
    # Each Mk uses 7 scalar multiplications. Total 49.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Block offsets
    # A block (bi, bj) corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Index in U: (2*bi + ri) * 4 + (2*bj + cj)
    # B block (bk, bl) corresponds to rows 2*bk..2*bk+1, cols 2*bl..2*bl+1
    # Index in V: (2*bk + rj) * 4 + (2*bl + ck)
    # C block (cm, cn) corresponds to rows 2*cm..2*cm+1, cols 2*cn..2*cn+1
    # Index in W: (2*cn + ck) * 4 + (2*cm + ri)
    
    # The 7 Strassen terms for 4x4 are:
    # M00 = (A00+A11)(B00+B11) -> Block (0,0) and (1,1) of A, (0,0) and (1,1) of B
    # M01 = (A10+A11)B00 -> Block (1,0) and (1,1) of A, (0,0) of B
    # M02 = A00(B01-B11) -> Block (0,0) of A, (0,1) and (1,1) of B
    # M03 = A11(B10-B00) -> Block (1,1) of A, (1,0) and (0,0) of B
    # M04 = (A00+A01)B11 -> Block (0,0) and (0,1) of A, (1,1) of B
    # M05 = (A10-A00)(B00+B01) -> Block (1,0) and (0,0) of A, (0,0) and (0,1) of B
    # M06 = (A01-A11)(B10+B11) -> Block (0,1) and (1,1) of A, (1,0) and (1,1) of B
    
    # For each of these 7 "super-terms", we use the 7 sub-terms from U2, V2, W2.
    
    # Helper to add a 2x2x2 term scaled and shifted
    def add_2x2_term(U, V, W, U2, V2, W2, col_offset, 
                     a_bi, a_bj, b_bk, b_bl, c_cm, c_cn, 
                     a_sign=1, b_sign=1, c_sign=1):
        # This function is a bit complex because the 2x2 Strassen terms
        # apply to the internal structure of the blocks.
        # We iterate over the 7 sub-terms of the 2x2 Strassen.
        for c_sub in range(7):
            c_global = col_offset + c_sub
            # U2 entries
            for i2 in range(2):
                for j2 in range(2):
                    if U2[i2*2+j2, c_sub] != 0:
                        r = 2*a_bi + i2
                        cc = 2*a_bj + j2
                        U[r*4+cc, c_global] += a_sign * U2[i2*2+j2, c_sub]
            
            # V2 entries
            for j2 in range(2):
                for k2 in range(2):
                    if V2[j2*2+k2, c_sub] != 0:
                        r = 2*b_bk + j2
                        cc = 2*b_bl + k2
                        V[r*4+cc, c_global] += b_sign * V2[j2*2+k2, c_sub]
            
            # W2 entries
            for k2 in range(2):
                for i2 in range(2):
                    if W2[k2*2+i2, c_sub] != 0:
                        r = 2*c_cn + k2
                        cc = 2*c_cm + i2
                        W[r*4+cc, c_global] += c_sign * W2[k2*2+i2, c_sub]

    # Term 0: (A00+A11)(B00+B11)
    # This is sum of A00B00, A00B11, A11B00, A11B11
    # We can just call add_2x2_term for each combination?
    # No, Strassen is linear.
    # (A00+A11) is a sum of blocks.
    # We can add the term for A00B00, A00B11, etc?
    # Actually, it's easier to just add the 4 components.
    # But wait, the 2x2 Strassen inside handles the 2x2 structure.
    # So for "A00 * B00", we use the 2x2 Strassen with a_bi=0, a_bj=0, b_bk=0, b_bl=0, c_cm=0, c_cn=0.
    # For "A00 * B11", we use a_bi=0, a_bj=0, b_bk=1, b_bl=1, c_cm=0, c_cn=1.
    # etc.
    
    # M0 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    add_2x2_term(U4, V4, W4, U2, V2, W2, 0, 0, 0, 0, 0, 0, 0)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 0, 0, 0, 1, 1, 0, 1) # A00 B11 -> C01? No.
    # Wait. The result of A00 B11 contributes to C01.
    # The result of A11 B00 contributes to C10.
    # The result of A11 B11 contributes to C11.
    # So the W indices must be correct.
    
    # Let's restart the mapping.
    # We are computing C = A B.
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # Strassen terms:
    # M0 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M1 = (A10+A11)B00 = A10B00 + A11B00
    # M2 = A00(B01-B11) = A00B01 - A00B11
    # M3 = A11(B10-B00) = A11B10 - A11B00
    # M4 = (A00+A01)B11 = A00B11 + A01B11
    # M5 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
    # M6 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    
    # C00 = M0 - M3 + M5 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 + M4 - M5
    
    # We will add each "block product" (e.g., A00B00) using the 2x2 Strassen.
    # Each block product contributes to a specific C block.
    
    # A00B00 -> C00
    add_2x2_term(U4, V4, W4, U2, V2, W2, 0, 0, 0, 0, 0, 0, 0)
    # A00B11 -> C01 (from M0) and C01 (from M4) and C01 (from M2 with -1)
    # It's easier to just sum the contributions.
    # M0 has A00B11 with coeff 1.
    add_2x2_term(U4, V4, W4, U2, V2, W2, 0, 0, 0, 1, 1, 0, 1)
    # A11B00 -> C10 (from M0) and C10 (from M1) and C10 (from M3 with -1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 0, 1, 1, 0, 0, 1, 0)
    # A11B11 -> C11 (from M0) and C11 (from M4) and C11 (from M6 with -1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 0, 1, 1, 1, 1, 1, 1)
    
    # M1: A10B00 (coeff 1), A11B00 (coeff 1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 7, 1, 0, 0, 0, 1, 0)
    # A11B00 already added? No, we need to add it again with coeff 1 to the M1 slot.
    # The col_offset is different.
    add_2x2_term(U4, V4, W4, U2, V2, W2, 7, 1, 1, 0, 0, 1, 0)
    
    # M2: A00B01 (coeff 1), A00B11 (coeff -1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 14, 0, 0, 0, 1, 0, 1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 14, 0, 0, 1, 1, 0, -1)
    
    # M3: A11B10 (coeff 1), A11B00 (coeff -1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 21, 1, 1, 1, 0, 1, 0)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 21, 1, 1, 0, 0, 1, -1)
    
    # M4: A00B11 (coeff 1), A01B11 (coeff 1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 28, 0, 0, 1, 1, 0, 1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 28, 0, 1, 1, 1, 0, 1)
    
    # M5: A10B00 (coeff 1), A10B01 (coeff 1), A00B00 (coeff -1), A00B01 (coeff -1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 35, 1, 0, 0, 0, 1, 0)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 35, 1, 0, 0, 1, 1, 1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 35, 0, 0, 0, 0, 0, -1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 35, 0, 0, 0, 1, 0, -1)
    
    # M6: A01B10 (coeff 1), A01B11 (coeff 1), A11B10 (coeff -1), A11B11 (coeff -1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 42, 0, 1, 1, 0, 0, 1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 42, 0, 1, 1, 1, 0, 1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 42, 1, 1, 1, 0, 1, -1)
    add_2x2_term(U4, V4, W4, U2, V2, W2, 42, 1, 1, 1, 1, 1, -1)
    
    # 3. Schoolbook for 4x4x3 (columns 4,5,6)
    U_rest = np.zeros((16, 48))
    V_rest = np.zeros((28, 48)) # p=7, so 4*7=28
    W_rest = np.zeros((28, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                k = k_prime + 4
                U_rest[i*4+j, col] = 1
                V_rest[j*7+k, col] = 1
                W_rest[k*4+i, col] = 1
                col += 1
                
    # 4. Combine
    U_final = np.hstack([U4, U_rest])
    V_final = np.vstack([V4, V_rest]) # V4 is 16x49, V_rest is 28x48. 
    # Wait, V4 only has entries for k=0..3. 
    # V4 shape is (16, 49). 
    # We need V_final to be (28, 97).
    # V_final[0:16, 0:49] = V4
    # V_final[16:28, 0:49] = 0
    # V_final[:, 49:97] = V_rest
    
    V_final = np.zeros((28, 97))
    V_final[0:16, 0:49] = V4
    V_final[:, 49:97] = V_rest
    
    W_final = np.hstack([W4, W_rest])
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END

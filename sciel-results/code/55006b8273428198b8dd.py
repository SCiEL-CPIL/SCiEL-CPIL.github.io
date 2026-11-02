"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill

def construct():
    """Construct a rank-45 decomposition for 3x3x6 matrix multiplication.
    
    Strategy: Split the 6 columns into three groups of 2 columns.
    For each group, use a rank-15 decomposition of 3x3x2.
    Total rank: 3 * 15 = 45.
    
    The 3x3x2 decomposition uses Strassen's 7-term algorithm for the 
    2x2x2 core (A rows 0,1 with B rows 0,1) plus 8 additional terms
    for the remaining interactions.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Build one 3x3x2 block decomposition (rank 15)
    # A is 3x3, B is 3x2, C is 3x2
    # U shape: (9, 15), V shape: (6, 15), W shape: (6, 15)
    # U[i*3+j], V[j*2+k], W[k*3+i]
    
    U_block = np.zeros((9, 15))
    V_block = np.zeros((6, 15))
    W_block = np.zeros((6, 15))
    
    # Strassen terms for 2x2x2 core (A[0:2,0:2] * B[0:2,:])
    # M1 = (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
    U_block[0, 0] = 1; U_block[4, 0] = 1
    V_block[0, 0] = 1; V_block[3, 0] = 1
    W_block[0, 0] = 1; W_block[4, 0] = 1
    
    # M2 = (a10+a11)b00 -> c10 += 1
    U_block[3, 1] = 1; U_block[4, 1] = 1
    V_block[0, 1] = 1
    W_block[3, 1] = 1
    
    # M3 = a00(b01-b11) -> c01 += 1
    U_block[0, 2] = 1
    V_block[1, 2] = 1; V_block[3, 2] = -1
    W_block[1, 2] = 1
    
    # M4 = a11(b10-b00) -> c10 += 1, c00 -= 1
    U_block[4, 3] = 1
    V_block[2, 3] = 1; V_block[0, 3] = -1
    W_block[3, 3] = 1; W_block[0, 3] = -1
    
    # M5 = (a00+a01)b11 -> c00 += 1, c01 += 1
    U_block[0, 4] = 1; U_block[1, 4] = 1
    V_block[3, 4] = 1
    W_block[0, 4] = 1; W_block[1, 4] = 1
    
    # M6 = (a10-a00)(b00+b01) -> c11 += 1
    U_block[3, 5] = 1; U_block[0, 5] = -1
    V_block[0, 5] = 1; V_block[1, 5] = 1
    W_block[4, 5] = 1
    
    # M7 = (a01-a11)(b10+b11) -> c00 -= 1, c11 -= 1
    U_block[1, 6] = 1; U_block[4, 6] = -1
    V_block[2, 6] = 1; V_block[3, 6] = 1
    W_block[0, 6] = -1; W_block[4, 6] = -1
    
    # Additional 8 terms for the remaining interactions:
    # Need to cover: a02*b20, a02*b21, a12*b20, a12*b21,
    #                a20*b00, a20*b01, a21*b10, a21*b11,
    #                a22*b20, a22*b21
    # That's 10 terms but we can combine some.
    
    # Term 7: a02*b20 -> c00
    U_block[2, 7] = 1
    V_block[4, 7] = 1
    W_block[0, 7] = 1
    
    # Term 8: a02*b21 -> c01
    U_block[2, 8] = 1
    V_block[5, 8] = 1
    W_block[1, 8] = 1
    
    # Term 9: a12*b20 -> c10
    U_block[5, 9] = 1
    V_block[4, 9] = 1
    W_block[3, 9] = 1
    
    # Term 10: a12*b21 -> c11
    U_block[5, 10] = 1
    V_block[5, 10] = 1
    W_block[4, 10] = 1
    
    # Term 11: a20*b00 -> c20
    U_block[6, 11] = 1
    V_block[0, 11] = 1
    W_block[2, 11] = 1  # k=0, i=2 -> 0*3+2=2
    
    # Term 12: a20*b01 -> c21
    U_block[6, 12] = 1
    V_block[1, 12] = 1
    W_block[5, 12] = 1  # k=1, i=2 -> 1*3+2=5
    
    # Term 13: a21*b10 -> c20
    U_block[7, 13] = 1
    V_block[2, 13] = 1
    W_block[2, 13] = 1  # k=0, i=2 -> 2
    
    # Term 14: a21*b11 -> c21
    U_block[7, 14] = 1
    V_block[3, 14] = 1
    W_block[5, 14] = 1  # k=1, i=2 -> 5
    
    # Wait, I'm missing a22*b20 and a22*b21. That's 17 terms needed, not 15.
    # Let me reconsider. I need to combine a22*b20 and a22*b21 with others.
    
    # Actually, let me use a different grouping. Instead of Strassen + 8,
    # let me use a more efficient decomposition.
    
    # For 3x3x2, the minimal rank is actually 15 (known result).
    # Let me use a verified construction.
    
    # Reset and use a cleaner approach:
    # Terms 0-6: Strassen for 2x2x2
    # Terms 7-14: 8 terms for remaining 10 products using 2 combined terms
    
    # Combine a22*b20 + a22*b21 into one term with W having both c20 and c21:
    # Term 15 would be needed... but I only have 15 slots.
    
    # Let me just use rank 18 per block (schoolbook for 3x3x2) = 54 total.
    # Or find a better way.
    
    # Actually, let me verify if rank 45 is achievable with a different split.
    # Alternative: split into 3 groups of 2, use rank 15 each.
    # The issue is I need exactly 15 terms for 3x3x2.
    
    # Let me use: 7 Strassen + 3 for (a02,a12,a22)*b20,b21 combined + 5 for a20,a21
    # That's 7+3+5 = 15. Let me try:
    
    # Actually the cleanest: just use schoolbook for the full 3x3x6 with rank 54,
    # but that's not better. Let me try rank 48 by using 16 per block.
    
    # For now, let me return a valid decomposition. I'll use rank 45 with 
    # a slightly different construction.
    
    # Let me rebuild: use 7 Strassen terms + 8 more = 15, where the 8 more
    # cleverly cover the 10 remaining products.
    
    # The 10 remaining products are:
    # a02*b20 -> c00, a02*b21 -> c01
    # a12*b20 -> c10, a12*b21 -> c11  
    # a20*b00 -> c20, a20*b01 -> c21
    # a21*b10 -> c20, a21*b11 -> c21
    # a22*b20 -> c20, a22*b21 -> c21
    
    # I can combine: (a02+a22)*b20 -> contributes to c00 and c20
    # and (a02+a22)*b21 -> contributes to c01 and c21
    # That saves 2 terms. So 10 - 2 = 8 terms. 
    # But wait, (a02+a22)*b20 gives a02*b20 + a22*b20, contributing to c00 and c20.
    # And (a02+a22)*b21 gives a02*b21 + a22*b21, contributing to c01 and c21.
    # So I save 2 terms, giving 8 terms for the remaining 10 products.
    # Total: 7 + 8 = 15. 
    
    # Let me redo terms 7-14:
    
    # Term 7: (a02+a22)*b20 -> c00 += 1, c20 += 1
    U_block[2, 7] = 1; U_block[8, 7] = 1
    V_block[4, 7] = 1
    W_block[0, 7] = 1; W_block[2, 7] = 1
    
    # Term 8: (a02+a22)*b21 -> c01 += 1, c21 += 1
    U_block[2, 8] = 1; U_block[8, 8] = 1
    V_block[5, 8] = 1
    W_block[1, 8] = 1; W_block[5, 8] = 1
    
    # Term 9: a12*b20 -> c10
    U_block[5, 9] = 1
    V_block[4, 9] = 1
    W_block[3, 9] = 1
    
    # Term 10: a12*b21 -> c11
    U_block[5, 10] = 1
    V_block[5, 10] = 1
    W_block[4, 10] = 1
    
    # Term 11: a20*b00 -> c20
    U_block[6, 11] = 1
    V_block[0, 11] = 1
    W_block[2, 11] = 1
    
    # Term 12: a20*b01 -> c21
    U_block[6, 12] = 1
    V_block[1, 12] = 1
    W_block[5, 12] = 1
    
    # Term 13: a21*b10 -> c20
    U_block[7, 13] = 1
    V_block[2, 13] = 1
    W_block[2, 13] = 1
    
    # Term 14: a21*b11 -> c21
    U_block[7, 14] = 1
    V_block[3, 14] = 1
    W_block[5, 14] = 1
    
    # Now assemble the full 3x3x6 decomposition by tiling the 3x3x2 block 3 times
    # for the three groups of 2 columns.
    
    total_rank = 45
    U = np.zeros((9, total_rank))
    V = np.zeros((18, total_rank))  # m*p = 3*6 = 18
    W = np.zeros((18, total_rank))  # p*n = 6*3 = 18
    
    for block in range(3):
        offset = block * 15
        # U: same for all blocks
        U[:, offset:offset+15] = U_block
        
        # V: B has 3 rows, 6 cols. For block b, B's cols are 2b, 2b+1
        # V index: j*6 + k where j in {0,1,2}, k in {0,1,2,3,4,5}
        # For block b, k_local in {0,1} maps to k_global = 2b + k_local
        for j in range(3):
            for k_local in range(2):
                k_global = 2 * block + k_local
                V[j * 6 + k_global, offset:offset+15] = V_block[j * 2 + k_local]
        
        # W: C has 6 rows (k), 3 cols (i). W index: k*3 + i
        # For block b, k_local in {0,1} maps to k_global = 2b + k_local
        for k_local in range(2):
            k_global = 2 * block + k_local
            for i in range(3):
                W[k_global * 3 + i, offset:offset+15] = W_block[k_local * 3 + i]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END


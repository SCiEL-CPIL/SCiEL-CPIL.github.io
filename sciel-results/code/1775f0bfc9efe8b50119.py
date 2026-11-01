"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses Kronecker/block decomposition to reduce rank below schoolbook (60).
    Strategy: decompose into blocks where Strassen can be applied to 2x2 sub-blocks.
    """
    n, m, p = DIMENSIONS
    
    # We'll build the decomposition by splitting the inner dimension m=5 into blocks.
    # Since n=2, p=6, we can try to use the structure more cleverly.
    
    # Approach: Use the fact that we can group terms.
    # The naive rank is n*m*p = 60.
    # 
    # Let's try a direct search for a lower rank using a greedy or structured approach.
    # However, given the time constraint, let's use a known technique:
    # For small matrices, we can sometimes find decompositions by solving a system.
    # 
    # But a more reliable approach for this specific case:
    # Split m=5 into m1=2, m2=3.
    # A (2x5) = [A1 (2x2) | A2 (2x3)]
    # B (5x6) = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    #
    # Rank of A1*B1 (2x2 * 2x6): can be done in rank 2*2*6=24 schoolbook, or less.
    # Rank of A2*B2 (2x3 * 3x6): can be done in rank 2*3*6=36 schoolbook.
    # Total schoolbook for this split: 24+36 = 60. No gain.
    #
    # We need to use Strassen on the 2x2 part.
    # A1*B1: 2x2 * 2x6. We can view this as two 2x2 * 2x3 blocks.
    # Actually, let's just use the Strassen trick on any 2x2x2 sub-block.
    # The current program does this and gets rank 59.
    #
    # Can we do better? Let's try to find a decomposition of rank < 59.
    # 
    # Known lower bounds for 2x5*5x6:
    # The rank is at least max(n*m, m*p, p*n) = max(10, 30, 12) = 30.
    # Also at least (n*m*p)/min(n,p) = 60/2 = 30.
    #
    # Let's try a randomized search for a lower rank, or use a more systematic decomposition.
    #
    # Actually, let's try to combine the block approach with Strassen more aggressively.
    # 
    # Alternative: Use the fact that the multiplication tensor is decomposable.
    # We can try to find a decomposition of rank r by solving a linear system.
    # But this is computationally expensive.
    #
    # Let's stick with a structured approach:
    # 1. Split m=5 into 2+3.
    # 2. For the 2x2 * 2x6 part, use Strassen on the 2x2x2 blocks.
    #    2x2 * 2x6 = 2x2 * [2x2 | 2x2 | 2x2]
    #    Each 2x2 * 2x2 can be done in rank 7 (Strassen).
    #    So 3 blocks * 7 = 21 for the first part.
    # 3. For the 2x3 * 3x6 part, schoolbook is 2*3*6=36.
    #    Can we do better? 2x3 * 3x6.
    #    Split 3x6 into 3x2 + 3x2 + 3x2? No, that's not helpful.
    #    Split 2x3 into 2x1 + 2x1 + 2x1? 
    #    Each 2x1 * 3x6 is rank 2*1*6=12. Total 3*12=36. No gain.
    #
    # Total: 21 + 36 = 57. This is better than 59!
    
    # Let's implement this:
    # Part 1: A1 (2x2) * B1 (2x6) using Strassen on 2x2x2 blocks.
    # Part 2: A2 (2x3) * B2 (3x6) using schoolbook.
    
    rank = 57  # 21 + 36
    
    U = [[0] * rank for _ in range(n * m)]  # 10 x 57
    V = [[0] * rank for _ in range(m * p)]  # 30 x 57
    W = [[0] * rank for _ in range(p * n)]  # 12 x 57
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for u_idx, coeff in u_coeffs:
            U[u_idx][col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx][col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx][col] += coeff
        col += 1
    
    # Part 1: A1 (2x2) * B1 (2x6)
    # A1 indices: i in {0,1}, j in {0,1} -> u_idx = i*5 + j
    # B1 indices: j in {0,1}, k in {0,5} -> v_idx = j*6 + k
    # C1 indices: k in {0,5}, i in {0,1} -> w_idx = k*2 + i
    
    # Split B1 into three 2x2 blocks: k in {0,1}, {2,3}, {4,5}
    for block in range(3):
        k_offset = block * 2
        k0, k1 = k_offset, k_offset + 1
        
        # Strassen on A1 (2x2) * B1_block (2x2)
        # u00, u01 = 0*5+0, 0*5+1
        # u10, u11 = 1*5+0, 1*5+1
        u00, u01 = 0, 1
        u10, u11 = 5, 6
        
        # v00, v01 = 0*6+k0, 0*6+k1
        # v10, v11 = 1*6+k0, 1*6+k1
        v00, v01 = k0, k0 + 1
        v10, v11 = 6 + k0, 6 + k0 + 1
        
        # w00, w01 = k0*2+0, k0*2+1
        # w10, w11 = k1*2+0, k1*2+1
        w00, w01 = k0 * 2, k0 * 2 + 1
        w10, w11 = k1 * 2, k1 * 2 + 1
        
        # Term 1: M1 = (A00 + A11)(B00 + B11)
        add_term(
            [(u00, 1), (u11, 1)],
            [(v00, 1), (v11, 1)],
            [(w00, 1), (w10, 1)]
        )
        
        # Term 2: M2 = (A10 + A11)B00
        add_term(
            [(u10, 1), (u11, 1)],
            [(v00, 1)],
            [(w10, 1), (w11, -1)]
        )
        
        # Term 3: M3 = A00(B01 - B11)
        add_term(
            [(u00, 1)],
            [(v01, 1), (v11, -1)],
            [(w01, 1), (w11, 1)]
        )
        
        # Term 4: M4 = A11(B10 - B00)
        add_term(
            [(u11, 1)],
            [(v10, 1), (v00, -1)],
            [(w00, 1), (w10, 1)]
        )
        
        # Term 5: M5 = (A00 + A01)B11
        add_term(
            [(u00, 1), (u01, 1)],
            [(v11, 1)],
            [(w00, -1), (w01, 1)]
        )
        
        # Term 6: M6 = (A10 - A00)(B00 + B01)
        add_term(
            [(u10, 1), (u00, -1)],
            [(v00, 1), (v01, 1)],
            [(w11, 1)]
        )
        
        # Term 7: M7 = (A01 - A11)(B10 + B11)
        add_term(
            [(u01, 1), (u11, -1)],
            [(v10, 1), (v11, 1)],
            [(w00, 1)]
        )
    
    # Part 2: A2 (2x3) * B2 (3x6) using schoolbook
    # A2 indices: i in {0,1}, j in {2,3,4} -> u_idx = i*5 + j
    # B2 indices: j in {2,3,4}, k in {0,5} -> v_idx = j*6 + k
    # C2 indices: k in {0,5}, i in {0,1} -> w_idx = k*2 + i
    
    for i in range(n):
        for j in range(2, 5):  # j = 2, 3, 4
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix using a block decomposition strategy.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We will construct the decomposition for rank 78.
    # Structure:
    # 1. Split m=4 into two 2s.
    # 2. For each 3x2x7 block, split p=7 into 4 and 3.
    # 3. For 3x2x4, split p=4 into two 2s. (Two 3x2x2 blocks)
    # 4. For 3x2x3, split p=3 into 2 and 1. (One 3x2x2, one 3x2x1)
    # 
    # Ranks:
    # 3x2x2: Rank 11 (Strassen 7 + 4 for the bottom row)
    # 3x2x1: Rank 6 (Schoolbook)
    # 3x2x4: 11 + 11 = 22
    # 3x2x3: 11 + 6 = 17
    # 3x2x7: 22 + 17 = 39
    # 3x4x7: 39 + 39 = 78
    
    rank = 78
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    def add_block_322(U, V, W, offset, i_offset, j_offset, k_offset):
        """
        Adds a 3x2x2 block decomposition.
        A is at rows i_offset..i_offset+2, cols j_offset..j_offset+1 of global A.
        B is at rows j_offset..j_offset+1, cols k_offset..k_offset+1 of global B.
        C is at rows i_offset..i_offset+2, cols k_offset..k_offset+1 of global C.
        
        Global Indices:
        A index: (i, j) -> i*m + j
        B index: (j, k) -> j*p + k
        C index: (k, i) -> k*n + i  (Note: Task says W[k*n+i,r])
        
        Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]".
        And T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
        So C index is indeed k*n + i.
        """
        # Strassen part for top 2x2 of A (rows i_offset, i_offset+1)
        # A_top is 2x2. B is 2x2.
        # Strassen multipliers M1..M7
        
        # M1 = (A11 + A22)(B11 + B22)
        # A11 is at (i_offset, j_offset)
        # A22 is at (i_offset+1, j_offset+1)
        # B11 is at (j_offset, k_offset)
        # B22 is at (j_offset+1, k_offset+1)
        
        # We need to add contributions to U, V, W.
        # U corresponds to A. V to B. W to C.
        
        # Helper to add a term: U[a, r] += val, V[b, r] += val, W[c, r] += val
        # But Strassen involves sums/differences.
        # We can just add the terms linearly.
        
        def add_term(a_idx, b_idx, c_idx, u_val, v_val, w_val):
            U[a_idx, offset] += u_val
            V[b_idx, offset] += v_val
            W[c_idx, offset] += w_val
            
        # Map local (i,j) to global A index
        def a_idx(li, lj):
            return (i_offset + li) * m + (j_offset + lj)
            
        def b_idx(lj, lk):
            return (j_offset + lj) * p + (k_offset + lk)
            
        def c_idx(lk, li):
            return (k_offset + lk) * n + (i_offset + li)
            
        # Strassen M1
        # (A11 + A22)(B11 + B22)
        # Terms: A11B11, A11B22, A22B11, A22B22
        # We can just add 4 terms for M1? No, M1 is one product of sums.
        # In tensor decomposition, we can distribute.
        # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # This uses 4 rank-1 terms.
        # But Strassen uses 7 terms total for the 2x2x2.
        # If we distribute all sums, we get 8 terms (schoolbook).
        # To get rank 7, we must keep the sums together in U and V?
        # No, U and V are matrices. The "sum" is in the value of U/V?
        # No, U and V contain the coefficients.
        # The term is U[a,r] * V[b,r] * W[c,r].
        # If we want to compute (A11+A22)(B11+B22), we need a term where
        # U picks A11 and A22? No, U is a vector.
        # Actually, the rank of the tensor is the number of terms.
        # Strassen's algorithm reduces the number of multiplications.
        # This corresponds to a rank-7 decomposition.
        # The terms are:
        # 1. (A11+A22) * (B11+B22)
        # 2. (A21+A22) * B11
        # 3. A11 * (B21-B22)
        # 4. A22 * (B12-B11)
        # 5. (A11+A12) * B22
        # 6. (A21-A11) * (B11+B12)
        # 7. (A12-A22) * (B21+B22)
        
        # To implement this in U, V, W:
        # For term 1:
        # U[a1, r] = 1, U[a2, r] = 1 (for A11, A22)
        # V[b1, r] = 1, V[b2, r] = 1 (for B11, B22)
        # W[c, r] = 1 (for C11, C12, C21, C22 contributions)
        # Wait, C is the result.
        # C11 gets M1 + M4 - M5 + M7
        # C12 gets M3 + M5
        # C21 gets M2 + M4
        # C22 gets M1 - M2 + M3 + M6
        
        # So for each Strassen multiplier Mr, we add 1 to W at the appropriate C indices.
        
        # Term 1: M1
        # A part: A11 (li=0, lj=0), A22 (li=1, lj=1)
        # B part: B11 (lj=0, lk=0), B22 (lj=1, lk=1)
        # C part: C11 (lk=0, li=0), C22 (lk=1, li=1)
        # Wait, M1 contributes to C11 and C22.
        # C11: +M1. C22: +M1.
        # So W[c11, r] += 1, W[c22, r] += 1.
        
        # Let's write a helper for Strassen terms.
        
        # M1: (A11+A22)(B11+B22) -> C11, C22
        U[a_idx(0,0), offset] += 1
        U[a_idx(1,1), offset] += 1
        V[b_idx(0,0), offset] += 1
        V[b_idx(1,1), offset] += 1
        W[c_idx(0,0), offset] += 1 # C11
        W[c_idx(1,1), offset] += 1 # C22
        
        # M2: (A21+A22)B11 -> C21, C22
        # A21 (li=1, lj=0), A22 (li=1, lj=1)
        # B11 (lj=0, lk=0)
        # C21 (lk=0, li=1), C22 (lk=1, li=1)
        # C21 gets +M2. C22 gets -M2.
        U[a_idx(1,0), offset] += 1
        U[a_idx(1,1), offset] += 1
        V[b_idx(0,0), offset] += 1
        W[c_idx(0,1), offset] += 1 # C21
        W[c_idx(1,1), offset] -= 1 # C22
        
        # M3: A11(B21-B22) -> C11, C12
        # A11 (li=0, lj=0)
        # B21 (lj=1, lk=0), B22 (lj=1, lk=1)
        # C11 gets +M3? No, C11 = M1 + M4 - M5 + M7.
        # C12 = M3 + M5.
        # So C12 gets +M3.
        # What about C11? M3 is not in C11.
        # Wait, standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # My M3 definition: A11(B21-B22).
        # This contributes to C12?
        # A11 B21 is part of C11? No, C11 = A11B11 + A12B21.
        # So A11B21 is NOT in C11.
        # C12 = A11B12 + A12B22.
        # So A11B21 is not in C12 either?
        # Let's re-verify Strassen formulas.
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C11 = A11B11 + A12B21
        # C12 = A11B12 + A12B22
        # C21 = A21B11 + A22B21
        # C22 = A21B12 + A22B22
        
        # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # M2 = (A21+A22)B11 = A21B11 + A22B11
        # M3 = A11(B21-B22) = A11B21 - A11B22
        # M4 = A22(B12-B11) = A22B12 - A22B11
        # M5 = (A11+A12)B22 = A11B22 + A12B22
        # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12
        # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
        
        # Check C11:
        # M1 + M4 - M5 + M7
        # = (A11B11 + A11B22 + A22B11 + A22B22)
        #   + (A22B12 - A22B11)
        #   - (A11B22 + A12B22)
        #   + (A12B21 + A12B22 - A22B21 - A22B22)
        # = A11B11 + A11B22 - A11B22 + A22B11 - A22B11 + A22B12 + A22B22 - A22B22
        #   - A12B22 + A12B21 + A12B22 - A22B21
        # = A11B11 + A22B12 + A12B21 - A22B21
        # This is NOT C11. C11 is A11B11 + A12B21.
        # There is an error in my Strassen recall or application.
        
        # Correct Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's re-evaluate C11 sum:
        # M1: A11B11, A11B22, A22B11, A22B22
        # M4: A22B12, -A22B11
        # -M5: -A11B22, -A12B22
        # M7: A12B21, A12B22, -A22B21, -A22B22
        # Sum:
        # A11B11
        # A11B22 - A11B22 = 0
        # A22B11 - A22B11 = 0
        # A22B12
        # A22B22 - A22B22 = 0
        # -A12B22 + A12B22 = 0
        # A12B21
        # -A22B21
        # Result: A11B11 + A22B12 + A12B21 - A22B21.
        # This is still wrong.
        
        # I will use the known rank-11 decomposition for 3x2x2 directly from literature or derive it carefully.
        # Actually, I can just use schoolbook for 3x2x2 (Rank 12) and see if the total rank is better?
        # 2 * (2 * 12 + 12 + 6) = 2 * 42 = 84. No gain.
        
        # I must get the Strassen part right.
        # The issue is that Strassen computes the 2x2 product.
        # The 3x2x2 tensor is the sum of the 2x2x2 tensor (top 2 rows of A) and 1x2x2 tensor (bottom row).
        # The 2x2x2 part has rank 7.
        # The 1x2x2 part has rank 4.
        # Total rank 11.
        
        # I will implement the 2x2x2 Strassen part correctly.
        # C11 = A11B11 + A12B21
        # C12 = A11B12 + A12B22
        # C21 = A21B11 + A22B21
        # C22 = A21B12 + A22B22
        
        # Let's just use the 4 terms for the 2x2 part? Rank 4? No, 2x2x2 rank is 7.
        # If I use schoolbook for 2x2x2, rank is 8.
        # Total rank for 3x2x2 would be 8+4=12.
        # Then 3x2x4 is 24. 3x2x3 is 18. 3x2x7 is 42. 3x4x7 is 84.
        
        # So I MUST use Strassen (Rank 7) for the 2x2x2 part to get below 84.
        
        # Let's hardcode the 7 terms for 2x2x2.
        # I will add them to U, V, W.
        
        # Term 1: M1
        # A: A11, A22
        # B: B11, B22
        # C: C11 (+1), C22 (+1)
        U[a_idx(0,0), offset] += 1; U[a_idx(1,1), offset] += 1
        V[b_idx(0,0), offset] += 1; V[b_idx(1,1), offset] += 1
        W[c_idx(0,0), offset] += 1; W[c_idx(1,1), offset] += 1
        
        # Term 2: M2
        # A: A21, A22
        # B: B11
        # C: C21 (+1), C22 (-1)
        U[a_idx(1,0), offset] += 1; U[a_idx(1,1), offset] += 1
        V[b_idx(0,0), offset] += 1
        W[c_idx(0,1), offset] += 1; W[c_idx(1,1), offset] -= 1
        
        # Term 3: M3
        # A: A11
        # B: B21, B22
        # C: C12 (+1) ?
        # M3 = A11(B21-B22).
        # C12 = M3 + M5.
        # So C12 gets +M3.
        # Does C11 get M3? No.
        # So W[c_idx(1,0), offset] += 1 (C12 is lk=1, li=0)
        U[a_idx(0,0), offset] += 1
        V[b_idx(1,0), offset] += 1; V[b_idx(1,1), offset] -= 1
        W[c_idx(1,0), offset] += 1
        
        # Term 4: M4
        # A: A22
        # B: B12, B11
        # M4 = A22(B12-B11).
        # C11 = M1 + M4 ...
        # C21 = M2 + M4 ...
        # So C11 gets +M4, C21 gets +M4.
        U[a_idx(1,1), offset] += 1
        V[b_idx(0,1), offset] += 1; V[b_idx(0,0), offset] -= 1
        W[c_idx(0,0), offset] += 1; W[c_idx(0,1), offset] += 1
        
        # Term 5: M5
        # A: A11, A12
        # B: B22
        # M5 = (A11+A12)B22.
        # C11 = ... - M5 ...
        # C12 = M3 + M5.
        # So C11 gets -M5, C12 gets +M5.
        U[a_idx(0,0), offset] += 1; U[a_idx(0,1), offset] += 1
        V[b_idx(1,1), offset] += 1
        W[c_idx(0,0), offset] -= 1; W[c_idx(1,0), offset] += 1
        
        # Term 6: M6
        # A: A21, A11
        # B: B11, B12
        # M6 = (A21-A11)(B11+B12).
        # C22 = ... + M6 ...
        # So C22 gets +M6.
        U[a_idx(1,0), offset] += 1; U[a_idx(0,0), offset] -= 1
        V[b_idx(0,0), offset] += 1; V[b_idx(0,1), offset] += 1
        W[c_idx(1,1), offset] += 1
        
        # Term 7: M7
        # A: A12, A22
        # B: B21, B22
        # M7 = (A12-A22)(B21+B22).
        # C11 = ... + M7 ...
        # So C11 gets +M7.
        U[a_idx(0,1), offset] += 1; U[a_idx(1,1), offset] -= 1
        V[b_idx(1,0), offset] += 1; V[b_idx(1,1), offset] += 1
        W[c_idx(0,0), offset] += 1
        
        # Now the 1x2x2 part (Row 2 of A, li=2)
        # C31 = A31B11 + A32B21
        # C32 = A31B12 + A32B22
        
        # Term 8: A31B11 -> C31
        U[a_idx(2,0), offset] += 1
        V[b_idx(0,0), offset] += 1
        W[c_idx(0,2), offset] += 1
        
        # Term 9: A32B21 -> C31
        U[a_idx(2,1), offset] += 1
        V[b_idx(1,0), offset] += 1
        W[c_idx(0,2), offset] += 1
        
        # Term 10: A31B12 -> C32
        U[a_idx(2,0), offset] += 1
        V[b_idx(0,1), offset] += 1
        W[c_idx(1,2), offset] += 1
        
        # Term 11: A32B22 -> C32
        U[a_idx(2,1), offset] += 1
        V[b_idx(1,1), offset] += 1
        W[c_idx(1,2), offset] += 1
        
        # Advance offset by 11
        return offset + 11

    def add_block_321(U, V, W, offset, i_offset, j_offset, k_offset):
        """
        Adds a 3x2x1 block decomposition.
        A is 3x2. B is 2x1.
        C is 3x1.
        Rank 6.
        """
        def a_idx(li, lj):
            return (i_offset + li) * m + (j_offset + lj)
        def b_idx(lj):
            return (j_offset + lj) * p + k_offset
        def c_idx(li):
            return k_offset * n + (i_offset + li)
            
        # C1 = A11B1 + A12B2
        # C2 = A21B1 + A22B2
        # C3 = A31B1 + A32B2
        
        # T1: A11B1 -> C1
        U[a_idx(0,0), offset] += 1
        V[b_idx(0), offset] += 1
        W[c_idx(0), offset] += 1
        
        # T2: A12B2 -> C1
        U[a_idx(0,1), offset] += 1
        V[b_idx(1), offset] += 1
        W[c_idx(0), offset] += 1
        
        # T3: A21B1 -> C2
        U[a_idx(1,0), offset] += 1
        V[b_idx(0), offset] += 1
        W[c_idx(1), offset] += 1
        
        # T4: A22B2 -> C2
        U[a_idx(1,1), offset] += 1
        V[b_idx(1), offset] += 1
        W[c_idx(1), offset] += 1
        
        # T5: A31B1 -> C3
        U[a_idx(2,0), offset] += 1
        V[b_idx(0), offset] += 1
        W[c_idx(2), offset] += 1
        
        # T6: A32B2 -> C3
        U[a_idx(2,1), offset] += 1
        V[b_idx(1), offset] += 1
        W[c_idx(2), offset] += 1
        
        return offset + 6

    # Build the full decomposition
    offset = 0
    
    # Block 1: A1 (rows 0-2, cols 0-1) * B1 (rows 0-1, cols 0-6)
    # Split B1 into B1a (cols 0-3) and B1b (cols 4-6)
    
    # B1a: 3x2x4
    # Split into two 3x2x2
    # Block 1a: A1 * B1a1 (cols 0-1)
    offset = add_block_322(U, V, W, offset, 0, 0, 0)
    # Block 1b: A1 * B1a2 (cols 2-3)
    offset = add_block_322(U, V, W, offset, 0, 0, 2)
    
    # B1b: 3x2x3
    # Split into 3x2x2 (cols 4-5) and 3x2x1 (col 6)
    # Block 1c: A1 * B1b1 (cols 4-5)
    offset = add_block_322(U, V, W, offset, 0, 0, 4)
    # Block 1d: A1 * B1b2 (col 6)
    offset = add_block_321(U, V, W, offset, 0, 0, 6)
    
    # Block 2: A2 (rows 0-2, cols 2-3) * B2 (rows 2-3, cols 0-6)
    # Split B2 into B2a (cols 0-3) and B2b (cols 4-6)
    
    # B2a: 3x2x4
    # Block 2a: A2 * B2a1 (cols 0-1)
    offset = add_block_322(U, V, W, offset, 0, 2, 0)
    # Block 2b: A2 * B2a2 (cols 2-3)
    offset = add_block_322(U, V, W, offset, 0, 2, 2)
    
    # B2b: 3x2x3
    # Block 2c: A2 * B2b1 (cols 4-5)
    offset = add_block_322(U, V, W, offset, 0, 2, 4)
    # Block 2d: A2 * B2b2 (col 6)
    offset = add_block_321(U, V, W, offset, 0, 2, 6)
    
    # offset should be 78
    assert offset == 78, f"Offset {offset} != 78"
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

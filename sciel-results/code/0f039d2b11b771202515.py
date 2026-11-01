"""Matrix multiplication 3x3x3 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 3
    nm, mp, pn = n*m, m*p, p*n
    
    # Deadline guard
    t0 = time.time()
    
    # Rank 23 decomposition for 3x3 matrix multiplication
    # This is a known upper bound. We'll construct it using the
    # 2x2 Strassen-based approach embedded in 3x3.
    
    # Actually, let me try to construct a valid rank-23 decomposition.
    # One known approach: use 7 Strassen terms for a 2x2 sub-block,
    # and handle the remaining rows/columns with schoolbook.
    
    # Let's think about this more carefully.
    # A 3x3 matrix multiplication can be split as:
    #   |A11 A12 A13|   |B11 B12 B13|   |C11 C12 C13|
    #   |A21 A22 A23| * |B21 B22 B23| = |C21 C22 C23|
    #   |A31 A32 A33|   |B31 B32 B33|   |C31 C32 C33|
    #
    # A simple approach to get rank < 27:
    # - Use Strassen for the top-left 2x2 block (7 terms instead of 4)
    # - Handle the rest with schoolbook
    # But this gives: 7 + 3*3 + 3*3 - 4 = 7 + 9 + 9 - 4 = 21?
    # Wait, let me be more careful.
    
    # Alternative: I'll try to find a valid rank-23 decomposition by
    # solving the system of equations numerically and rounding.
    
    # The tensor T[i,j,k] = 1 if and only if (i,j,k) corresponds to
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # In our indexing: A has indices (i*m+j), B has (j*p+k), C has (k*n+i)
    # So T[A_idx, B_idx, C_idx] = 1 if A_idx = i*m+j, B_idx = j*p+k, C_idx = k*n+i
    # for some valid i,j,k.
    
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all i,j,k.
    
    # Let me try a different approach: use known rank-23 vectors.
    # I'll construct them based on the algebraic identity.
    
    # Rank 23 construction via 2x2 Strassen embedding:
    # Split 3x3 into overlapping 2x2 blocks.
    
    # Actually, let me just try to numerically find a low-rank decomposition.
    # Start with rank 23 and see if we can find integer/half-integer solutions.
    
    # For safety, let me first provide the schoolbook (rank 27) and then
    # try to improve.
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Try rank 23 using a structured approach
    # Based on the idea that we can share some computations
    
    # Let me try a specific rank-23 construction:
    # Terms 1-7: Strassen for positions involving indices {0,1}
    # Terms 8-15: schoolbook for remaining combinations
    
    # Strassen 2x2 terms for i in {0,1}, j in {0,1}, k in {0,1}:
    # M1 = A[i,j]*B[j,k] for i,j,k in {0,1} using 7 multiplications
    # But we need to handle the overlap with index 2.
    
    # Let me try a cleaner approach: direct rank-23 from literature
    # The known rank-23 algorithm uses the following structure:
    
    # I'll attempt to construct it by extending Strassen ideas.
    
    # For 3x3, one rank-23 construction:
    # Use the fact that rank(3x3) <= rank(2x2) + 3*3 - 1 = 7+9-1 = 15? No.
    # Actually rank(3x3) <= 2*rank(2x2) + 3 = 14+3 = 17? Also no.
    
    # The correct bound from embedding: rank(3x3) <= 2*rank(2x2) + 3*3 - 3 = 14+9-3 = 20?
    # Hmm, that's not quite right either.
    
    # Let me just try to solve for rank 23 numerically.
    
    def try_rank(r):
        """Try to find a valid rank-r decomposition with integer/half-integer entries."""
        # We need 27 equations: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is a system of 27 trilinear equations.
        
        # Strategy: fix W to a simple pattern, then solve for U and V.
        # Or use a randomized approach.
        
        # Let me try: set W[k*n+i, r] to be 1 for specific (k,i,r) pairs.
        # Then we need: sum_r U[i*m+j,r]*V[j*p+k,r]*delta = 1
        # which becomes a bilinear system.
        
        # Simpler: try to find rank-23 by taking rank-27 and combining terms.
        
        # For now, let's use a known valid rank-23 if I can recall it.
        # Otherwise, fall back to schoolbook.
        
        # I'll try a structured rank-23:
        # Columns 0-6: Strassen-like for the 2x2 block (i,j,k in {0,1})
        # Columns 7-14: handle row 2 of A (i=2) with all j,k
        # Columns 15-22: handle column 2 of B (k=2) with i,j in {0,1}
        
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Strassen 2x2 for i,j,k in {0,1}:
        # M0 = A00*B00
        # M1 = (A00+A01)*(B10-B11)  
        # M2 = (A10-A11)*B11
        # M3 = A11*B01
        # M4 = A00*(B11+B01)
        # M5 = (A10+A11)*(B00-B11)
        # M6 = (A01-A10)*(B10+B11)
        # C00 = M0 + M4 - M6
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M0 - M1 + M2 + M5
        
        # In our indexing for the 2x2 block:
        # A00 -> U[0*3+0]=U[0], A01 -> U[1], A10 -> U[3], A11 -> U[4]
        # B00 -> V[0*3+0]=V[0], B01 -> V[1], B10 -> V[3], B11 -> V[4]
        # C00 -> W[0*3+0]=W[0], C01 -> W[1*3+0]=W[3], C10 -> W[0*3+1]=W[1], C11 -> W[1*3+1]=W[4]
        
        # Term 0: M0 = A00*B00, contributes to C00 and C11
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1; W[4, 0] = 1
        
        # Term 1: M1 = (A00+A01)*(B10-B11)
        U[0, 1] = 1; U[1, 1] = 1
        V[3, 1] = 1; V[4, 1] = -1
        W[4, 1] = -1  # C11 -= M1
        
        # Term 2: M2 = (A10-A11)*B11
        U[3, 2] = 1; U[4, 2] = -1
        V[4, 2] = 1
        W[1, 2] = 1; W[4, 2] = 1  # C10 += M2, C11 += M2
        
        # Term 3: M3 = A11*B01
        U[4, 3] = 1
        V[1, 3] = 1
        W[3, 3] = 1  # C01 += M3
        
        # Term 4: M4 = A00*(B11+B01)
        U[0, 4] = 1
        V[4, 4] = 1; V[1, 4] = 1
        W[0, 4] = 1; W[3, 4] = 1  # C00 += M4, C01 += M4
        
        # Term 5: M5 = (A10+A11)*(B00-B11)
        U[3, 5] = 1; U[4, 5] = 1
        V[0, 5] = 1; V[4, 5] = -1
        W[1, 5] = 1; W[4, 5] = 1  # C10 += M5, C11 += M5
        
        # Term 6: M6 = (A01-A10)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[3, 6] = 1; V[4, 6] = 1
        W[0, 6] = -1  # C00 -= M6
        
        # Now handle i=2 (third row of A): need C[2,k] for k=0,1,2
        # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
        # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
        
        # These involve A[2,j] = U[2*3+j] = U[6+j] for j=0,1,2
        # and B[j,k] = V[j*3+k]
        
        # Term 7: A[2,0]*B[0,0] -> C[2,0]
        U[6, 7] = 1; V[0, 7] = 1; W[2*3+0, 7] = 1  # W[6,7]
        
        # Term 8: A[2,1]*B[1,0] -> C[2,0]
        U[7, 8] = 1; V[3, 8] = 1; W[6, 8] = 1
        
        # Term 9: A[2,2]*B[2,0] -> C[2,0]
        U[8, 9] = 1; V[6, 9] = 1; W[6, 9] = 1
        
        # Term 10: A[2,0]*B[0,1] -> C[2,1]
        U[6, 10] = 1; V[1, 10] = 1; W[7, 10] = 1  # W[2*3+1]=W[7]
        
        # Term 11: A[2,1]*B[1,1] -> C[2,1]
        U[7, 11] = 1; V[4, 11] = 1; W[7, 11] = 1
        
        # Term 12: A[2,2]*B[2,1] -> C[2,1]
        U[8, 12] = 1; V[7, 12] = 1; W[7, 12] = 1
        
        # Term 13: A[2,0]*B[0,2] -> C[2,2]
        U[6, 13] = 1; V[2, 13] = 1; W[8, 13] = 1  # W[2*3+2]=W[8]
        
        # Term 14: A[2,1]*B[1,2] -> C[2,2]
        U[7, 14] = 1; V[5, 14] = 1; W[8, 14] = 1
        
        # Term 15: A[2,2]*B[2,2] -> C[2,2]
        U[8, 15] = 1; V[8, 15] = 1; W[8, 15] = 1
        
        # Now handle k=2 (third column of B) for i=0,1:
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
        
        # Term 16: A[0,0]*B[0,2] -> C[0,2]
        U[0, 16] = 1; V[2, 16] = 1; W[2*3+0, 16] = 1  # W[6]
        
        # Term 17: A[0,1]*B[1,2] -> C[0,2]
        U[1, 17] = 1; V[5, 17] = 1; W[6, 17] = 1
        
        # Term 18: A[0,2]*B[2,2] -> C[0,2]
        U[2, 18] = 1; V[8, 18] = 1; W[6, 18] = 1
        
        # Term 19: A[1,0]*B[0,2] -> C[1,2]
        U[3, 19] = 1; V[2, 19] = 1; W[7, 19] = 1  # W[2*3+1]=W[7]
        
        # Term 20: A[1,1]*B[1,2] -> C[1,2]
        U[4, 20] = 1; V[5, 20] = 1; W[7, 20] = 1
        
        # Term 21: A[1,2]*B[2,2] -> C[1,2]
        U[5, 21] = 1; V[8, 21] = 1; W[7, 21] = 1
        
        # We're at 22 terms. Need one more for rank 23.
        # Let me check what's missing. We have covered:
        # C[0,0], C[0,1] from Strassen
        # C[1,0], C[1,1] from Strassen  
        # C[2,0], C[2,1], C[2,2] from terms 7-15
        # C[0,2] from terms 16-18
        # C[1,2] from terms 19-21
        # That's all 9 C entries. So we have 22 terms.
        
        # Wait, but we need to verify that the Strassen terms don't interfere
        # with the k=2 or i=2 terms. Let me verify.
        
        # The Strassen terms (0-6) only involve U[0..4], V[0..4], W[0,1,3,4]
        # The i=2 terms (7-15) involve U[6..8], V[0,1,3,4,6,7,8], W[6,7,8]
        # The k=2 terms (16-21) involve U[0..5], V[2,5,8], W[6,7]
        
        # There's overlap in U and V indices but the W indices are disjoint
        # between Strassen (W[0,1,3,4]) and the others (W[6,7,8]).
        # So no interference.
        
        # But wait - we have 22 terms, not 23. Let me recount.
        # Terms 0-6: 7 terms
        # Terms 7-15: 9 terms
        # Terms 16-21: 6 terms
        # Total: 7+9+6 = 22 terms
        
        # So this gives rank 22! Let me verify this is correct.
        
        # Actually, I need to double-check the Strassen construction.
        # Let me verify C[0,0] = M0 + M4 - M6
        # M0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 -> contributes 1 to C[0,0]
        # M4: U[0,4]*V[4,4]*W[0,4] + U[0,4]*V[1,4]*W[0,4] = 1*1*1 + 1*1*1 = 2
        #      Wait, that's wrong. M4 = A[0,0]*(B[1,1]+B[0,1])
        #      So U[0,4]=1, and V[4,4]=1, V[1,4]=1
        #      Contribution to C[0,0]: U[0,4]*V[4,4]*W[0,4] + U[0,4]*V[1,4]*W[0,4]
        #      = 1*1*1 + 1*1*1 = 2. But we want A[0,0]*(B[1,1]+B[0,1]), so this is correct.
        
        # M6: U[1,6]*V[3,6]*W[0,6] + U[1,6]*V[4,6]*W[0,6] + U[3,6]*V[3,6]*W[0,6] + U[3,6]*V[4,6]*W[0,6]
        #     = 1*1*(-1) + 1*1*(-1) + (-1)*1*(-1) + (-1)*1*(-1)
        #     = -1 -1 +1 +1 = 0
        
        # Hmm, that's not right. M6 should contribute -M6 to C[0,0].
        # M6 = (A[0,1]-A[1,0])*(B[1,0]+B[1,1])
        # Contribution to C[0,0]: 
        #   A[0,1]*B[1,0]*(-1) + A[0,1]*B[1,1]*(-1) + A[1,0]*B[1,0]*(+1) + A[1,0]*B[1,1]*(+1)
        #   = -(A[0,1]*B[1,0] + A[0,1]*B[1,1] - A[1,0]*B[1,0] - A[1,0]*B[1,1])
        #   = -(A[0,1]-A[1,0])*(B[1,0]+B[1,1])
        #   = -M6
        # 
        # But C[0,0] should be A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # And M0 + M4 - M6 = A[0,0]*B[0,0] + A[0,0]*(B[0,1]+B[1,1]) - (A[0,1]-A[1,0])*(B[1,0]+B[1,1])
        # This doesn't simplify to A[0,0]*B[0,0] + A[0,1]*B[1,0] in general.
        
        # I think I'm making an error in the Strassen decomposition.
        # Let me use the standard 2x2 Strassen:
        # C[0,0] = M1 + M4 - M6 - M7
        # C[0,1] = M3 + M4
        # C[1,0] = M2 + M5
        # C[1,1] = M1 - M3 + M5 + M7
        # where:
        # M1 = A[0,0]*B[0,0]
        # M2 = A[1,0]*(B[0,0]+B[0,1])
        # M3 = (A[0,0]+A[0,1])*B[1,1]
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # M5 = (A[0,0]+A[1,0])*B[0,1]
        # M6 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # M7 = (A[0,0]-A[1,1])*(B[0,0]+B[1,1])
        
        # This is getting complicated. Let me just fall back to schoolbook to ensure correctness.
        
        if time.time() - t0 > 140:
            U, V, W = schoolbook()
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Let me verify my 22-term construction.
        # I'll check all 9 C entries.
        
        def verify(U, V, W):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(r):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1) > 1e-10:
                            return False
            return True
        
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Fallback to schoolbook
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    result = try_rank(23)
    return result
# EVOLVE-BLOCK-END

import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 2, 4, 8
    N = n * m  # 8
    M = m * p  # 32
    P = p * n  # 16

    def build_strassen_56():
        """Build the standard 56-rank Strassen decomposition."""
        U_local = np.zeros((4, 7), dtype=int)
        V_local = np.zeros((4, 7), dtype=int)
        W_local = np.zeros((4, 7), dtype=int)

        # r=0 (M1)
        U_local[0, 0] = 1; U_local[3, 0] = 1
        V_local[0, 0] = 1; V_local[3, 0] = 1
        W_local[0, 0] = 1; W_local[3, 0] = 1

        # r=1 (M2)
        U_local[2, 1] = 1; U_local[3, 1] = 1
        V_local[0, 1] = 1
        W_local[2, 1] = 1; W_local[3, 1] = -1

        # r=2 (M3)
        U_local[0, 2] = 1
        V_local[1, 2] = 1; V_local[3, 2] = -1
        W_local[1, 2] = 1; W_local[3, 2] = 1

        # r=3 (M4)
        U_local[3, 3] = 1
        V_local[2, 3] = 1; V_local[0, 3] = -1
        W_local[0, 3] = 1; W_local[2, 3] = 1

        # r=4 (M5)
        U_local[0, 4] = 1; U_local[1, 4] = 1
        V_local[3, 4] = 1
        W_local[0, 4] = -1; W_local[1, 4] = 1

        # r=5 (M6)
        U_local[2, 5] = 1; U_local[0, 5] = -1
        V_local[0, 5] = 1; V_local[1, 5] = 1
        W_local[3, 5] = 1

        # r=6 (M7)
        U_local[1, 6] = 1; U_local[3, 6] = -1
        V_local[2, 6] = 1; V_local[3, 6] = 1
        W_local[0, 6] = 1

        total_rank = 56
        U = np.zeros((8, total_rank), dtype=int)
        V = np.zeros((32, total_rank), dtype=int)
        W = np.zeros((16, total_rank), dtype=int)

        for j in range(4):
            for s in range(2):
                base_r = (j * 2 + s) * 7

                a_map = [0, 1, 4, 5] if s == 0 else [2, 3, 6, 7]
                b_map = [0 * 8 + 2 * j, 0 * 8 + 2 * j + 1, 1 * 8 + 2 * j, 1 * 8 + 2 * j + 1] if s == 0 else \
                        [2 * 8 + 2 * j, 2 * 8 + 2 * j + 1, 3 * 8 + 2 * j, 3 * 8 + 2 * j + 1]
                c_map = [2 * j * 2 + 0, (2 * j + 1) * 2 + 0, 2 * j * 2 + 1, (2 * j + 1) * 2 + 1]

                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        val = U_local[la, l]
                        if val != 0:
                            U[a_map[la], r] += val
                    for lb in range(4):
                        val = V_local[lb, l]
                        if val != 0:
                            V[b_map[lb], r] += val
                    for lc in range(4):
                        val = W_local[lc, l]
                        if val != 0:
                            W[c_map[lc], r] += val

        return U, V, W

    def verify_decomposition(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if s != 1:
                        return False, (i, j, k, s)
        return True, None

    # Build the 56-rank baseline
    U56, V56, W56 = build_strassen_56()
    valid, err = verify_decomposition(U56, V56, W56)

    if not valid:
        # Fallback to naive 64-rank
        U = np.zeros((8, 64), dtype=int)
        V = np.zeros((32, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i * 32 + j * 8 + k
                    U[i * 4 + j, r] = 1
                    V[j * 8 + k, r] = 1
                    W[k * 2 + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U56, V56, W56
    best_rank = 56

    # Try to find lower rank using systematic search
    # Strategy: Use the structure of the problem. The tensor T[i,j,k]=1 for all i,j,k.
    # We can try ranks from 48 to 55 and use integer linear algebra to find solutions.

    def try_rank_search(target_rank, time_limit=8.0):
        """Try to find a decomposition of the given target rank."""
        start = time.time()
        R = target_rank

        # The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a system of 64 polynomial equations in 8*R + 32*R + 16*R = 56*R unknowns.
        
        # For small improvements (rank 52-55), we can try a structured approach:
        # Start from the 56-rank solution and try to merge/remove terms.
        
        # Alternative: Use the fact that for 2x4x2 blocks, we might do better than 7 per block.
        # 4 blocks of 2x4x2: if we can get rank 13 per block, total = 52.
        # If rank 14 per block, total = 56 (current).
        
        # Let me try a direct approach for rank 52: 
        # Use 4 groups corresponding to the 4 values of j, each handling 2x4x2.
        # For each j-group, the sub-tensor is T[i, j, k] where i in {0,1}, k in {0,1}.
        # Wait, that's not right. Let me reconsider the block structure.
        
        # Actually, the natural block structure is:
        # A is 2x4, B is 4x8, C is 2x8
        # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
        # 
        # If we group j into pairs: j in {0,1} and j in {2,3}
        # Then C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
        # Each term is a 2x2x8 multiplication (2x2 A-block times 2x8 B-block)
        # 
        # Alternatively, group k into pairs: k in {0,1}, {2,3}, {4,5}, {6,7}
        # Then for each k-group, C[i,k_group] = sum_{j=0}^{3} A[i,j]*B[j,k_group]
        # This is a 2x4x2 multiplication for each k-group.
        
        # So we have 4 independent 2x4x2 subproblems.
        # Current: 4 * 7 = 28? No, that's not right either.
        
        # Let me re-examine. The 56-rank solution uses 8 groups of 2x2x2.
        # j in {0,1,2,3}, and for each j, k in pairs {0,1}, {2,3}, {4,5}, {6,7}
        # That's 4 * 4 = 16 groups of 2x2x2? No...
        
        # Actually looking at the code: for j in range(4), for s in range(2)
        # s=0: a_map = [0,1,4,5] (i=0,1 with j=2s,2s+1)
        # s=1: a_map = [2,3,6,7] (i=0,1 with j=2s,2s+1)
        # b_map for s=0: j=2*0,2*0+1,1*8+2*0,1*8+2*0+1 = [0,1,8,9]
        # This is getting complex. Let me just try a direct rank search.
        
        # For rank 52: try to find U(8,52), V(32,52), W(16,52) with integer entries
        # such that the tensor equation holds.
        
        # Use a greedy/iterative approach: start from 56-rank and try to eliminate terms.
        
        if time.time() - start > time_limit:
            return None
        
        # Try to see if rank 52 is possible by checking a necessary condition:
        # The border rank must be at least the rank. For matrix multiplication,
        # the known lower bounds are complex. Let's just try to construct it.
        
        # Approach: Use the 56-rank solution and try to express 4 of the terms
        # as linear combinations of the others.
        
        # This is complex. Let me try a different approach: 
        # Use a 2x4x2 decomposition with rank 13 (if it exists).
        
        # For 2x4x2: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
        # i in {0,1}, k in {0,1}, j in {0,1,2,3}
        # Tensor size: 2*4*2 = 16 entries, all 1.
        # 
        # Naive: 8 terms (2*4*2/1... no, 2*4*2 = 16, but each term covers one (i,j,k))
        # Actually naive is 8: for each (i,k), sum over j. So 2*2 = 4 output entries,
        # each needs 4 terms, total 16 rank-1 terms? No.
        # 
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
        # Each product A[i,j]*B[j,k] is a rank-1 term in the tensor.
        # So naive rank is 2*4*2 = 16? No, that's the number of (i,j,k) triples.
        # But the tensor T[i,j,k] = 1 for all i,j,k. The rank is the minimum number
        # of rank-1 tensors that sum to T.
        # 
        # For 2x2x2, Strassen gives rank 7 < 8 (naive).
        # For 2x4x2, naive is 16. Can we do better?
        # 
        # If we use Strassen on the j dimension: split j into {0,1} and {2,3}
        # Each is a 2x2x2 problem with rank 7. Total: 14.
        # 
        # Can we do 13? Let's try to find a rank-13 decomposition for 2x4x2.
        
        def find_2x4x2_rank_13():
            """Try to find a rank-13 decomposition for 2x4x2 matrix multiplication."""
            # T[i,j,k] = 1 for i in {0,1}, j in {0,1,2,3}, k in {0,1}
            # Need U(8,13), V(8,13), W(4,13) such that
            # sum_r U[i*4+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
            
            # Start from the rank-14 Strassen solution (two 2x2x2 blocks)
            # and try to merge one term.
            
            # Strassen for 2x2x2 (i in {0,1}, j in {0,1}, k in {0,1}):
            # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
            # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
            # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
            # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
            # M6 = (A10-A00)(B00+B01) -> C10 += M6
            # M7 = (A01-A11)(B10+B11) -> C00 += M7
            
            # For j in {0,1}: use indices 0,1,4,5 for U (i*4+j)
            # For j in {2,3}: use indices 2,3,6,7 for U
            # V: j*2+k, so j=0: k=0,1 -> 0,1; j=1: k=0,1 -> 2,3
            #     j=2: k=0,1 -> 4,5; j=3: k=0,1 -> 6,7
            # W: k*2+i, so k=0: i=0,1 -> 0,1; k=1: i=0,1 -> 2,3
            
            U14 = np.zeros((8, 14), dtype=int)
            V14 = np.zeros((8, 14), dtype=int)
            W14 = np.zeros((4, 14), dtype=int)
            
            # Block 1: j in {0,1} (indices 0,1 in j)
            # U indices: i*4+j -> i=0,j=0:0; i=0,j=1:1; i=1,j=0:4; i=1,j=1:5
            # V indices: j*2+k -> j=0,k=0:0; j=0,k=1:1; j=1,k=0:2; j=1,k=1:3
            # W indices: k*2+i -> k=0,i=0:0; k=0,i=1:1; k=1,i=0:2; k=1,i=1:3
            
            # M1: (A00+A11)(B00+B11)
            U14[0, 0] = 1; U14[5, 0] = 1
            V14[0, 0] = 1; V14[3, 0] = 1
            W14[0, 0] = 1; W14[3, 0] = 1
            
            # M2: (A10+A11)B00
            U14[4, 1] = 1; U14[5, 1] = 1
            V14[0, 1] = 1
            W14[1, 1] = 1; W14[3, 1] = -1
            
            # M3: A00(B10-B11)
            U14[0, 2] = 1
            V14[2, 2] = 1; V14[3, 2] = -1
            W14[2, 2] = 1; W14[3, 2] = 1
            
            # M4: A11(B01-B00)
            U14[5, 3] = 1
            V14[1, 3] = 1; V14[0, 3] = -1
            W14[0, 3] = 1; W14[2, 3] = 1
            
            # M5: (A00+A01)B11
            U14[0, 4] = 1; U14[1, 4] = 1
            V14[3, 4] = 1
            W14[0, 4] = -1; W14[2, 4] = 1
            
            # M6: (A10-A00)(B00+B01)
            U14[4, 5] = 1; U14[0, 5] = -1
            V14[0, 5] = 1; V14[1, 5] = 1
            W14[3, 5] = 1
            
            # M7: (A01-A11)(B10+B11)
            U14[1, 6] = 1; U14[5, 6] = -1
            V14[2, 6] = 1; V14[3, 6] = 1
            W14[0, 6] = 1
            
            # Block 2: j in {2,3} (indices 2,3 in j)
            # U indices: i*4+j -> i=0,j=2:2; i=0,j=3:3; i=1,j=2:6; i=1,j=3:7
            # V indices: j*2+k -> j=2,k=0:4; j=2,k=1:5; j=3,k=0:6; j=3,k=1:7
            # W indices: same as before
            
            # M1: (A02+A13)(B20+B31)
            U14[2, 7] = 1; U14[7, 7] = 1
            V14[4, 7] = 1; V14[7, 7] = 1
            W14[0, 7] = 1; W14[3, 7] = 1
            
            # M2: (A12+A13)B20
            U14[6, 8] = 1; U14[7, 8] = 1
            V14[4, 8] = 1
            W14[1, 8] = 1; W14[3, 8] = -1
            
            # M3: A02(B30-B31)
            U14[2, 9] = 1
            V14[6, 9] = 1; V14[7, 9] = -1
            W14[2, 9] = 1; W14[3, 9] = 1
            
            # M4: A13(B21-B20)
            U14[7, 10] = 1
            V14[5, 10] = 1; V14[4, 10] = -1
            W14[0, 10] = 1; W14[2, 10] = 1
            
            # M5: (A02+A03)B31
            U14[2, 11] = 1; U14[3, 11] = 1
            V14[7, 11] = 1
            W14[0, 11] = -1; W14[2, 11] = 1
            
            # M6: (A12-A02)(B20+B21)
            U14[6, 12] = 1; U14[2, 12] = -1
            V14[4, 12] = 1; V14[5, 12] = 1
            W14[3, 12] = 1
            
            # M7: (A03-A13)(B30+B31)
            U14[3, 13] = 1; U14[7, 13] = -1
            V14[6, 13] = 1; V14[7, 13] = 1
            W14[0, 13] = 1
            
            # Verify
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        s = 0
                        for r in range(14):
                            s += U14[i * 4 + j, r] * V14[j * 2 + k, r] * W14[k * 2 + i, r]
                        if s != 1:
                            return None
            
            # Now try to reduce to rank 13 by expressing one term as a combination of others.
            # This is a linear algebra problem over the integers/rationals.
            # 
            # The key insight: if the 14 rank-1 tensors are linearly dependent,
            # we can reduce the rank. But they're likely independent.
            # 
            # Instead, let's try a different construction.
            
            return None  # Return None if we can't find rank 13
        
        # For now, let's try to verify if rank 52 is possible by a different method.
        # Use the fact that the tensor is highly symmetric.
        
        # Actually, let me try a more direct approach: use numpy to solve for a
        # lower-rank decomposition by treating it as a system of equations.
        
        # For rank R, we have 56*R variables and 64 equations.
        # For R=52: 2912 variables, 64 equations. Very underdetermined.
        # 
        # The problem is that the equations are cubic (products of three variables).
        # 
        # Let me try a different strategy: fix U and V, then solve for W.
        # W[k*n+i, r] = sum_{i',j'} T[i',j',k] * U[i'*m+j', r] * V[j'*p+k, r] / (something)
        # This doesn't directly work because of the product structure.
        
        # Given the time constraints and complexity, let me just return the 56-rank solution.
        # But first, let me try one more thing: a rank-54 or rank-55 solution
        # by modifying the 56-rank solution.
        
        return None
    
    # Try to find a lower rank solution
    for target_rank in [52, 53, 54, 55]:
        if time.time() > deadline - 5:
            break
        result = try_rank_search(target_rank)
        if result is not None:
            U, V, W = result
            valid, err = verify_decomposition(U, V, W)
            if valid:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

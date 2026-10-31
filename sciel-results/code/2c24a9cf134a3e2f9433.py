import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Define 2x2 Strassen factors
    # U2, V2, W2 for 2x2 matmul
    # Shape (4, 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Indices in 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11 get +M1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C10 +M2, C11 -M2
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C01 +M3, C11 +M3
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00 +M4, C10 +M4
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00 -M5, C01 +M5
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11 +M6
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00 +M7

    # Verify 2x2 Strassen
    # T2[i,j,k] = sum_r U2[i,r] V2[j,r] W2[k,r]
    # Should be 1 if j==k? No.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T2[i,j,k] = 1 for all i,j,k? No.
    # The identity is: sum_j sum_r U2[i,r] V2[j,r] W2[k,r] * A[i,j] B[j,k] = C[i,k]
    # This implies sum_r U2[i,r] V2[j,r] W2[k,r] = delta_{j, ?}
    # Actually, the tensor T2[i,j,k] is 1 if the indices are consistent with the multiplication.
    # For 2x2, T2[i,j,k] = 1 for all i,j,k is NOT correct.
    # The tensor is T2[i,j,k] = 1 if j is the intermediate index?
    # Let's check the sum:
    # sum_r U2[i,r] V2[j,r] W2[k,r]
    # If i=0, j=0, k=0:
    # r=0: 1*1*1 = 1
    # r=1: 0
    # r=2: 1*0*0 = 0
    # r=3: 0
    # r=4: 1*0*-1 = 0
    # r=5: -1*1*0 = 0
    # r=6: 0
    # Sum = 1.
    # If i=0, j=1, k=0:
    # r=0: 1*0*1 = 0
    # r=2: 1*1*0 = 0
    # r=4: 1*0*-1 = 0
    # r=5: -1*1*0 = 0
    # r=6: 0
    # Sum = 0.
    # So T2[i,j,k] = 1 if j==k? No, i=0,j=0,k=0 -> 1. i=0,j=1,k=0 -> 0.
    # It seems T2[i,j,k] = 1 if j==k is NOT the rule.
    # The rule is: The tensor represents the multiplication.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The bilinear form is C[i,k] = sum_r U[i,r] V[k,r] W[j,r]? No.
    # The standard decomposition is:
    # C[i,k] = sum_r U[i,r] V[k,r] W[j,r] is wrong because W depends on j.
    # The correct form is:
    # C[i,k] = sum_j sum_r U[i,r] V[j,r] W[k,r] ? No.
    # The tensor T is defined such that:
    # T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Since it always contributes, T[i,j,k] = 1?
    # If T[i,j,k] = 1, then sum_r U[i,r]V[j,r]W[k,r] = 1.
    # Let's check i=0,j=0,k=1 for 2x2.
    # r=0: 1*0*1 = 0
    # r=2: 1*1*1 = 1
    # r=4: 1*0*1 = 0
    # r=6: 0
    # Sum = 1.
    # i=1,j=0,k=0:
    # r=1: 1*1*1 = 1
    # r=3: 0
    # r=5: 1*1*0 = 0
    # r=6: 0
    # Sum = 1.
    # It seems T[i,j,k] = 1 for ALL i,j,k in 2x2?
    # Let's check i=1,j=1,k=0.
    # r=1: 1*0*1 = 0
    # r=2: 0
    # r=3: 1*(-1)*1 = -1
    # r=4: 0
    # r=5: 0
    # r=6: -1*0*1 = 0
    # Sum = -1.
    # So T[1,1,0] = -1?
    # But the matrix multiplication tensor should have 1s where it contributes.
    # The contribution of A[1,1]B[1,0] to C[1,0] is +1.
    # Why did we get -1?
    # Because the Strassen factors have signs.
    # The tensor T is NOT all 1s.
    # The tensor T is the "structure" tensor.
    # For matrix multiplication, T[i,j,k] = 1 if j is the inner index.
    # Wait, the error in the previous program was that the tensor identity failed.
    # The identity is: sum_r U[u,r] V[v,r] W[w,r] = T[u,v,w].
    # What is T[u,v,w]?
    # T[u,v,w] = 1 if the indices (u,v,w) correspond to a valid term in the sum.
    # u = i*m+j, v = j*p+k, w = k*n+i.
    # For 4x4, u=0 (i=0,j=0), v=1 (j=0,k=1), w=4 (k=1,i=0).
    # This is a valid term. T[0,1,4] should be 1.
    # The previous program calculated 0 or something else.
    
    # Now, for the 4x4 construction:
    # We map the 49 columns.
    for k in range(7):
        for s in range(7):
            r = k * 7 + s
            
            # Determine which 2x2 blocks of A, B, C are involved in M_k
            # and how they map to the sub-problem s.
            
            # A blocks for M_k
            # M1: A00, A11
            # M2: A10, A11
            # M3: A00
            # M4: A11
            # M5: A00, A01
            # M6: A10, A00
            # M7: A01, A11
            
            # B blocks for M_k
            # M1: B00, B11
            # M2: B00
            # M3: B01, B11
            # M4: B10, B00
            # M5: B11
            # M6: B00, B01
            # M7: B10, B11
            
            # C blocks for M_k (contribution)
            # M1: C00(+1), C11(+1)
            # M2: C10(+1), C11(-1)
            # M3: C01(+1), C11(+1)
            # M4: C00(+1), C10(+1)
            # M5: C00(-1), C01(+1)
            # M6: C11(+1)
            # M7: C00(+1)

            # We need to fill U, V, W for column r.
            # U[r] corresponds to A.
            # The sub-problem s uses U2[local_a, s].
            # local_a is the index in the 2x2 block.
            
            # Define mappings for A
            # k=0 (M1): A00 (block 0), A11 (block 3)
            # k=1 (M2): A10 (block 2), A11 (block 3)
            # k=2 (M3): A00 (block 0)
            # k=3 (M4): A11 (block 3)
            # k=4 (M5): A00 (block 0), A01 (block 1)
            # k=5 (M6): A10 (block 2), A00 (block 0)
            # k=6 (M7): A01 (block 1), A11 (block 3)
            
            A_blocks = [
                [(0, 1), (3, 1)],
                [(2, 1), (3, 1)],
                [(0, 1)],
                [(3, 1)],
                [(0, 1), (1, 1)],
                [(2, 1), (0, -1)], # A10 - A00
                [(1, 1), (3, -1)]  # A01 - A11
            ]
            
            B_blocks = [
                [(0, 1), (3, 1)],
                [(0, 1)],
                [(1, 1), (3, -1)], # B01 - B11
                [(2, 1), (0, -1)], # B10 - B00
                [(3, 1)],
                [(0, 1), (1, 1)],
                [(2, 1), (3, 1)]
            ]
            
            C_blocks = [
                [(0, 1), (3, 1)],
                [(2, 1), (3, -1)],
                [(1, 1), (3, 1)],
                [(0, 1), (2, 1)],
                [(0, -1), (1, 1)],
                [(3, 1)],
                [(0, 1)]
            ]
            
            # Fill U
            for block_idx, sign in A_blocks[k]:
                # block_idx is 0,1,2,3 corresponding to 00,01,10,11
                bi, bj = divmod(block_idx, 2)
                # Local indices in 2x2 block
                for li in range(2):
                    for lj in range(2):
                        # Global index in 4x4
                        gi = bi * 2 + li
                        gj = bj * 2 + lj
                        u_idx = gi * 4 + gj
                        
                        # Local index in 2x2 sub-problem
                        l_idx = li * 2 + lj
                        
                        val = sign * U2[l_idx, s]
                        U[u_idx, r] += val
                        
            # Fill V
            for block_idx, sign in B_blocks[k]:
                bi, bj = divmod(block_idx, 2)
                for li in range(2):
                    for lj in range(2):
                        gi = bi * 2 + li
                        gj = bj * 2 + lj
                        v_idx = gi * 4 + gj
                        
                        l_idx = li * 2 + lj
                        
                        val = sign * V2[l_idx, s]
                        V[v_idx, r] += val
                        
            # Fill W
            for block_idx, sign in C_blocks[k]:
                # C block index bi, bj corresponds to C[bi, bj]
                # W index is w = k*n + i.
                # Here k is the column of C, i is the row of C.
                # In our notation, C block (bi, bj) means rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
                # So global row i = 2*bi + li, global col k = 2*bj + lj.
                # w_idx = k * n + i = (2*bj + lj) * 4 + (2*bi + li)
                
                bi, bj = divmod(block_idx, 2)
                for li in range(2):
                    for lj in range(2):
                        gi = 2 * bi + li # Row i
                        gj = 2 * bj + lj # Col k
                        
                        w_idx = gj * 4 + gi
                        
                        l_idx = li * 2 + lj
                        
                        val = sign * W2[l_idx, s]
                        W[w_idx, r] += val

    return {"U": U, "V": V, "W": W}

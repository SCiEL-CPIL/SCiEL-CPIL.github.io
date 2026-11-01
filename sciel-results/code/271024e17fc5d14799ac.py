<<<<<<< SEARCH
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Build rank-11 decomposition for 2x3x2 tensor
    # C[i,k] = sum_j A[i,j] * B[j,k], i in {0,1}, j in {0,1,2}, k in {0,1}
    # 
    # Rank-11 decomposition:
    # 7 terms from Strassen on the "core" 2x2x2 (using j=0,1 for the 2x2 part)
    # plus 4 terms for the j=2 column contribution
    #
    # Actually, let me use a direct rank-11 construction.
    # The 2x3x2 tensor T[i,j,k] = 1 for all valid i,j,k
    # We need: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # (with the trace convention W[k*n+i, c])
    
    # For 2x3x2: U shape (2*3, 11) = (6,11), V shape (3*2, 11) = (6,11), W shape (2*2, 11) = (4,11)
    # W uses trace convention: W[k*n+i, c] where n=2
    
    # Rank-11 decomposition of 2x3x2:
    # Terms 0-6: Strassen-like for the 2x2x2 sub-tensor (indices j=0,1)
    #   For Strassen 2x2x2 with A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    #   m0 = (a+d)(e+h) -> contributes to (0,0) and (1,1)
    #   m1 = (c+d)e     -> contributes to (1,0)
    #   m2 = c(f-h)     -> contributes to (1,1)
    #   m3 = a(g-h)     -> contributes to (0,1)
    #   m4 = (a+b)h     -> contributes to (0,0)
    #   m5 = (d-b)(e+g) -> contributes to (1,0)
    #   m6 = (c-a)(f+h) -> contributes to (0,1)
    #   
    #   But we have 3 rows in A (j=0,1,2) and 2 columns in B (k=0,1)
    #   Let's define A[i,j] for i=0,1 and j=0,1,2
    #   B[j,k] for j=0,1,2 and k=0,1
    #   
    #   C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    #   C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    #   C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    #   C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # Let me construct a rank-11 decomposition directly.
    # I'll use the following approach:
    # - 6 rank-1 terms for the "diagonal" (each handles one (i,j,k) combination partially)
    # - 5 additional terms that combine multiple products
    
    # Actually, let me use a known construction. For 2x3x2, rank 11:
    # The key insight is that we can write:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # Define intermediate variables:
    # s0 = A[0,0] + A[1,1]
    # s1 = A[0,1] + A[1,0]  
    # s2 = A[0,0] - A[1,1]
    # s3 = A[0,1] - A[1,0]
    # s4 = A[0,2] + A[1,2]
    # s5 = A[0,2] - A[1,2]
    
    # t0 = B[0,0] + B[1,1]
    # t1 = B[0,1] + B[1,0]
    # t2 = B[0,0] - B[1,1]
    # t3 = B[0,1] - B[1,0]
    # t4 = B[2,0]
    # t5 = B[2,1]
    
    # This is getting complex. Let me use a simpler direct construction.
    
    # For 2x3x2, I'll use the following rank-11 decomposition:
    # Each term c has U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # where n=2, m=3, p=2 for the sub-tensor
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    # Term 1: A[0,0]*B[0,1] -> C[0,1]
    # Term 2: A[0,1]*B[1,0] -> C[0,0]
    # Term 3: A[0,1]*B[1,1] -> C[0,1]
    # Term 4: A[0,2]*B[2,0] -> C[0,0]
    # Term 5: A[0,2]*B[2,1] -> C[0,1]
    # Term 6: A[1,0]*B[0,0] -> C[1,0]
    # Term 7: A[1,0]*B[0,1] -> C[1,1]
    # Term 8: A[1,1]*B[1,0] -> C[1,0]
    # Term 9: A[1,1]*B[1,1] -> C[1,1]
    # Term 10: A[1,2]*B[2,0] + A[1,2]*B[2,1]... no, that's not rank-1
    
    # That's only 10 terms but doesn't cover all. Let me count: we need 12 products
    # in schoolbook, but rank 11 means we can save 1.
    
    # The saving comes from combining: e.g., (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) 
    # = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    # But A[0,0]*B[1,1] and A[1,1]*B[0,0] are cross terms that don't appear in C.
    
    # Let me just use the known rank-11 for 2x3x2. I'll construct it carefully.
    
    # For the 2x3x2 sub-tensor with local indices (i,j,k) where i in {0,1}, j in {0,1,2}, k in {0,1}:
    # We want T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Here's a rank-11 construction:
    # c=0: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> A[0,0]*B[0,0] contributes to C[0,0]
    # c=1: U[0,0]=1, V[0,1]=1, W[1,0]=1  -> A[0,0]*B[0,1] contributes to C[0,1]
    # c=2: U[0,1]=1, V[1,0]=1, W[0,0]=1  -> A[0,1]*B[1,0] contributes to C[0,0]
    # c=3: U[0,1]=1, V[1,1]=1, W[1,0]=1  -> A[0,1]*B[1,1] contributes to C[0,1]
    # c=4: U[0,2]=1, V[2,0]=1, W[0,0]=1  -> A[0,2]*B[2,0] contributes to C[0,0]
    # c=5: U[0,2]=1, V[2,1]=1, W[1,0]=1  -> A[0,2]*B[2,1] contributes to C[0,1]
    # c=6: U[1,0]=1, V[0,0]=1, W[0,1]=1  -> A[1,0]*B[0,0] contributes to C[1,0]
    # c=7: U[1,0]=1, V[0,1]=1, W[1,1]=1  -> A[1,0]*B[0,1] contributes to C[1,1]
    # c=8: U[1,1]=1, V[1,0]=1, W[0,1]=1  -> A[1,1]*B[1,0] contributes to C[1,0]
    # c=9: U[1,1]=1, V[1,1]=1, W[1,1]=1  -> A[1,1]*B[1,1] contributes to C[1,1]
    # c=10: Need to handle A[1,2]*B[2,0] and A[1,2]*B[2,1]
    
    # That's 10 terms for 10 of the 12 products. We still need:
    # A[1,2]*B[2,0] -> C[1,0]
    # A[1,2]*B[2,1] -> C[1,1]
    
    # That's 12 terms total (schoolbook). To get rank 11, we need to combine two of them.
    
    # Combine c=6 and c=7: (A[1,0]+A[1,0])... no.
    # Combine using: (A[1,0]+A[1,2])*(B[0,0]+B[2,0]) = A[1,0]*B[0,0] + A[1,0]*B[2,0] + A[1,2]*B[0,0] + A[1,2]*B[2,0]
    # But A[1,0]*B[2,0] and A[1,2]*B[0,0] are unwanted cross terms.
    
    # Alternative: Use the structure more cleverly.
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # Let u = A[1,0] + A[1,2], v = A[1,0] - A[1,2]
    # Let s = B[0,0] + B[2,0], t = B[0,0] - B[2,0]
    # Then u*s = A[1,0]*B[0,0] + A[1,0]*B[2,0] + A[1,2]*B[0,0] + A[1,2]*B[2,0]
    # v*t = A[1,0]*B[0,0] - A[1,0]*B[2,0] - A[1,2]*B[0,0] + A[1,2]*B[2,0]
    # u*s + v*t = 2*A[1,0]*B[0,0] + 2*A[1,2]*B[2,0]
    # So A[1,0]*B[0,0] + A[1,2]*B[2,0] = (u*s + v*t)/2
    
    # Similarly for the k=1 case:
    # C[1,1] needs A[1,0]*B[0,1] + A[1,2]*B[2,1]
    # Let s' = B[0,1] + B[2,1], t' = B[0,1] - B[2,1]
    # u*s' + v*t' = 2*A[1,0]*B[0,1] + 2*A[1,2]*B[2,1]
    
    # So we can compute:
    # p1 = (A[1,0]+A[1,2])*(B[0,0]+B[2,0])  [1 mult]
    # p2 = (A[1,0]-A[1,2])*(B[0,0]-B[2,0])  [1 mult]
    # p3 = (A[1,0]+A[1,2])*(B[0,1]+B[2,1])  [1 mult]
    # p4 = (A[1,0]-A[1,2])*(B[0,1]-B[2,1])  [1 mult]
    # Then A[1,0]*B[0,0] + A[1,2]*B[2,0] = (p1+p2)/2
    # And A[1,0]*B[0,1] + A[1,2]*B[2,1] = (p3+p4)/2
    
    # But we still need A[1,1]*B[1,0] and A[1,1]*B[1,1] separately (2 more mults)
    # Total for row 1: 4 + 2 = 6 mults (vs 3 in schoolbook for just the j=0,1,2 terms... wait)
    
    # Schoolbook for row 1: 3 mults (A[1,0]*B[0,0], A[1,1]*B[1,0], A[1,2]*B[2,0] for C[1,0])
    # My approach: 4 mults for the combined j=0,2 terms + 1 for j=1 = 5 mults for C[1,0]
    # That's worse!
    
    # Let me reconsider. The rank-11 result for 2x3x2 must use a different combination.
    
    # Actually, I think the correct approach is to use Strassen on a 2x2x2 sub-problem
    # and handle the third column/row separately.
    
    # Let me just use a verified rank-11 construction from literature.
    # For 2x3x2, one known rank-11 decomposition uses the following 11 terms:
    
    # I'll construct it as follows for the sub-tensor (n_sub=2, m_sub=3, p_sub=2):
    # Local U: (6, 11), V: (6, 11), W: (4, 11)
    
    # Let me define the 11 terms explicitly:
    # Using the convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    sub_U = [[0]*11 for _ in range(6)]  # (2*3, 11)
    sub_V = [[0]*11 for _ in range(6)]  # (3*2, 11)
    sub_W = [[0]*11 for _ in range(4)]  # (2*2, 11)
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    sub_U[0][0] = 1  # i=0, j=0
    sub_V[0][0] = 1  # j=0, k=0
    sub_W[0][0] = 1  # k=0, i=0 -> index 0*2+0=0
    
    # Term 1: A[0,0]*B[0,1] -> C[0,1]
    sub_U[0][1] = 1
    sub_V[1][1] = 1  # j=0, k=1
    sub_W[1][1] = 1  # k=1, i=0 -> index 1*2+0=2... wait
    
    # W[k*n+i, c] with n=2: W[k*2+i, c]
    # k=0, i=0: W[0, c]
    # k=0, i=1: W[1, c]
    # k=1, i=0: W[2, c]
    # k=1, i=1: W[3, c]
    
    # Let me redo:
    # Term 0: A[0,0]*B[0,0] -> C[0,0]: U[0,0]=1, V[0,0]=1, W[0,0]=1
    # Term 1: A[0,0]*B[0,1] -> C[0,1]: U[0,1]=1, V[1,1]=1, W[2,1]=1 (k=1,i=0)
    # Term 2: A[0,1]*B[1,0] -> C[0,0]: U[1,2]=1, V[2,2]=1, W[0,2]=1
    # Term 3: A[0,1]*B[1,1] -> C[0,1]: U[1,3]=1, V[3,3]=1, W[2,3]=1
    # Term 4: A[0,2]*B[2,0] -> C[0,0]: U[2,4]=1, V[4,4]=1, W[0,4]=1
    # Term 5: A[0,2]*B[2,1] -> C[0,1]: U[2,5]=1, V[5,5]=1, W[2,5]=1
    # Term 6: A[1,0]*B[0,0] -> C[1,0]: U[3,6]=1, V[0,6]=1, W[1,6]=1
    # Term 7: A[1,0]*B[0,1] -> C[1,1]: U[3,7]=1, V[1,7]=1, W[3,7]=1
    # Term 8: A[1,1]*B[1,0] -> C[1,0]: U[4,8]=1, V[2,8]=1, W[1,8]=1
    # Term 9: A[1,1]*B[1,1] -> C[1,1]: U[4,9]=1, V[3,9]=1, W[3,9]=1
    # Term 10: A[1,2]*B[2,0] -> C[1,0]: U[5,10]=1, V[4,10]=1, W[1,10]=1
    
    # That's 11 terms but I'm missing A[1,2]*B[2,1] -> C[1,1]!
    # I have 12 products but only 11 terms. I need to combine two.
    
    # Let me combine terms 6 and 10:
    # Instead of separate A[1,0]*B[0,0] and A[1,2]*B[2,0], use:
    # (A[1,0]+A[1,2])*(B[0,0]+B[2,0]) and subtract cross terms... but cross terms aren't free.
    
    # I think I need to accept that a simple rank-11 for 2x3x2 requires a more sophisticated construction.
    # Let me look at this differently.
    
    # The 2x3x2 tensor rank is indeed 11. Let me use a known construction.
    # One approach: view 2x3x2 as 2x2x2 (using j=0,1) plus a 2x1x2 (using j=2).
    # 2x2x2 has rank 7 (Strassen). 2x1x2 has rank 2. Total 9? No, that's not right either.
    
    # Actually, I realize I should just use the schoolbook rank 48 or try a different partition.
    # Let me try: 2x3x8 = 2x3x4 + 2x3x4, where each 2x3x4 might have rank < 24.
    
    # Or: 2x3x8 = 1x3x8 + 1x3x8, where each 1x3x8 has rank 8 (schoolbook). Total 16? No, 1x3x8 schoolbook is 3*8=24, two of them is 48.
    
    # Let me just stick with the 4x(2x3x2) approach and use rank 12 for each (schoolbook), total 48.
    # But that's the same as the current program!
    
    # Wait, the measured improvement was 48->44, so rank 11 for 2x3x2 IS achievable.
    # Let me try a different construction for rank 11.
    
    # Key idea: For C[1,0] and C[1,1], instead of 6 separate products, use 5:
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # Let p = (A[1,0]+A[1,2])*(B[0,0]+B[0,1]+B[2,0]+B[2,1])
    # This is too complex.
    
    # I'll use a verified rank-11 from the literature. The construction is:
    # For 2x3x2, the rank-11 decomposition is non-trivial. Let me use a simpler approach:
    # Just use rank 12 per sub-tensor (schoolbook) for now, giving total rank 48.
    # This matches the current score. To improve, I need the actual rank-11 construction.
    
    # Given time constraints, let me use a hybrid: schoolbook for most, but try to find
    # a better decomposition programmatically.
    
    # Actually, let me just return the schoolbook for now since I can't verify rank 11 easily.
    # The current program already does this.
    
    # Let me try a different partition: 2x3x8 = 2x2x8 + 0x1x8... no.
    # 2x3x8 = 2x2x8 + 2x1x8. 
    # 2x2x8: schoolbook rank 32. 2x1x8: schoolbook rank 16. Total 48. Same.
    
    # Or: 2x3x8 = 1x3x8 + 1x3x8. Each 1x3x8: rank 24. Total 48.
    
    # I think the best I can do without a verified rank-11 for 2x3x2 is 48.
    # Let me just return the schoolbook decomposition.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Build rank-11 decomposition for 2x3x2 sub-tensor using Strassen(7) + 4 rank-1 terms
    # Sub-tensor: n_sub=2, m_sub=3, p_sub=2
    # U shape: (2*3, 11) = (6, 11)
    # V shape: (3*2, 11) = (6, 11)  
    # W shape: (2*2, 11) = (4, 11), W[k*2+i, c]
    
    def build_2x3x2_rank11():
        """Return (U, V, W) for 2x3x2 with rank 11."""
        n_s, m_s, p_s = 2, 3, 2
        r = 11
        U = [[0.0]*r for _ in range(n_s*m_s)]  # (6, 11)
        V = [[0.0]*r for _ in range(m_s*p_s)]  # (6, 11)
        W = [[0.0]*r for _ in range(p_s*n_s)]  # (4, 11)
        
        # Strassen 2x2x2 for the j=0,1 part:
        # A_core = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B_core = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
        # 
        # m0 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> C[0,0] += m0, C[1,1] += m0
        # m1 = (A[1,0]+A[1,1])*B[0,0]           -> C[1,0] += m1
        # m2 = A[1,0]*(B[0,1]-B[1,1])           -> C[1,1] += m2
        # m3 = A[0,0]*(B[1,0]-B[1,1])           -> C[0,1] += m3
        # m4 = (A[0,0]+A[0,1])*B[1,1]           -> C[0,0] += m4
        # m5 = (A[1,1]-A[0,1])*(B[0,0]+B[1,0])  -> C[1,0] += m5
        # m6 = (A[1,0]-A[0,0])*(B[0,1]+B[1,1])  -> C[0,1] += m6
        
        # Term 0: m0 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # U: A[0,0] coeff 1, A[1,1] coeff 1
        # V: B[0,0] coeff 1, B[1,1] coeff 1
        # W: C[0,0] coeff 1, C[1,1] coeff 1
        U[0][0] = 1.0   # i=0,j=0
        U[4][0] = 1.0   # i=1,j=1
        V[0][0] = 1.0   # j=0,k=0
        V[3][0] = 1.0   # j=1,k=1
        W[0][0] = 1.0   # k=0,i=0 -> W[0*2+0]=W[0]
        W[3][0] = 1.0   # k=1,i=1 -> W[1*2+1]=W[3]
        
        # Term 1: m1 = (A[1,0]+A[1,1])*B[0,0]
        U[3][1] = 1.0   # i=1,j=0
        U[4][1] = 1.0   # i=1,j=1
        V[0][1] = 1.0   # j=0,k=0
        W[1][1] = 1.0   # k=0,i=1 -> W[0*2+1]=W[1]
        
        # Term 2: m2 = A[1,0]*(B[0,1]-B[1,1])
        U[3][2] = 1.0   # i=1,j=0
        V[1][2] = 1.0   # j=0,k=1
        V[3][2] = -1.0  # j=1,k=1
        W[3][2] = 1.0   # k=1,i=1 -> W[3]
        
        # Term 3: m3 = A[0,0]*(B[1,0]-B[1,1])
        U[0][3] = 1.0   # i=0,j=0
        V[2][3] = 1.0   # j=1,k=0
        V[3][3] = -1.0  # j=1,k=1
        W[2][3] = 1.0   # k=1,i=0 -> W[2]
        
        # Term 4: m4 = (A[0,0]+A[0,1])*B[1,1]
        U[0][4] = 1.0   # i=0,j=0
        U[1][4] = 1.0   # i=0,j=1
        V[3][4] = 1.0   # j=1,k=1
        W[0][4] = 1.0   # k=0,i=0 -> W[0]
        
        # Term 5: m5 = (A[1,1]-A[0,1])*(B[0,0]+B[1,0])
        U[4][5] = 1.0   # i=1,j=1
        U[1][5] = -1.0  # i=0,j=1
        V[0][5] = 1.0   # j=0,k=0
        V[2][5] = 1.0   # j=1,k=0
        W[1][5] = 1.0   # k=0,i=1 -> W[1]
        
        # Term 6: m6 = (A[1,0]-A[0,0])*(B[0,1]+B[1,1])
        U[3][6] = 1.0   # i=1,j=0
        U[0][6] = -1.0  # i=0,j=0
        V[1][6] = 1.0   # j=0,k=1
        V[3][6] = 1.0   # j=1,k=1
        W[2][6] = 1.0   # k=1,i=0 -> W[2]
        
        # Terms 7-10: rank-1 terms for j=2 contributions
        # Term 7: A[0,2]*B[2,0] -> C[0,0]
        U[2][7] = 1.0   # i=0,j=2
        V[4][7] = 1.0   # j=2,k=0
        W[0][7] = 1.0   # k=0,i=0 -> W[0]
        
        # Term 8: A[0,2]*B[2,1] -> C[0,1]
        U[2][8] = 1.0   # i=0,j=2
        V[5][8] = 1.0   # j=2,k=1
        W[2][8] = 1.0   # k=1,i=0 -> W[2]
        
        # Term 9: A[1,2]*B[2,0] -> C[1,0]
        U[5][9] = 1.0   # i=1,j=2
        V[4][9] = 1.0   # j=2,k=0
        W[1][9] = 1.0   # k=0,i=1 -> W[1]
        
        # Term 10: A[1,2]*B[2,1] -> C[1,1]
        U[5][10] = 1.0  # i=1,j=2
        V[5][10] = 1.0  # j=2,k=1
        W[3][10] = 1.0  # k=1,i=1 -> W[3]
        
        return U, V, W
    
    sub_U, sub_V, sub_W = build_2x3x2_rank11()
    
    # Assemble full 2x3x8 decomposition: 4 copies of 2x3x2
    # Full: n=2, m=3, p=8
    # U shape: (2*3, 44) = (6, 44)
    # V shape: (3*8, 44) = (24, 44)
    # W shape: (8*2, 44) = (16, 44), W[k*2+i, c]
    
    rank = 4 * 11  # 44
    U = [[0.0]*rank for _ in range(n*m)]   # (6, 44)
    V = [[0.0]*rank for _ in range(m*p)]   # (24, 44)
    W = [[0.0]*rank for _ in range(p*n)]   # (16, 44)
    
    for group in range(4):
        # This group handles columns k = group*2, group*2+1
        for c_local in range(11):
            c_global = group * 11 + c_local
            k_offset = group * 2
            
            # Copy U (same for all groups since A is shared)
            for i in range(2):
                for j in range(3):
                    U[i*m + j][c_global] = sub_U[i*m_s + j][c_local]
            
            # Copy V with k offset
            for j in range(3):
                for k_local in range(2):
                    k_global = k_offset + k_local
                    V[j*p + k_global][c_global] = sub_V[j*p_s + k_local][c_local]
            
            # Copy W with k offset
            for k_local in range(2):
                for i in range(2):
                    k_global = k_offset + k_local
                    W[k_global*n + i][c_global] = sub_W[k_local*p_s + i][c_local]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

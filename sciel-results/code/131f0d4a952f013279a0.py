import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Rank-6 decomposition for 4x3 matrix multiplication
    # Treat 2x2 input A as a 4-dim vector: [a, b, c, d]
    # Multiply by 2x3 block B to get 2x3 output C
    # Partition 12 columns into 4 blocks of 3 columns each
    # Total rank = 4 * 6 = 24
    
    num_blocks = p // 3  # 4 blocks of 3 columns
    rank_per_block = 6
    rank = num_blocks * rank_per_block  # 24
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 24)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 24)
    W = [[0] * rank for _ in range(p * n)]  # shape (24, 24)
    
    # Rank-6 decomposition for 4x3 matrix multiplication
    # A is a 4x1 vector [a,b,c,d]^T, B is a 2x3 matrix
    # C = A * B is 2x3: C[i,j] = sum_k A[k]*B[k,j]
    # Wait, this isn't standard matrix mult. Let me reconsider.
    #
    # Actually, for 2x2 times 2x3:
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    #
    # We can view this as: for each output column j (0,1,2),
    # C[:,j] = A * B[:,j] where B[:,j] is a 2x1 vector
    # This is just 2x1 times 1x1 for each column, which is rank 2 per column
    # Total rank = 2 * 3 = 6 per block
    #
    # But we need a rank-6 decomposition that works across all 3 columns
    # using the shared 4-dim input vector.
    #
    # The tensor T(A, B) = C has entries:
    # T[a,b,c,d; e,f,g,h,i,j; col,row] 
    # where col=0..2, row=0..1
    #
    # For rank-6: we need 6 rank-1 terms
    # Each term: u_k * v_k * w_k
    # u_k is in R^4 (A-space), v_k in R^6 (B-space for one block), w_k in R^6 (C-space)
    #
    # Let's use the fact that C[row, col] = a*B[0,col] + b*B[1,col] for row=0
    #                                   = c*B[0,col] + d*B[1,col] for row=1
    #
    # Rank-1 decomposition:
    # Term 1: u=[1,0,0,0], v=[1,0,0,0,0,0], w=[1,0,0,0,0,0] -> a*e -> C[0,0]
    # Term 2: u=[0,1,0,0], v=[0,0,0,1,0,0], w=[1,0,0,0,0,0] -> b*h -> C[0,0]
    # Term 3: u=[1,0,0,0], v=[0,1,0,0,0,0], w=[0,1,0,0,0,0] -> a*f -> C[0,1]
    # Term 4: u=[0,1,0,0], v=[0,0,0,0,1,0], w=[0,1,0,0,0,0] -> b*i -> C[0,1]
    # Term 5: u=[1,0,0,0], v=[0,0,1,0,0,0], w=[0,0,1,0,0,0] -> a*g -> C[0,2]
    # Term 6: u=[0,1,0,0], v=[0,0,0,0,0,1], w=[0,0,1,0,0,0] -> b*j -> C[0,2]
    #
    # But this only gives C[0,*]. We also need C[1,*].
    # This is rank 12, not 6.
    #
    # Let me reconsider. The 4x3 matrix multiplication interpretation:
    # If we stack A as a 4x1 vector, and B as... this doesn't quite work.
    #
    # Alternative: Use bilinear form approach
    # C[0,0] = a*e + b*h
    # C[0,1] = a*f + b*i  
    # C[0,2] = a*g + b*j
    # C[1,0] = c*e + d*h
    # C[1,1] = c*f + d*i
    # C[1,2] = c*g + d*j
    #
    # Rank-6 decomposition:
    # We can group by the B coefficients:
    # e: contributes to C[0,0] (via a) and C[1,0] (via c)
    # f: contributes to C[0,1] (via a) and C[1,1] (via c)
    # g: contributes to C[0,2] (via a) and C[1,2] (via c)
    # h: contributes to C[0,0] (via b) and C[1,0] (via d)
    # i: contributes to C[0,1] (via b) and C[1,1] (via d)
    # j: contributes to C[0,2] (via b) and C[1,2] (via d)
    #
    # Each B coefficient gives 2 output entries, but with different A coefficients.
    # For e: a*e and c*e -> need 2 rank-1 terms unless we can combine
    #
    # Actually, for each column of B (say column 0: e,h), we have:
    # C[0,0] = a*e + b*h
    # C[1,0] = c*e + d*h
    #
    # This is a 2x2 matrix product A * [e,h]^T, which has rank 2.
    # So 3 columns * rank 2 = rank 6 total!
    #
    # Rank-1 terms for column 0:
    # Term 1: u=[1,0,0,0], v=[1,0,0,0,0,0], w=[1,0,0,0,0,0] -> a*e -> C[0,0]
    # Term 2: u=[0,0,1,0], v=[1,0,0,0,0,0], w=[0,0,1,0,0,0] -> c*e -> C[1,0]
    # Term 3: u=[0,1,0,0], v=[0,0,0,1,0,0], w=[1,0,0,0,0,0] -> b*h -> C[0,0]
    # Term 4: u=[0,0,0,1], v=[0,0,0,1,0,0], w=[0,0,1,0,0,0] -> d*h -> C[1,0]
    #
    # Wait, that's 4 terms for one column. Let me re-examine.
    #
    # For one 2x2 * 2x1 multiplication: rank is 2 (it's just 2 scalar mults)
    # A * [e,h]^T = [a*e+b*h, c*e+d*h]
    # Rank-1: 
    #   Term 1: u=[1,0,0,0], v=[1,0], w=[1,0] -> a*e
    #   Term 2: u=[0,1,0,0], v=[0,1], w=[1,0] -> b*h
    #   Term 3: u=[0,0,1,0], v=[1,0], w=[0,1] -> c*e
    #   Term 4: u=[0,0,0,1], v=[0,1], w=[0,1] -> d*h
    # That's rank 4, not 2.
    #
    # I think the "rank 6 for 4x3" refers to a different decomposition.
    # Let me try a different approach: use the known optimal rank for 2x2x2x3.
    #
    # Actually, let's just use the schoolbook approach but more cleverly.
    # For 2x2 * 2x3, the rank is at most 2*2*3/2 = 6 if we can find a rank-6 decomp.
    #
    # Let me try:
    # M1 = a*(e+h), M2 = c*(e+h), M3 = b*(e+h), M4 = d*(e+h)
    # No, that doesn't work directly.
    #
    # Let's use:
    # For each pair of columns (0,1) and column 2 separately:
    # Columns 0,1: 2x2*2x2 -> rank 7 (Strassen) or rank 4 (schoolbook)
    # Column 2: 2x2*2x1 -> rank 2
    # Total: 4 + 2 = 6 per block of 3 columns!
    #
    # Schoolbook for 2x2*2x2:
    # C[0,0] = a*e + b*h
    # C[0,1] = a*f + b*i
    # C[1,0] = c*e + d*h
    # C[1,1] = c*f + d*i
    # Rank-4:
    # T1: a*e -> C[0,0]
    # T2: b*h -> C[0,0]
    # T3: a*f -> C[0,1]
    # T4: b*i -> C[0,1]
    # T5: c*e -> C[1,0]
    # T6: d*h -> C[1,0]
    # T7: c*f -> C[1,1]
    # T8: d*i -> C[1,1]
    # That's rank 8.
    #
    # I'm stuck. Let me just keep the Strassen rank-42 approach for now
    # and try to verify it's correct.
    
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 42)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # shape (24, 42)
    
    # Strassen 7-product decomposition for 2x2x2:
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[C00,C01],[C10,C11]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e)
    # M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    #
    # Local A indices: 0=a(A[0,0]), 1=b(A[0,1]), 2=c(A[1,0]), 3=d(A[1,1])
    # Local B indices: 0=e(B[0,0]), 1=f(B[0,1]), 2=g(B[1,0]), 3=h(B[1,1])
    # Local C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_terms = [
        # (u: dict of A_idx->coeff, v: dict of local_B_idx->coeff, w: dict of local_C_idx->coeff)
        {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},       # M1
        {2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1},      # M2
        {0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1},       # M3
        {3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1},       # M4
        {0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1},      # M5
        {0: -1, 2: 1},{0: 1, 1: 1},  {3: 1},             # M6
        {1: 1, 3: -1},{2: 1, 3: 1},  {0: 1},             # M7
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs = strassen_terms[s * 3]
            v_coeffs = strassen_terms[s * 3 + 1]
            w_coeffs = strassen_terms[s * 3 + 2]
            
            # Set U (A indices are global, no offset needed)
            for a_idx, val in u_coeffs.items():
                U[a_idx][col] = val
            
            # Set V (map local B index to global)
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t        # B[0, 2t]
                elif local_b == 1:
                    global_b = 2 * t + 1    # B[0, 2t+1]
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t  # B[1, 2t] = 12+2t
                else:  # local_b == 3
                    global_b = m * p // 2 + 2 * t + 1  # B[1, 2t+1]
                V[global_b][col] = val
            
            # Set W (map local C index to global W index: W[k*n+i] where k=col, i=row)
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0       # C[0, 2t]
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0   # C[0, 2t+1]
                elif local_c == 2:
                    global_w = (2 * t) * n + 1       # C[1, 2t]
                else:  # local_c == 3
                    global_w = (2 * t + 1) * n + 1   # C[1, 2t+1]
                W[global_w][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

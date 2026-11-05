def construct():
    n, m, p = 4, 4, 5
    rank = 65
    U = [[0]*rank for _ in range(n*m)]  # 16 x 65
    V = [[0]*rank for _ in range(m*p)]  # 20 x 65
    W = [[0]*rank for _ in range(p*n)]  # 20 x 65
    
    # Strassen 2x2x2 base
    def strassen_2x2x2():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        u[0][0]=1; u[3][0]=1
        v[0][0]=1; v[3][0]=1
        w[0][0]=1; w[3][0]=1
        # M2 = (A10+A11)B00 -> C10 += 1
        u[2][1]=1; u[3][1]=1
        v[0][1]=1
        w[2][1]=1
        # M3 = A00(B01-B11) -> C01 += 1
        u[0][2]=1
        v[1][2]=1; v[3][2]=-1
        w[1][2]=1
        # M4 = A11(B10-B00) -> C10 += 1, C11 += 1
        u[3][3]=1
        v[2][3]=1; v[0][3]=-1
        w[2][3]=1; w[3][3]=1
        # M5 = (A00+A01)B11 -> C11 -= 1
        u[0][4]=1; u[1][4]=1
        v[3][4]=1
        w[3][4]=-1
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        u[2][5]=1; u[0][5]=-1
        v[0][5]=1; v[1][5]=1
        w[3][5]=1
        # M7 = (A01-A11)(B10+B11) -> C01 += 1, C11 += 1
        u[1][6]=1; u[3][6]=-1
        v[2][6]=1; v[3][6]=1
        w[1][6]=1; w[3][6]=1
        return u, v, w
    
    u2, v2, w2 = strassen_2x2x2()
    
    # Build 4x4x4 Strassen (rank 49)
    # For each pair (s1, s2) in 0..6:
    #   The 2x2x2 Strassen for block row s1 and block col s2
    #   A block (i,j) where i,j in 0..1 corresponds to global A[2i+a, 2j+b]
    
    col = 0
    for s1 in range(7):
        for s2 in range(7):
            for i in range(2):
                for j in range(2):
                    # U index: (2i+a, 2j+b) for a,b in 0..1
                    # The local 2x2x2 Strassen uses u2[a*2+b][s1] for the A part
                    # But we need to map this to the global indices
                    
                    # For the A factor: local (a,b) -> global (2i+a, 2j+b)
                    # u2[a*2+b][s1] gives the coefficient
                    # But u2 is indexed by [a*2+b] where a is row, b is col of the 2x2 block
                    # Wait, u2 has 4 rows: 0=A00, 1=A01, 2=A10, 3=A11
                    # So u2[idx][s1] where idx = a*2+b
                    
                    for a in range(2):
                        for b in range(2):
                            u_val = u2[a*2+b][s1]
                            if u_val != 0:
                                U[(2*i+a)*4 + (2*j+b)][col] += u_val
                    
                    # For the B factor: local (a,b) -> global (2a+c, 2b+d) for c,d in 0..1
                    # v2 is indexed by [a*2+b] where a is row, b is col of the 2x2 block
                    for a in range(2):
                        for b in range(2):
                            v_val = v2[a*2+b][s2]
                            if v_val != 0:
                                V[(2*a)*5 + (2*b)][col] += v_val  # Wait, B is 4x5, so col is 0..4
                                # Actually for 4x4x4, B is 4x4, so V has 16 rows
                                # But in our problem, B is 4x5, so V has 20 rows
                                # For the 4x4x4 part, we only use the first 4 columns of B
                                # So V index: (2a+c)*5 + (2b+d) where c,d in 0..1
                                # But we're only using b in 0..1 (first 2 cols of the 2x2 block)
                                # Wait, I'm confusing myself.
                    
                    # Let me redo this more carefully.
                    # The 4x4x4 Strassen:
                    # A is 4x4, split into 2x2 blocks: A[i,j] for i,j in 0..1, each 2x2
                    # B is 4x4, split into 2x2 blocks: B[i,j] for i,j in 0..1, each 2x2
                    # C is 4x4, split into 2x2 blocks: C[i,j] for i,j in 0..1, each 2x2
                    #
                    # Strassen: M_s1_s2 = (sum of A blocks with u2 coeffs) * (sum of B blocks with v2 coeffs)
                    # C block (i,j) = sum over s1,s2 of w2[i*2+j][s1] * w2[i*2+j][s2] * M_s1_s2
                    #
                    # Wait, that's not right either. Let me think about this more carefully.
                    #
                    # The recursive Strassen for 4x4x4:
                    # Split A, B, C into 2x2 blocks of 2x2 matrices.
                    # Apply Strassen at the block level:
                    #   M1 = (A00+A11)(B00+B11)
                    #   M2 = (A10+A11)B00
                    #   ...
                    # Each Ms is a 2x2x2 matrix multiplication, which we compute with Strassen (rank 7).
                    # Total rank: 7 * 7 = 49.
                    #
                    # For block-level Strassen product M_s (s=0..6):
                    #   M_s = sum_{i,j} u2[i*2+j][s] * A_block(i,j) * B_block(i,j)
                    #   Wait, no. M_s = (sum_i u2[i][s] * A_i) * (sum_j v2[j][s] * B_j)
                    #   where A_i is a 2x2 block and B_j is a 2x2 block.
                    #
                    # Actually, in Strassen:
                    #   M1 = (A00+A11)(B00+B11)
                    #   This is a product of two 2x2 matrices.
                    #   To compute this product with Strassen (2x2x2), we use 7 rank-1 terms.
                    #   Each rank-1 term is: (linear form in (A00+A11)) * (linear form in (B00+B11))
                    #
                    # So for the 4x4x4 decomposition:
                    #   For each block-level Strassen index s (0..6):
                    #     For each 2x2x2 Strassen index t (0..6):
                    #       Rank-1 term:
                    #         U part: sum of A_block(i,j) with coefficient u2[i*2+j][s] * u2_2x2[i*2+j][t]
                    #         V part: sum of B_block(i,j) with coefficient v2[i*2+j][s] * v2_2x2[i*2+j][t]
                    #         W part: C_block(i,j) with coefficient w2[i*2+j][s] * w2_2x2[i*2+j][t]
                    
                    pass  # This is getting too complex for inline reasoning
    
    # Let me just hardcode the schoolbook for now and return rank 80
    # Actually, let me just return the current schoolbook result
    rank = 80
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(20)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(5):
                U[i*4+j][col] = 1
                V[j*5+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

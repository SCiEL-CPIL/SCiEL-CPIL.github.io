"""Kronecker-structured baseline for 2x4x8 matrix multiplication tensor."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Strassen decomposition for 2x2x2 matrix multiplication (rank 7)
    # Indices: i in {0,1} (row of A), j in {0,1} (col of A / row of B), k in {0,1} (col of B)
    # We define 7 rank-1 tensors (u_r, v_r, w_r) such that:
    # T[i,j,k] = sum_r U2[i,j,r] * V2[j,k,r] * W2[k,i,r]
    #
    # Standard Strassen for 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (b-c)(e+g)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Let's map to our tensor convention:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For 2x2x2: n=2, m=2, p=2
    # U indices: i*2+j
    # V indices: j*2+k
    # W indices: k*2+i

    # Strassen rank-7 factors for 2x2x2
    # r=0: M1 = (a+d)(e+h)
    # U: a is (0,0), d is (1,1). So U[0,0]=1, U[1,1]=1
    # V: e is (0,0), h is (1,1). So V[0,0]=1, V[1,1]=1
    # W: contributes to C11 (M1), C22 (M1).
    # C11 is i=0, k=0. C22 is i=1, k=1.
    # W indices: k*2+i.
    # W[0*2+0] = W[0] += 1
    # W[1*2+1] = W[3] += 1
    
    # r=1: M2 = (c+d)e
    # U: c is (1,0), d is (1,1). U[1,0]=1, U[1,1]=1
    # V: e is (0,0). V[0,0]=1
    # W: contributes to C21 (M2), C22 (-M2).
    # C21 is i=1, k=0. C22 is i=1, k=1.
    # W[0*2+1] = W[1] += 1
    # W[1*2+1] = W[3] -= 1

    # r=2: M3 = a(f-h)
    # U: a is (0,0). U[0,0]=1
    # V: f is (0,1), h is (1,1). V[0,1]=1, V[1,1]=-1
    # W: contributes to C12 (M3), C22 (M3).
    # C12 is i=0, k=1. C22 is i=1, k=1.
    # W[1*2+0] = W[2] += 1
    # W[1*2+1] = W[3] += 1

    # r=3: M4 = d(g-e)
    # U: d is (1,1). U[1,1]=1
    # V: g is (1,0), e is (0,0). V[1,0]=1, V[0,0]=-1
    # W: contributes to C11 (M4), C21 (M4).
    # C11 is i=0, k=0. C21 is i=1, k=0.
    # W[0*2+0] = W[0] += 1
    # W[0*2+1] = W[1] += 1

    # r=4: M5 = (a+b)h
    # U: a is (0,0), b is (0,1). U[0,0]=1, U[0,1]=1
    # V: h is (1,1). V[1,1]=1
    # W: contributes to C11 (-M5), C12 (M5).
    # C11 is i=0, k=0. C12 is i=0, k=1.
    # W[0*2+0] = W[0] -= 1
    # W[1*2+0] = W[2] += 1

    # r=5: M6 = (c-d)(f+h)
    # U: c is (1,0), d is (1,1). U[1,0]=1, U[1,1]=-1
    # V: f is (0,1), h is (1,1). V[0,1]=1, V[1,1]=1
    # W: contributes to C22 (M6).
    # C22 is i=1, k=1.
    # W[1*2+1] = W[3] += 1

    # r=6: M7 = (b-c)(e+g)
    # U: b is (0,1), c is (1,0). U[0,1]=1, U[1,0]=-1
    # V: e is (0,0), g is (1,0). V[0,0]=1, V[1,0]=1
    # W: contributes to C11 (M7).
    # C11 is i=0, k=0.
    # W[0*2+0] = W[0] += 1

    # Let's verify W indices for r=0..6
    # r=0: W[0]+=1, W[3]+=1
    # r=1: W[1]+=1, W[3]-=1
    # r=2: W[2]+=1, W[3]+=1
    # r=3: W[0]+=1, W[1]+=1
    # r=4: W[0]-=1, W[2]+=1
    # r=5: W[3]+=1
    # r=6: W[0]+=1

    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]

    # r=0
    U2[0][0] += 1; U2[3][0] += 1
    V2[0][0] += 1; V2[3][0] += 1
    W2[0][0] += 1; W2[3][0] += 1
    # r=1
    U2[2][1] += 1; U2[3][1] += 1
    V2[0][1] += 1
    W2[1][1] += 1; W2[3][1] -= 1
    # r=2
    U2[0][2] += 1
    V2[1][2] += 1; V2[3][2] -= 1
    W2[2][2] += 1; W2[3][2] += 1
    # r=3
    U2[3][3] += 1
    V2[2][3] += 1; V2[0][3] -= 1
    W2[0][3] += 1; W2[1][3] += 1
    # r=4
    U2[0][4] += 1; U2[1][4] += 1
    V2[3][4] += 1
    W2[0][4] -= 1; W2[2][4] += 1
    # r=5
    U2[2][5] += 1; U2[3][5] -= 1
    V2[1][5] += 1; V2[3][5] += 1
    W2[3][5] += 1
    # r=6
    U2[1][6] += 1; U2[2][6] -= 1
    V2[0][6] += 1; V2[2][6] += 1
    W2[0][6] += 1

    # Now construct the full 2x4x8 tensor using Kronecker products
    # The 2x4x8 multiplication can be seen as:
    # A (2x4) * B (4x8) = C (2x8)
    # Let's view A as a 2x2 block matrix of 1x2 blocks? No, 2x4 is 2 rows, 4 cols.
    # Let's view A as 2x2 blocks of 1x2?
    # Or better: view the indices.
    # i in 0..1, j in 0..3, k in 0..7.
    # We can decompose j into j1 (0..1) and j2 (0..1)? No, j is 0..3.
    # Let's use the structure:
    # A is 2x4. B is 4x8.
    # Let's group the middle dimension j (0..3) into two parts: j = 2*j1 + j2, where j1 in 0..1, j2 in 0..1.
    # And k (0..7) into k = 4*k1 + k2? No, k is 0..7.
    # Let's group k into k1 (0..1) and k2 (0..3)?
    # Actually, the Kronecker product strategy suggested in the prompt is:
    # "2x2x2 tensor (Strassen rank 7) multiplied by an 8x2x4 tensor (schoolbook rank 16)"
    # This implies a block structure.
    # Let's assume the 2x4x8 tensor is a Kronecker product of a 2x2x2 tensor and a 1x4x4 tensor?
    # Or 2x4x8 = (2x2x2) x (1x2x4)?
    # If we take A (2x4) and B (4x8), we can write A as a 2x2 block matrix where each block is 1x2.
    # A = [[A00, A01], [A10, A11]] where Aij is 1x2.
    # B = [[B00, B01, B02, B03], [B10, B11, B12, B13], [B20, ...], [B30, ...]]
    # This is getting complicated. Let's stick to the explicit rank-112 construction.
    #
    # The prompt says: "Kronecker product structure to decompose the 2x4x8 tensor into a 2x2x2 tensor (Strassen rank 7) multiplied by an 8x2x4 tensor (schoolbook rank 16)"
    # This phrasing is a bit odd. "8x2x4 tensor" usually means dimensions (8,2,4).
    # If we have a tensor T = T1 x T2, the rank is rank(T1)*rank(T2).
    # 7 * 16 = 112.
    #
    # Let's define the decomposition explicitly.
    # We will create 112 rank-1 terms.
    # The first 7 terms correspond to the Strassen part, and for each Strassen term, we have 16 "expansion" terms.
    #
    # Let's define the mapping from the 2x2x2 indices to the 2x4x8 indices.
    # i2 in {0,1} -> i in {0,1}
    # j2 in {0,1} -> j in {0,1,2,3} ?
    # k2 in {0,1} -> k in {0,1,2,3,4,5,6,7} ?
    #
    # A natural mapping:
    # j = 2*j2 + j1, where j1 in {0,1}.
    # k = 4*k2 + k1, where k1 in {0,1,2,3}.
    #
    # Then the 2x4x8 multiplication tensor T[i, j, k] = T[i, 2*j2+j1, 4*k2+k1].
    #
    # The "8x2x4 tensor" part likely refers to the expansion of the middle and last dimensions.
    # Let's assume the 2x2x2 tensor handles the i, j2, k2 dimensions.
    # And the "8x2x4" tensor handles the i, j1, k1 dimensions?
    # No, the i dimension is shared.
    #
    # Let's try a different interpretation.
    # Maybe the 2x4x8 tensor is viewed as a 2x2 block matrix of 1x4 blocks?
    # A = [[A0, A1], [A2, A3]] where A0..A3 are 1x4.
    # B = [[B0, B1, B2, B3], [B4, B5, B6, B7], [B8, B9, B10, B11], [B12, B13, B14, B15]]
    # This doesn't seem to lead to a simple Kronecker product.
    #
    # Let's just build the rank-112 decomposition by combining the Strassen rank-1 terms with a schoolbook rank-16 expansion.
    #
    # For each of the 7 Strassen terms (r in 0..6), we have factors U2[i2, j2, r], V2[j2, k2, r], W2[k2, i2, r].
    # We need to expand this to the full 2x4x8 tensor.
    #
    # Let's assume the following structure:
    # The 2x4x8 tensor is the sum over r=0..6 of (U2_r x U1_r) * (V2_r x V1_r) * (W2_r x W1_r)
    # where U1, V1, W1 are the factors of the "8x2x4" tensor.
    #
    # What is the "8x2x4" tensor?
    # It must be a tensor that, when combined with the 2x2x2 tensor, gives the 2x4x8 multiplication tensor.
    #
    # Let's define the "8x2x4" tensor as the tensor for multiplying a 1x4 matrix by a 4x8 matrix? No, that's 1x4x8.
    #
    # Let's look at the dimensions again.
    # 2x4x8.
    # If we split 4 into 2x2 and 8 into 2x4, we get (2x2x2) x (1x2x4)?
    # (2x2x2) has dimensions (2,2,2).
    # (1x2x4) has dimensions (1,2,4).
    # The Kronecker product of tensors T1 (n1,m1,p1) and T2 (n2,m2,p2) is a tensor of dimensions (n1*n2, m1*m2, p1*p2).
    # So (2,2,2) x (1,2,4) = (2*1, 2*2, 2*4) = (2,4,8).
    # This matches!
    #
    # So the 2x4x8 tensor is the Kronecker product of a 2x2x2 tensor and a 1x2x4 tensor.
    # The 2x2x2 tensor is the matrix multiplication tensor for 2x2 matrices (rank 7 via Strassen).
    # The 1x2x4 tensor is the tensor for multiplying a 1x2 matrix by a 2x4 matrix.
    # The 1x2x4 multiplication tensor has rank 1*2*4 = 8 (schoolbook).
    # Wait, the prompt says "8x2x4 tensor (schoolbook rank 16)".
    # Maybe it's (2,2,2) x (4,1,2)? No.
    # Maybe (2,2,2) x (4,2,1)?
    # (2,2,2) x (4,2,1) = (8,4,2). No.
    #
    # Let's re-read: "2x2x2 tensor (Strassen rank 7) multiplied by an 8x2x4 tensor (schoolbook rank 16)"
    # If the second tensor is 8x2x4, its rank is 8*2*4 = 64.
    # 7 * 64 = 448. That's worse than 64.
    #
    # Maybe it means the 2x4x8 tensor is decomposed into a sum of 7 terms, each of which is a "8x2x4" structure?
    # Or maybe the "8x2x4" refers to the dimensions of the factors?
    #
    # Let's assume the prompt's "8x2x4 tensor" is a typo or misinterpretation, and it means the 1x2x4 tensor (rank 8).
    # Then total rank = 7 * 8 = 56.
    # This matches the "Strassen decomposition reduces rank from 64 to 56" finding!
    #
    # So the strategy is:
    # 1. Decompose 2x4x8 into (2x2x2) x (1x2x4).
    # 2. Use Strassen (rank 7) for 2x2x2.
    # 3. Use schoolbook (rank 8) for 1x2x4.
    # 4. Total rank = 7 * 8 = 56.
    #
    # Let's implement this.
    #
    # 1x2x4 multiplication tensor:
    # A (1x2) * B (2x4) = C (1x4)
    # i=0, j in 0..1, k in 0..3.
    # T[i,j,k] = 1 if i=0.
    # Rank 8: U[i,j,r], V[j,k,r], W[k,i,r].
    # For each j in 0..1, k in 0..3:
    # r = j*4 + k
    # U[0, j, r] = 1
    # V[j, k, r] = 1
    # W[k, 0, r] = 1
    #
    # Now combine with 2x2x2.
    # The Kronecker product of two tensors T1 (n1,m1,p1) and T2 (n2,m2,p2) is defined as:
    # (T1 x T2)[(i1,i2), (j1,j2), (k1,k2)] = T1[i1,j1,k1] * T2[i2,j2,k2]
    #
    # In our case:
    # T1 is 2x2x2 (n1=2, m1=2, p1=2)
    # T2 is 1x2x4 (n2=1, m2=2, p2=4)
    # Result is 2x4x8 (n=2, m=4, p=8)
    #
    # Indices:
    # i = i1 * n2 + i2 = i1 * 1 + i2 = i1 (since i2=0)
    # j = j1 * m2 + j2 = j1 * 2 + j2
    # k = k1 * p2 + k2 = k1 * 4 + k2
    #
    # So:
    # i1 in 0..1, i2 = 0 -> i = i1
    # j1 in 0..1, j2 in 0..1 -> j = 2*j1 + j2
    # k1 in 0..1, k2 in 0..3 -> k = 4*k1 + k2
    #
    # The decomposition is:
    # T[i, j, k] = sum_{r1=0..6} sum_{r2=0..7} U1[i1, j1, r1] * U2[i2, j2, r2] * V1[j1, k1, r1] * V2[j2, k2, r2] * W1[k1, i1, r1] * W2[k2, i2, r2]
    #
    # Let R = r1 * 8 + r2. Total rank = 7 * 8 = 56.
    #
    # U[i, j, R] = U1[i1, j1, r1] * U2[i2, j2, r2]
    # V[j, k, R] = V1[j1, k1, r1] * V2[j2, k2, r2]
    # W[k, i, R] = W1[k1, i1, r1] * W2[k2, i2, r2]
    #
    # Where:
    # i1 = i, i2 = 0
    # j1 = j // 2, j2 = j % 2
    # k1 = k // 4, k2 = k % 4
    #
    # Let's construct U, V, W.
    rank = 7 * 8
    U = [[0] * rank for _ in range(n * m)]  # 2*4 = 8
    V = [[0] * rank for _ in range(m * p)]  # 4*8 = 32
    W = [[0] * rank for _ in range(p * n)]  # 8*2 = 16

    # Precompute U2, V2, W2 for 1x2x4 (rank 8)
    U2 = [[0] * 8 for _ in range(1 * 2)]  # 1*2 = 2
    V2 = [[0] * 8 for _ in range(2 * 4)]  # 2*4 = 8
    W2 = [[0] * 8 for _ in range(4 * 1)]  # 4*1 = 4
    for j2 in range(2):
        for k2 in range(4):
            r2 = j2 * 4 + k2
            U2[0 * 2 + j2][r2] = 1
            V2[j2 * 4 + k2][r2] = 1
            W2[k2 * 1 + 0][r2] = 1

    # Now combine
    for r1 in range(7):
        for r2 in range(8):
            R = r1 * 8 + r2
            for i1 in range(2):
                i2 = 0
                i = i1 * 1 + i2
                for j1 in range(2):
                    for j2 in range(2):
                        j = j1 * 2 + j2
                        u_val = U2[i1 * 2 + j1][r1] * U2[0 * 2 + j2][r2]
                        if u_val != 0:
                            U[i * 4 + j][R] += u_val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(4):
                            j = j1 * 2 + j2
                            k = k1 * 4 + k2
                            v_val = U2[j1 * 2 + 0][r1] * U2[j2 * 4 + 0][r2] # Wait, V1 and V2
                            # V1[j1, k1, r1] * V2[j2, k2, r2]
                            v_val = (1 if (j1 == 0 and k1 == 0) or (j1 == 0 and k1 == 1) or (j1 == 1 and k1 == 0) or (j1 == 1 and k1 == 1) else 0) # This is wrong, V1 is from Strassen
                            # Let's use the U2, V2, W2 arrays correctly.
                            # V1 is the V factor of the 2x2x2 tensor.
                            # V1[j1, k1, r1]
                            # V2[j2, k2, r2]
                            v_val = V2[j1 * 2 + k1][r1] * V2[j2 * 4 + k2][r2] # Wrong, V2 is for 1x2x4, V1 is for 2x2x2
                            # Let's rename.
                            # V_strassen[j1, k1, r1]
                            # V_small[j2, k2, r2]
                            pass

    # Let's restart the combination loop with clear variable names.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r1 in range(7):
        for r2 in range(8):
            R = r1 * 8 + r2
            # U factors
            for i1 in range(2):
                i2 = 0
                i = i1
                for j1 in range(2):
                    for j2 in range(2):
                        j = j1 * 2 + j2
                        u_val = U2[i1 * 2 + j1][r1] * U2[0 * 2 + j2][r2]
                        if u_val != 0:
                            U[i * 4 + j][R] += u_val
            # V factors
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(4):
                            j = j1 * 2 + j2
                            k = k1 * 4 + k2
                            v_val = V2[j1 * 2 + k1][r1] * V2[j2 * 4 + k2][r2]
                            if v_val != 0:
                                V[j * 8 + k][R] += v_val
            # W factors
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(4):
                        i2 = 0
                        k = k1 * 4 + k2
                        i = i1
                        w_val = W2[k1 * 2 + i1][r1] * W2[k2 * 1 + 0][r2]
                        if w_val != 0:
                            W[k * 2 + i][R] += w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

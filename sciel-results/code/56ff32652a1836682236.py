def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen: for X*Y, the 7 multiplications are:
    # Each multiplication is: (linear combo of X entries) * (linear combo of Y entries)
    # contributing to specific entries of Z = X*Y
    # 
    # We represent each Strassen multiplication as:
    # (x_coeff, y_coeff, z_target) where x_coeff is a dict from (row,col) to coeff for X,
    # y_coeff is a dict from (row,col) to coeff for Y, and z_target is a dict from (row,col) to coeff for Z
    
    # Actually, for the tensor decomposition, each scalar multiplication is a rank-1 term.
    # The term for computing z_rc = sum of x_ij * y_jk type products.
    
    # Let me define the 7 Strassen multiplications for 2x2:
    # m1: (x00+x11) * (y00+y11) -> contributes to z00, z01, z10, z11
    # m2: (x10+x11) * y00 -> contributes to z10, z11
    # m3: x00 * (y01-y11) -> contributes to z01, z11
    # m4: x11 * (y10-y00) -> contributes to z10, z11
    # m5: (x00+x01) * y11 -> contributes to z00, z01
    # m6: (x10-x00) * (y00+y01) -> contributes to z10, z11
    # m7: (x01-x11) * (y10+y11) -> contributes to z00, z01, z10, z11
    
    # For the tensor decomposition, each scalar multiplication m is a rank-1 tensor.
    # m = (linear combo of A entries) * (linear combo of B entries) * (linear combo of C entries)
    # But wait, in the tensor decomposition, each rank-1 term has a SINGLE (i,j) for A,
    # a SINGLE (j,k) for B, and a SINGLE (k,i) for C.
    
    # So I can't directly use Strassen's linear combinations in the U, V, W matrices
    # unless I expand them.
    
    # The correct way: expand each Strassen multiplication into its component scalar
    # multiplications. Each (x_ij * y_jk) product is a separate rank-1 term.
    
    # But that would give back the schoolbook 4*4=16 terms per 2x2 multiplication,
    # and 7*16 = 112 total, which is worse.
    
    # I'm confusing myself. The tensor rank decomposition is about the NUMBER of scalar
    # multiplications, not the number of rank-1 tensors in a naive sense.
    
    # Let me reconsider. The multiplication tensor T for n×n×n has T[i,j,k] = 1 for
    # all i,j,k (since A[i,j]*B[j,k] appears in C[i,k]). The rank is the minimum number
    # of rank-1 tensors summing to T.
    
    # A rank-1 tensor R has R[i,j,k] = a[i,j] * b[j,k] * c[k,i] for some vectors a, b, c.
    
    # For the schoolbook, we have 64 rank-1 tensors, each with a single 1 in a, b, c.
    
    # For Strassen on 2×2, we have 7 rank-1 tensors where a, b, c have multiple non-zero entries.
    
    # So for 4×4 with Kronecker Strassen, we have 49 rank-1 tensors.
    
    # Let me define them properly.
    
    # For the 2×2 case, the 7 rank-1 tensors are:
    # Term 1: a = [1,0,0,1] (x00+x11), b = [1,0,0,1] (y00+y11), c = [1,1,1,1] (z00+z01+z10+z11)
    # Wait, that's not right either. Let me be more careful.
    
    # In the 2×2 case, A has entries a00,a01,a10,a11 (indexed 0-3), B has b00,b01,b10,b11, C has c00,c01,c10,c11.
    # The multiplication tensor T has T[i,j,k] = 1 where i indexes A row, j indexes A col/B row, k indexes B col/C col.
    # Actually, let me use the convention from the problem:
    # U has shape (n*m, rank) = (4, rank) for 2×2, indexing A[i,j] as i*m+j
    # V has shape (m*p, rank) = (4, rank), indexing B[j,k] as j*p+k
    # W has shape (p*n, rank) = (4, rank), indexing C[k,i] as k*n+i
    
    # The tensor T[i,j,k] = 1 for all i,j,k in {0,1} (for 2×2 case).
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1.
    
    # For 2×2, the 7 Strassen terms give 7 rank-1 tensors. Let me verify one:
    # Term 1: a00*a00 + a00*a11 + a11*a00 + a11*a11 type thing... 
    
    # I think I'm overcomplicating this. Let me just look up the explicit 48-term list.
    
    # After more thought, let me use a verified approach: I'll implement the 49-term
    # Kronecker Strassen by explicitly constructing the 49 rank-1 tensors.
    
    # For the 2×2 Strassen, the 7 multiplications are:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    # And the output is:
    # c00 = m1 + m4 - m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m7
    
    # Each mj is a product of a linear form in A and a linear form in B.
    # In the tensor decomposition, each mj corresponds to ONE rank-1 tensor where:
    # The U vector has the coefficients of the A linear form
    # The V vector has the coefficients of the B linear form
    # The W vector has the coefficients of how mj contributes to C
    
    # For example, m1 = (a00+a11)(b00+b11):
    # U vector: [1, 0, 0, 1] (coefficients of a00, a01, a10, a11)
    # V vector: [1, 0, 0, 1] (coefficients of b00, b01, b10, b11)
    # W vector: [1, 1, 1, 1] (m1 contributes +1 to c00, c01, c10, c11)
    
    # Let me verify: sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r] should equal 1 for all i,j,k.
    # For i=0,j=0,k=0: U[0,0,1]*V[0,0,1]*W[0,0,1] = 1*1*1 = 1. But we need the SUM over all 7 terms to equal 1.
    
    # Let me check all 7 terms for (i,j,k) = (0,0,0):
    # Term 1: U[0,0,1]=1, V[0,0,1]=1, W[0,0,1]=1 -> 1
    # Term 2: U[0,0,2]=0, V[0,0,2]=1, W[0,0,2]=0 -> 0
    # Term 3: U[0,0,3]=1, V[0,0,3]=0, W[0,0,3]=0 -> 0
    # Term 4: U[0,0,4]=0, V[0,0,4]=-1, W[0,0,4]=1 -> 0
    # Term 5: U[0,0,5]=1, V[0,0,5]=0, W[0,0,5]=0 -> 0
    # Term 6: U[0,0,6]=-1, V[0,0,6]=1, W[0,0,6]=0 -> 0
    # Term 7: U[0,0,7]=0, V[0,0,7]=0, W[0,0,7]=-1 -> 0
    # Sum = 1. Correct!
    
    # For (i,j,k) = (0,1,0): c01 involves a00*b10 + a01*b11... wait, c01 = a00*b01 + a01*b11.
    # So T[0,0,1] = 1 (a00*b01) and T[0,1,1] = 1 (a01*b11).
    # Wait, I need to be careful with indexing.
    
    # Let me redefine: T[i,j,k] where i = row of A, j = col of A = row of B, k = col of B = col of C.
    # C[i,k] = sum_j A[i,j]*B[j,k]. So T[i,j,k] = 1 for all valid i,j,k.
    
    # For 2×2: T[i,j,k] = 1 for i,j,k in {0,1}.
    
    # The decomposition: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k.
    
    # Let me define the 7 terms for 2×2:
    # Using 0-indexed: a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
    # b00=B[0,0], b01=B[0,1], b10=B[1,0], b11=B[1,1]
    # c00=C[0,0], c01=C[0,1], c10=C[1,0], c11=C[1,1]
    
    # U indices: 0=a00, 1=a01, 2=a10, 3=a11
    # V indices: 0=b00, 1=b01, 2=b10, 3=b11
    # W indices: 0=c00 (k=0,i=0), 1=c01 (k=1,i=0), 2=c10 (k=0,i=1), 3=c11 (k=1,i=1)
    # W[k*2+i]: W[0]=c00, W[1]=c01, W[2]=c10, W[3]=c11
    
    # Term 1: m1 = (a00+a11)(b00+b11), contributes to c00,c01,c10,c11 with +1
    # U1 = [1,0,0,1], V1 = [1,0,0,1], W1 = [1,1,1,1]
    
    # Term 2: m2 = (a10+a11)b00, contributes to c10,c11 with +1
    # U2 = [0,0,1,1], V2 = [1,0,0,0], W2 = [0,0,1,1]
    
    # Term 3: m3 = a00(b01-b11), contributes to c01 with +1, c11 with -1
    # U3 = [1,0,0,0], V3 = [0,1,0,-1], W3 = [0,1,0,-1]
    
    # Term 4: m4 = a11(b10-b00), contributes to c10 with +1, c11 with -1
    # U4 = [0,0,0,1], V4 = [0,0,1,-1], W4 = [0,0,1,-1]
    
    # Term 5: m5 = (a00+a01)b11, contributes to c00 with -1, c01 with +1
    # U5 = [1,1,0,0], V5 = [0,0,0,1], W5 = [-1,1,0,0]
    
    # Term 6: m6 = (a10-a00)(b00+b01), contributes to c10 with +1, c11 with +1
    # U6 = [-1,0,1,0], V6 = [1,1,0,0], W6 = [0,0,1,1]
    
    # Term 7: m7 = (a01-a11)(b10+b11), contributes to c00 with -1, c01 with -1, c10 with +1, c11 with +1
    # Wait: c00 = m1+m4-m5-m7, so m7 contributes -1 to c00.
    # c01 = m3+m5, so m7 contributes 0 to c01.
    # c10 = m2+m4, so m7 contributes 0 to c10.
    # c11 = m1-m2+m3+m7, so m7 contributes +1 to c11.
    # U7 = [0,1,0,-1], V7 = [0,0,1,1], W7 = [-1,0,0,1]
    
    # Let me verify term 7 for (i,j,k)=(0,0,0): U[0,7]=0, V[0,7]=0, W[0,7]=-1 -> 0. Good.
    # For (i,j,k)=(0,1,0): U[1,7]=1, V[1,7]=0, W[0,7]=-1 -> 0. Good.
    # For (i,j,k)=(0,0,1): U[0,7]=0, V[1,7]=0... wait V[j*2+k] = V[0*2+1] = V[1] = 0. -> 0. Good.
    # For (i,j,k)=(1,1,1): U[3,7]=-1, V[3,7]=1, W[3,7]=1 -> -1. 
    # Sum for (1,1,1): term1: 1*1*1=1, term2: 1*0*1=0, term3: 0*(-1)*(-1)=0, term4: 1*(-1)*(-1)=1, term5: 0*1*0=0, term6: 0*0*1=0, term7: -1*1*1=-1. Sum=1. Correct!
    
    # Great, the 2×2 Strassen decomposition works. Now for 4×4, I use the Kronecker product.
    
    # The 4×4 case: A is 4×4, partitioned into 2×2 blocks A00,A01,A10,A11.
    # A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4].
    
    # The outer Strassen computes 7 "block products", each of which is a 2×2 matrix multiplication.
    # Each block product uses the 7 inner Strassen terms.
    
    # Total: 7*7 = 49 rank-1 tensors.
    
    # For each outer term s (1-7) and inner term t (1-7), we have a rank-1 tensor.
    
    # The U vector for term (s,t): the U_s vector from outer Strassen, combined with U_t from inner.
    # Specifically, U[i,j, (s,t)] = U_outer[s, block_index(i,j)] * U_inner[t, sub_index(i,j)]
    
    # This is getting very complex to code manually. Let me just generate it programmatically.
    
    # Define the 7 inner Strassen terms for 2×2:
    inner_U = [
        [1,0,0,1],  # t1
        [0,0,1,1],  # t2
        [1,0,0,0],  # t3
        [0,0,0,1],  # t4
        [1,1,0,0],  # t5
        [-1,0,1,0], # t6
        [0,1,0,-1], # t7
    ]
    inner_V = [
        [1,0,0,1],  # t1
        [1,0,0,0],  # t2
        [0,1,0,-1], # t3
        [0,0,1,-1], # t4
        [0,0,0,1],  # t5
        [1,1,0,0],  # t6
        [0,0,1,1],  # t7
    ]
    inner_W = [
        [1,1,1,1],  # t1
        [0,0,1,1],  # t2
        [0,1,0,-1], # t3
        [0,0,1,-1], # t4
        [-1,1,0,0], # t5
        [0,0,1,1],  # t6
        [-1,0,0,1], # t7
    ]
    
    # Outer Strassen for 2×2 block matrices:
    outer_U = [
        [1,0,0,1],  # s1: A00+A11
        [0,0,1,1],  # s2: A10+A11
        [1,0,0,0],  # s3: A00
        [0,0,0,1],  # s4: A11
        [1,1,0,0],  # s5: A00+A01
        [-1,0,1,0], # s6: A10-A00
        [0,1,0,-1], # s7: A01-A11
    ]
    outer_V = [
        [1,0,0,1],  # s1: B00+B11
        [1,0,0,0],  # s2: B00
        [0,1,0,-1], # s3: B01-B11
        [0,0,1,-1], # s4: B10-B00
        [0,0,0,1],  # s5: B11
        [1,1,0,0],  # s6: B00+B01
        [0,0,1,1],  # s7: B10+B11
    ]
    outer_W = [
        [1,1,1,1],  # s1
        [0,0,1,1],  # s2
        [0,1,0,-1], # s3
        [0,0,1,-1], # s4
        [-1,1,0,0], # s5
        [0,0,1,1],  # s6
        [-1,0,0,1], # s7
    ]
    
    # Now build the 49-term 4×4 decomposition:
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            # U: for each (i,j) in 4×4, U[i*4+j, r] = outer_U[s, block_ij] * inner_U[t, sub_ij]
            for i in range(4):
                for j in range(4):
                    block_i = i // 2
                    block_j = j // 2
                    sub_i = i % 2
                    sub_j = j % 2
                    U[i*4+j, r] = outer_U[s][block_i*2+block_j] * inner_U[t][sub_i*2+sub_j]
            
            # V: for each (j,k) in 4×4, V[j*4+k, r] = outer_V[s, block_jk] * inner_V[t, sub_jk]
            for j in range(4):
                for k in range(4):
                    block_j = j // 2
                    block_k = k // 2
                    sub_j = j % 2
                    sub_k = k % 2
                    V[j*4+k, r] = outer_V[s][block_j*2+block_k] * inner_V[t][sub_j*2+sub_k]
            
            # W: for each (k,i) in 4×4, W[k*4+i, r] = outer_W[s, block_ki] * inner_W[t, sub_ki]
            for k in range(4):
                for i in range(4):
                    block_k = k // 2
                    block_i = i // 2
                    sub_k = k % 2
                    sub_i = i % 2
                    W[k*4+i, r] = outer_W[s][block_k*2+block_i] * inner_W[t][sub_k*2+sub_i]
    
    return {"U": U, "V": V, "W": W}

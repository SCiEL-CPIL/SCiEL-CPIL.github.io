"""Matrix multiplication 4,4,8 - rank-minimal exact tensor decomposition using Strassen."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen 2x2: 7 multiplications
    # For A(2x2) @ B(2x2) -> C(2x2):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Strassen 4x4: Use 2x2 Strassen at block level, each block is 2x2
    # This gives rank 28 for 4x4x4 (7 multiplications, each of rank 4 for 2x2x2 schoolbook... 
    # No wait - we need to be more careful.
    
    # Actually, the standard approach: 
    # 4x4x4 can be done with rank 28 using the 2-level Strassen or direct rank-28 construction.
    
    # Let me use the known rank-28 decomposition for 4x4x4.
    # The 4x4x4 tensor has a rank-28 decomposition.
    
    # For simplicity and correctness, let me use the Kronecker product approach:
    # 4x4x4 = (2x2x2) ⊗ (2x2x2) in terms of structure
    # But Strassen at each level: rank(2x2x2)=7, so 4x4x4 via 2-level Strassen = 7*7 = 49? No.
    
    # Actually, the correct rank for 4x4x4 is 28 (proven by Alon et al. and others).
    # Let me construct it explicitly.
    
    # 4x4x4 rank-28 decomposition:
    # We can think of 4x4 as 2x2 blocks of 2x2.
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = A@B
    # 
    # Using Strassen at block level (7 block multiplications of 2x2 matrices):
    # Each 2x2 matrix product has rank 4 (schoolbook) or rank 7 is not possible for 2x2.
    # Wait, 2x2x2 has rank 7? No, 2x2x2 has rank 7 is WRONG. 
    # 2x2x2 has rank 7? Let me recall: the 2x2x2 matrix multiplication tensor has rank 7.
    # Yes! Strassen proved rank(2x2x2) = 7.
    
    # So for 4x4x4:
    # A is 4x4, B is 4x4, C is 4x4.
    # View as 2x2 blocks of 2x2.
    # Use Strassen: 7 products of 2x2 by 2x2.
    # Each 2x2x2 product has rank 7.
    # Total rank = 7 * 7 = 49? That's too high.
    
    # Actually no. The Strassen algorithm computes 7 products, but each product is a 2x2 matrix product.
    # The rank of the overall 4x4x4 tensor when using Strassen is 7 * rank(2x2x2) = 7*7 = 49?
    # But we know rank(4x4x4) <= 28, so this can't be right.
    
    # I think I'm confusing myself. Let me restart with a cleaner approach.
    
    # The rank of the 4x4x4 matrix multiplication tensor is known to be at most 28.
    # Let me just use the schoolbook for the inner part and Strassen at the outer level,
    # or better yet, let me use a known explicit rank-28 decomposition.
    
    # Alternative: use the fact that for 4x4x8, we can decompose as:
    # Treat B as [B_left, B_right] where each is 4x4.
    # C_left = A @ B_left (4x4x4)
    # C_right = A @ B_right (4x4x4)
    # 
    # If we have a rank-R decomposition for 4x4x4, then for 4x4x8 we can use
    # the same U for both, and combine V and W.
    # Rank would be R (for U) + R (for V's) + R (for W's)... no.
    # 
    # Actually, the tensor for 4x4x8 is the direct sum of two 4x4x4 tensors.
    # So rank(4x4x8) <= 2 * rank(4x4x4) = 2 * 28 = 56.
    # But we can do better by sharing the U part.
    # 
    # The U matrix for 4x4x8 has rows indexed by (i,j) for A entries, 16 rows.
    # The V matrix has rows indexed by (j,k) for B entries, 32 rows.
    # The W matrix has rows indexed by (k,i) for C entries, 32 rows.
    # 
    # In the decomposition C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # 
    # For the two halves (k=0..3 and k=4..7), we can use the same U coefficients
    # if the A part is the same. Let me define:
    # 
    # For 4x4x4: C[i,i'] = sum_{j,j'} A[i,j] * B[j,j']
    # Index: A[i*m+j], B[j*p+k], W[k*n+i]
    
    # Let me use a rank-28 decomposition for 4x4x4.
    # The 4x4x4 tensor rank 28 decomposition can be constructed from the 2x2x2 rank 7.
    
    # 2x2x2 Strassen rank-7:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e  
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4x4, view A, B, C as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # 
    # Strassen at block level:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B10-B11)
    # S4 = A11(B01-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    # 
    # Each S is a 2x2 matrix product, so each requires rank 7 to compute.
    # Total: 7 * 7 = 49. But we know it can be 28.
    
    # The key insight: when we say "each S is a 2x2 product", we mean computing a full 2x2 product.
    # But in the Strassen formula, we only need certain linear combinations.
    # 
    # Actually, the correct way: the 4x4x4 tensor, when decomposed via 2-level Strassen,
    # has rank 7*7 = 49. But the optimal is 28.
    
    # Let me just use a simpler but still improved approach:
    # Use Strassen for 4x4x4 with rank 28 (known explicit construction exists).
    
    # For the purposes of this task, let me use rank 28 for 4x4x4 and thus rank 56 for 4x4x8.
    # But actually, I can do better. Let me look at this differently.
    
    # 4x4x8: A is 4x4, B is 4x8, C is 4x8.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Think of B as having 8 columns. We can group them.
    # Use the rank-28 4x4x4 decomposition twice, sharing U.
    
    # Let me construct rank-28 for 4x4x4 first.
    # I'll use the known fact that it exists and construct it via 2x2 Strassen.
    
    # Actually, let me just use a practical approach: use schoolbook for the inner 2x2 
    # and Strassen at the outer 2x2 level, giving rank 7*4 = 28 for 4x4x4.
    # Wait, that's not right either.
    
    # Let me be very explicit. 4x4x4 tensor:
    # T[i,j,k] = delta[i,j,k] for the standard matmul tensor where C[i,k] = sum_j A[i,j]B[j,k]
    # 
    # Using 2x2 blocks: i = (i1,i0), j = (j1,j0), k = (k1,k0)
    # 
    # C[(i1,i0),(k1,k0)] = sum_{j1,j0} A[(i1,i0),(j1,j0)] * B[(j1,j0),(k1,k0)]
    # = sum_{j1,j0} A[i1,i0,j1,j0] * B[j1,j0,k1,k0]
    # 
    # = sum_{j1,j0} sum_{r=0}^{6} U_2x2[i1,j1,r] * V_2x2[j1,k1,r] * W_2x2[k1,i1,r]
    #   * (sum over i0,j0,k0 of the inner 2x2x2)
    # 
    # This is getting complicated. Let me just implement rank 28 for 4x4x4 using a known method.
    
    # Simplest correct approach: use the Kronecker product of 2x2x2 rank-7 with itself.
    # rank(4x4x4) <= rank(2x2x2)^2 = 49. Not optimal but correct.
    # For 4x4x8: rank <= 2 * 49 = 98. Still better than 128.
    
    # But I can do better. Let me use rank 28 for 4x4x4.
    # 
    # Here's a clean rank-28 construction for 4x4x4:
    # Use the 7 Strassen products, and for each, use a rank-4 (schoolbook) 2x2x2.
    # Total: 7 * 4 = 28. Yes! This works.
    
    # For each Strassen product S_r (a 2x2 matrix product), we use 4 scalar multiplications.
    # The U, V, W for each S_r are 4x4 matrices (for the 2x2x2 schoolbook).
    # 
    # Let me implement this.
    
    def strassen_2x2_decomp():
        """Return U, V, W for 2x2x2 Strassen rank-7."""
        # A = [[a, b], [c, d]] indexed as A[i,j] with i,j in {0,1}
        # B = [[e, f], [g, h]] indexed as B[j,k] with j,k in {0,1}
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # 
        # U has shape (2*2, 7) = (4, 7): U[i*2+j, r]
        # V has shape (2*2, 7) = (4, 7): V[j*2+k, r]
        # W has shape (2*2, 7) = (4, 7): W[k*2+i, r]
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a+d)(e+h): A part: a+d = A[0,0]+A[1,1], B part: e+h = B[0,0]+B[1,1]
        U[0][0] = 1; U[3][0] = 1  # a, d -> indices 0*2+0=0, 1*2+1=3
        V[0][0] = 1; V[3][0] = 1  # e, h -> indices 0*2+0=0, 1*2+1=3
        W[0][0] = 1; W[3][0] = 1  # C11 += M1: W[k*2+i], for C[0,0]: k=0,i=0 -> 0; C[1,1]: k=1,i=1 -> 3
        # Wait, C11 = M1+M4-M5+M7, so W for C[0,0] gets +M1, and C[1,1] gets +M1
        # W[0][0] = 1 (C[0,0] += M1), W[3][0] = 1 (C[1,1] += M1)
        
        # M2 = (c+d)e: A part: c+d = A[1,0]+A[1,1], B part: e = B[0,0]
        U[2][1] = 1; U[3][1] = 1  # c=A[1,0]->1*2+0=2, d=A[1,1]->3
        V[0][1] = 1  # e=B[0,0]->0
        W[2][1] = 1; W[3][1] = 1  # C21 += M2: C[1,0]->k=0,i=1->0*2+1=1; C[2,1]... 
        # C21 = M2+M4, so C[1,0] += M2. W[k*n+i] with k=0, i=1 -> 0*2+1=1
        # C22 = M1-M2+M3+M6, so C[1,1] -= M2. W[3][1] = -1
        W[1][1] = 1   # C[1,0] += M2
        W[3][1] = -1  # C[1,1] -= M2
        
        # M3 = a(f-h): A part: a = A[0,0], B part: f-h = B[0,1]-B[1,1]
        U[0][2] = 1  # a
        V[1][2] = 1; V[3][2] = -1  # f-h
        W[1][2] = 1; W[3][2] = 1  # C12 += M3: C[0,1]->k=1,i=0->1*2+0=2; C22 += M3: C[1,1]->3
        # C12 = M3+M5, C22 = M1-M2+M3+M6
        W[2][2] = 1   # C[0,1] += M3
        W[3][2] = 1   # C[1,1] += M3
        
        # M4 = d(g-e): A part: d = A[1,1], B part: g-e = B[1,0]-B[0,0]
        U[3][3] = 1  # d
        V[2][3] = 1; V[0][3] = -1  # g-e
        W[0][3] = 1; W[2][3] = 1  # C11 += M4: C[0,0]->0; C21 += M4: C[1,0]->1
        W[0][3] = 1   # C[0,0] += M4
        W[1][3] = 1   # C[1,0] += M4
        
        # M5 = (a+b)h: A part: a+b = A[0,0]+A[0,1], B part: h = B[1,1]
        U[0][4] = 1; U[1][4] = 1  # a+b
        V[3][4] = 1  # h
        W[0][4] = -1; W[2][4] = 1  # C11 -= M5: C[0,0]; C12 += M5: C[0,1]
        W[0][4] = -1  # C[0,0] -= M5
        W[2][4] = 1   # C[0,1] += M5
        
        # M6 = (c-a)(e+f): A part: c-a = A[1,0]-A[0,0], B part: e+f = B[0,0]+B[0,1]
        U[2][5] = 1; U[0][5] = -1  # c-a
        V[0][5] = 1; V[1][5] = 1  # e+f
        W[3][5] = 1  # C22 += M6: C[1,1]->3
        W[3][5] = 1
        
        # M7 = (b-d)(g+h): A part: b-d = A[0,1]-A[1,1], B part: g+h = B[1,0]+B[1,1]
        U[1][6] = 1; U[3][6] = -1  # b-d
        V[2][6] = 1; V[3][6] = 1  # g+h
        W[0][6] = 1  # C11 += M7: C[0,0]->0
        W[0][6] = 1
        
        return U, V, W
    
    def schoolbook_2x2_decomp():
        """Return U, V, W for 2x2x2 schoolbook rank-4."""
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(4)]
        W = [[0]*4 for _ in range(4)]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    r = i*4 + j*2 + k  # not right, let me use r = (i,j,k) mapping
        # Simpler: r = 0..3, r = j*2+k or similar
        # Let me use: for each (i,j,k), r = i*4 + j*2 + k... no, rank is 4 not 8.
        # Schoolbook 2x2x2: 4 multiplications, one for each (i,j,k) where... 
        # No, schoolbook for 2x2: C[i,k] = sum_j A[i,j]*B[j,k], so 4 terms.
        # r = 0,1,2,3 corresponding to (i,j,k) with i fixed? No.
        # Each multiplication is A[i,j]*B[j,k] contributing to C[i,k].
        # There are 4 such (i,j,k) triples where... actually 2*2*2 = 8 but we only need 4?
        # No, schoolbook 2x2x2 has rank 4? Let me recount.
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # Each A[i,j]*B[j,k] is one multiplication. There are 4 unique (i,j) pairs... 
        # No, there are 2*2 = 4 (i,j) and for each, 2 k values, so 8 multiplications in schoolbook.
        # But they share A[i,j] factors. The rank is still 4 because... 
        # Actually, the rank of 2x2x2 schoolbook is 4 (it's the identity-like tensor).
        # Let me just define it as 4 terms where r = j*2 + k for the 4 (j,k) pairs, 
        # but A depends on i too. This doesn't work cleanly.
        
        # The 2x2x2 tensor has rank 7 (Strassen) and the schoolbook has "rank" 4 in the 
        # sense of 4 scalar multiplications if we precompute... no.
        
        # I think I'm overcomplicating. Let me use rank 7 for 2x2x2 and 
        # total rank 7*7=49 for 4x4x4, giving 98 for 4x4x8.
        
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(4)]
        W = [[0]*4 for _ in range(4)]
        # 4 multiplications: for j in {0,1}, k in {0,1}, but we need all i.
        # This doesn't decompose cleanly with rank 4.
        # Let me abandon this and use the full schoolbook.
        return U, V, W
    
    # Given the complexity, let me use a simpler but effective approach:
    # Use Strassen for the 4x4x4 part with rank 49 (7x7 Kronecker), 
    # giving total rank 98 for 4x4x8. This is a clear improvement over 128.
    
    def strassen_2x2():
        """Strassen 2x2x2 rank-7 decomposition.
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # r=0: M1 = (a+d)(e+h) -> C[0,0]+=1, C[1,1]+=1
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=1
        
        # r=1: M2 = (c+d)*e -> C[1,0]+=1, C[1,1]-=1
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[1][1]=1; W[3][1]=-1
        
        # r=2: M3 = a*(f-h) -> C[0,1]+=1, C[1,1]+=1
        U[0][2]=1
        V[1][2]=1; V[3][2]=-1
        W[2][2]=1; W[3][2]=1
        
        # r=3: M4 = d*(g-e) -> C[0,0]+=1, C[1,0]+=1
        U[3][3]=1
        V[2][3]=1; V[0][3]=-1
        W[0][3]=1; W[1][3]=1
        
        # r=4: M5 = (a+b)*h -> C[0,0]-=1, C[0,1]+=1
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=-1; W[2][4]=1
        
        # r=5: M6 = (c-a)*(e+f) -> C[1,1]+=1
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1
        
        # r=6: M7 = (b-d)*(g+h) -> C[0,0]+=1
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        W[0][6]=1
        
        return U, V, W
    
    U2, V2, W2 = strassen_2x2()
    
    # For 4x4x4: Kronecker product of two 2x2x2 Strassen decompositions.
    # A is 4x4 = 2x2 blocks of 2x2. Index i=(i1,i0), j=(j1,j0).
    # A[i,j] = A[i1,i0,j1,j0]
    # B[j,k] = B[j1,j0,k1,k0]
    # C[i,k] = C[i1,i0,k1,k0]
    # 
    # C[i1,i0,k1,k0] = sum_{j1,j0} A[i1,i0,j1,j0]*B[j1,j0,k1,k0]
    # = sum_{j1,j0} sum_{r1=0}^6 sum_{r2=0}^6 
    #     U2[i1*2+j1,r1]*V2[j1*2+k1,r1]*W2[k1*2+i1,r1]
    #     * U2[i0*2+j0,r2]*V2[j0*2+k0,r2]*W2[k0*2+i0,r2]
    # 
    # = sum_{r1,r2} [sum_{j1} U2[i1*2+j1,r1]*V2[j1*2+k1,r1]... ]
    # 
    # This gives a rank 49 decomposition.
    # U4[i*4+j, r1*7+r2] = U2[i1*2+j1,r1]*U2[i0*2+j0,r2]
    # V4[j*4+k, r1*7+r2] = V2[j1*2+k1,r1]*V2[j0*2+k0,r2]
    # W4[k*4+i, r1*7+r2] = W2[k1*2+i1,r1]*W2[k0*2+i0,r2]
    
    def build_4x4x4():
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for i1 in range(2):
            for i0 in range(2):
                for j1 in range(2):
                    for j0 in range(2):
                        for r1 in range(7):
                            for r2 in range(7):
                                r = r1*7 + r2
                                i = i1*2 + i0
                                j = j1*2 + j0
                                U4[i*4+j][r] = U2[i1*2+j1][r1] * U2[i0*2+j0][r2]
        
        for j1 in range(2):
            for j0 in range(2):
                for k1 in range(2):
                    for k0 in range(2):
                        for r1 in range(7):
                            for r2 in range(7):
                                r = r1*7 + r2
                                j = j1*2 + j0
                                k = k1*2 + k0
                                V4[j*4+k][r] = V2[j1*2+k1][r1] * V2[j0*2+k0][r2]
        
        for k1 in range(2):
            for k0 in range(2):
                for i1 in range(2):
                    for i0 in range(2):
                        for r1 in range(7):
                            for r2 in range(7):
                                r = r1*7 + r2
                                k = k1*2 + k0
                                i = i1*2 + i0
                                W4[k*4+i][r] = W2[k1*2+i1][r1] * W2[k0*2+i0][r2]
        
        return U4, V4, W4
    
    U4, V4, W4 = build_4x4x4()
    
    # For 4x4x8: B is 4x8, split into B_left (4x4, k=0..3) and B_right (4x4, k=4..7).
    # C_left[i,k] for k=0..3: use U4, V4_left, W4_left
    # C_right[i,k] for k=4..7: use U4, V4_right, W4_right
    # 
    # U for 4x4x8: same as U4, shape (16, 49)
    # V for 4x4x8: shape (32, 98). First 49 columns for left, next 49 for right.
    #   V[j*8+k, r] for k=0..3: V4[j*4+k][r]
    #   V[j*8+k, 49+r] for k=4..7: V4[j*4+(k-4)][49+r]... wait, let me re-index.
    # 
    # Actually, let me define:
    # Rank = 98
    # U[i*4+j, r] for r=0..97
    #   For r=0..48: U[i*4+j, r] = U4[i*4+j, r]
    #   For r=49..97: U[i*4+j, r] = U4[i*4+j, r-49]
    # 
    # V[j*8+k, r]
    #   For k=0..3, r=0..48: V[j*8+k, r] = V4[j*4+k, r]
    #   For k=4..7, r=49..97: V[j*8+k, r] = V4[j*4+(k-4), r-49]
    # 
    # W[k*4+i, r] for r=0..97 (W has 32 rows since p*n=8*4=32)
    #   For k=0..3, r=0..48: W[k*4+i, r] = W4[k*4+i, r]
    #   For k=4..7, r=49..97: W[k*4+i, r] = W4[(k-4)*4+i, r-49]
    
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*4+j][r] = U4[i*4+j][r]
                U[i*4+j][49+r] = U4[i*4+j][r]
    
    for j in range(4):
        for k in range(8):
            for r in range(49):
                if k < 4:
                    V[j*8+k][r] = V4[j*4+k][r]
                else:
                    V[j*8+k][49+r] = V4[j*4+(k-4)][r]
    
    for k in range(8):
        for i in range(4):
            for r in range(49):
                if k < 4:
                    W[k*4+i][r] = W4[k*4+i][r]
                else:
                    W[k*4+i][49+r] = W4[(k-4)*4+i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

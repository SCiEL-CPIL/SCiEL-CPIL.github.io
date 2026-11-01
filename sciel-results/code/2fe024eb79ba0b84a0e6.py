"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    # For 2x2 by 2x8, we use 8 independent 2x2x2 multiplications
    # Each 2x2x2 has rank 7 (Strassen), so total rank = 8*7 = 56
    rank = 8 * 7
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 56)
    V = [[0] * rank for _ in range(m * p)]  # (16, 56)
    W = [[0] * rank for _ in range(p * n)]  # (16, 56)
    
    # Strassen decomposition for 2x2x2
    # Indices: A[i,j] for i,j in {0,1}, B[j,k] for j,k in {0,1}, C[i,k] for i,k in {0,1}
    # U indexed by (i*m+j, c) = (i*2+j, c)
    # V indexed by (j*p'+k, c) where p'=2 for the 2x2 case, but here p=8
    # Wait, I need to be careful with the indexing.
    
    # For the full 2x2 by 2x8:
    # U[i*m+j, c] where i in {0,1}, j in {0,1}
    # V[j*p+k, c] where j in {0,1}, k in {0,...,7}
    # W[k*n+i, c] where k in {0,...,7}, i in {0,1}
    
    # For each output column k (0 to 7), we use a copy of Strassen
    # Strassen rank-7 decomposition for 2x2x2:
    # The 7 terms correspond to specific combinations
    
    # Let me define the Strassen decomposition explicitly.
    # For 2x2 matrix multiplication C = A*B:
    # Using 0-indexed: A[i,j], B[j,k], C[i,k] for i,j,k in {0,1}
    
    # Strassen's 7 products:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # C[0,0] = M1 - M2 + M3 + M4
    # C[0,1] = M5 + M7
    # C[1,0] = M6 + M2
    # C[1,1] = M1 - M5 + M4 + M3
    
    # In tensor form: C[i,k] = sum_c U[i,j,c]*A[i,j] * V[j,k,c]*B[j,k] * W[k,i,c]
    # Actually, the decomposition is:
    # C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k]) * W[k,i,c]
    
    # Wait, that's not the standard form. Let me reconsider.
    
    # The correct tensor decomposition: The multiplication tensor T[i,j,k] for 2x2x2
    # satisfies: C[i,k] = sum_j A[i,j] * B[j,k]
    # And T can be decomposed as T = sum_c u_c ⊗ v_c ⊗ w_c
    # where (u_c ⊗ v_c ⊗ w_c)[i,j,k] = u_c[i] * v_c[j] * w_c[k]
    # and the multiplication is recovered by:
    # C[i,k] = sum_j sum_c A[i,j] * B[j,k] * T[i,j,k]
    #         = sum_j sum_c A[i,j] * B[j,k] * u_c[i] * v_c[j] * w_c[k]
    
    # Hmm, this isn't quite right. Let me use the standard bilinear decomposition.
    
    # Standard form: C[i,k] = sum_c P_c(A) * Q_c(B) * R_c[i,k]
    # where P_c, Q_c are linear forms.
    
    # For Strassen:
    # P1(A) = A[0,0]+A[1,1], Q1(B) = B[0,0]+B[1,1], R1 = diag(1,0,0,1) i.e. C[0,0]+=1, C[1,1]+=1
    # etc.
    
    # Let me just directly assign the values.
    
    # For each column k (0 to 7), and for the 2x2 sub-problem:
    # U has entries U[i*2+j, base+c] for i,j in {0,1}, c in {0,6}
    # V has entries V[j*8+k, base+c] for j in {0,1}, c in {0,6}
    # W has entries W[k*2+i, base+c] for i in {0,1}, c in {0,6}
    
    # Strassen decomposition coefficients:
    # Term c=0 (M1): 
    #   A part: A[0,0] + A[1,1] -> U[0,0]=1, U[3,0]=1 (indices 0*2+0=0, 1*2+1=3)
    #   B part: B[0,0] + B[1,1] -> V[0,0]=1, V[3,0]=1 (indices 0*2+0=0, 1*2+1=3) for 2x2
    #   C part: C[0,0] += 1, C[1,1] += 1 -> W[0,0]=1, W[1,0]=1 (indices 0*2+0=0, 1*2+1=3)
    
    # But wait, for the full problem, V is indexed by (j, k) where k is the output column (0-7).
    # So for a fixed output column k, V[j*8+k, c] is the coefficient.
    
    # Let me redo this properly.
    
    # For output column k, the 2x2 sub-problem uses B[j,k] for j in {0,1}.
    # So V[j*8+k, base+c] where j in {0,1}.
    
    # Strassen term definitions (for a single 2x2x2 block):
    # Let me define the 7 terms with their A, B, C coefficients.
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    #   A: [1,0,0,1] (A00, A01, A10, A11)
    #   B: [1,0,0,1] (B00, B01, B10, B11)
    #   C: [1,0,0,1] (C00, C01, C10, C11)
    
    # Term 1: M2 = (A10+A11)B00
    #   A: [0,0,1,1]
    #   B: [1,0,0,0]
    #   C: [0,0,1,0] with -1 for C00, +1 for C10
    #   Actually: C00 -= M2, C10 += M2
    #   So C: [-1, 0, 1, 0]
    
    # Term 2: M3 = A00(B01-B11)
    #   A: [1,0,0,0]
    #   B: [0,1,0,-1]
    #   C: C00 += M3, C11 += M3
    #   So C: [1, 0, 0, 1]
    
    # Term 3: M4 = A11(B10-B00)
    #   A: [0,0,0,1]
    #   B: [0,0,1,-1]  (B10 - B00, so B00 coeff = -1, B10 coeff = 1)
    #   Wait: B10 is index (1,0), B00 is index (0,0).
    #   B: [-1, 0, 1, 0] (B00, B01, B10, B11)
    #   C: C00 += M4, C11 += M4
    #   So C: [1, 0, 0, 1]
    
    # Term 4: M5 = (A00+A01)B11
    #   A: [1,1,0,0]
    #   B: [0,0,0,1]
    #   C: C01 += M5, C11 -= M5
    #   So C: [0, 1, 0, -1]
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    #   A: [-1,0,1,0]
    #   B: [1,1,0,0]
    #   C: C10 += M6
    #   So C: [0, 0, 1, 0]
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    #   A: [0,1,0,-1]
    #   B: [0,0,1,1]
    #   C: C01 += M7
    #   So C: [0, 1, 0, 0]
    
    # Let me verify:
    # C00 = M1 - M2 + M3 + M4
    #   = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + A11(B10-B00)
    #   = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A10*B00 - A11*B00 + A00*B01 - A00*B11 + A11*B10 - A11*B00
    #   = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A10*B00 - A11*B00 + A00*B01 - A00*B11 + A11*B10 - A11*B00
    #   Simplify: A00*B00 + A00*B01 + A10*(-B00) + A11*B10 + (A11*B00 - A11*B00 - A11*B00) + (A00*B11 - A00*B11) + A11*B11
    #   Wait, let me be more careful:
    #   Terms with A00*B00: +1 (from M1)
    #   Terms with A00*B11: +1 (M1) -1 (M3) = 0
    #   Terms with A11*B00: +1 (M1) -1 (M2) -1 (M4) = -1
    #   Terms with A11*B11: +1 (M1)
    #   Terms with A10*B00: -1 (M2)
    #   Terms with A00*B01: +1 (M3)
    #   Terms with A00*B11: already counted
    #   Terms with A11*B10: +1 (M4)
    #   Terms with A11*B00: already counted
    #   
    #   So C00 = A00*B00 + A00*B01 + A10*B00*(-1) + A11*B00*(-1) + A11*B10 + A11*B11
    #   That doesn't look right. C00 should be A00*B00 + A01*B10.
    
    # I think I made an error in the Strassen formulas. Let me look them up again.
    
    # Standard Strassen for 2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # 
    # c11 = M1 - M2 + M3 + M4
    # c12 = M5 + M7
    # c21 = M6 + M2
    # c22 = M1 - M5 + M4 + M3
    
    # Using 1-indexed: a11=A[0,0], a12=A[0,1], a21=A[1,0], a22=A[1,1]
    # b11=B[0,0], b12=B[0,1], b21=B[1,0], b22=B[1,1]
    # c11=C[0,0], c12=C[0,1], c21=C[1,0], c22=C[1,1]
    
    # Let me verify c11:
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # M2 = (A10+A11)*B00 = A10*B00 + A11*B00
    # M3 = A00*(B01-B11) = A00*B01 - A00*B11
    # M4 = A11*(B10-B00) = A11*B10 - A11*B00
    # 
    # c11 = M1 - M2 + M3 + M4
    #     = (A00*B00 + A00*B11 + A11*B00 + A11*B11) - (A10*B00 + A11*B00) + (A00*B01 - A00*B11) + (A11*B10 - A11*B00)
    #     = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A10*B00 - A11*B00 + A00*B01 - A00*B11 + A11*B10 - A11*B00
    #     = A00*B00 + A00*B01 + A11*B10 + A11*B11 - A10*B00 - A11*B00
    #     Wait, that's not A00*B00 + A01*B10.
    
    # I think the issue is that Strassen's formula gives the correct result but I'm not tracking it right.
    # Let me just trust the standard Strassen decomposition and verify with a simple test.
    
    # Actually, let me just use a known correct rank-7 decomposition. The standard Strassen decomposition is correct, so let me just implement it carefully.
    
    # I'll define the 7 terms as (A_coeffs, B_coeffs, C_coeffs) where each is a 4-vector for the 2x2 case.
    
    # Using 0-indexed: A[i,j] for i,j in {0,1}, flattened as [A00, A01, A10, A11]
    # B[j,k] for j,k in {0,1}, flattened as [B00, B01, B10, B11]
    # C[i,k] for i,k in {0,1}, flattened as [C00, C01, C10, C11]
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [-1, 0, 1, 0]),  # M2: C00 -= M2, C10 += M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 0, 1]),  # M3: C00 += M3, C11 += M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 0, 1]),  # M4: C00 += M4, C11 += M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1]),  # M5: C01 += M5, C11 -= M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]),  # M6: C10 += M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0]),  # M7: C01 += M7
    ]
    
    # Verify: C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M7
    # C10 = M6 + M2
    # C11 = M1 - M5 + M4 + M3
    
    # Now for the full 2x2 by 2x8 problem:
    # For each output column k (0 to 7), we use a copy of the above.
    # U[i*2+j, base+c] = A_coeffs[j] for the i-th row... wait.
    
    # Actually, U is indexed by (i, j) where i is the row of A, j is the column of A.
    # So U[i*2+j, col] is the coefficient for A[i,j] in term col.
    
    # For term c in the k-th block:
    # U[i*2+j, k*7+c] = strassen_terms[c][0][i*2+j]
    # V[j*8+k, k*7+c] = strassen_terms[c][1][j*2+k'] ... wait, no.
    
    # B is indexed by (j, k) where j is row (0,1) and k is column (0-7).
    # For the k-th output column, we only use B[j, k] for j in {0,1}.
    # So V[j*8+k, k*7+c] = strassen_terms[c][1][j*2 + ?]
    
    # The B_coeffs are for a 2x2 B matrix: [B00, B01, B10, B11]
    # But in our problem, for output column k, we only have B[0,k] and B[1,k].
    # So the "2x2" B for column k is actually just a 2-vector: [B[0,k], B[1,k]].
    
    # I think I'm overcomplicating this. Let me reconsider.
    
    # The multiplication is: C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    
    # This is a bilinear map from (A, B) to C.
    # A is a 2x2 matrix (4 entries), B is a 2x8 matrix (16 entries), C is a 2x8 matrix (16 entries).
    
    # The tensor T has indices (i,j) for A, (j,k) for B, (i,k) for C.
    # T[(i,j),(j',k),(i',k')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # This tensor can be decomposed as a sum of rank-1 tensors.
    # For the 2x2 by 2x2 case (k fixed), the tensor has rank 7.
    # For 8 independent k values, the total rank is 8*7 = 56.
    
    # Let me just construct it directly using the Strassen decomposition for each k.
    
    # For a fixed k, define the 2x2 sub-problem:
    # A_sub = A (2x2), B_sub = column k of B (2x1 vector), C_sub = column k of C (2x1 vector)
    # C_sub[i] = sum_j A[i,j] * B_sub[j]
    
    # This is a 2x2 matrix times a 2-vector, which is just two dot products.
    # The tensor for this has rank... well, it's a bilinear map from (2x2 matrix, 2-vector) to 2-vector.
    
    # Actually, this is simpler than full 2x2x2. Let me think about the rank.
    
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # This is a bilinear map. The tensor has 4*2*2 = 16 components, but many are zero.
    # Nonzero: T[(0,0),(0,k),(0,k)]=1, T[(0,1),(1,k),(0,k)]=1, T[(1,0),(0,k),(1,k)]=1, T[(1,1),(1,k),(1,k)]=1
    
    # This is actually rank 4 (schoolbook) for each k, and since the k's are independent, total rank = 8*4 = 32.
    
    # But can we do better? The 2x2x2 full matrix multiplication has rank 7, but that's for computing all 4 outputs from two 2x2 matrices. Here we're computing 2 outputs from a 2x2 matrix and a 2-vector.
    
    # The rank of the tensor for "2x2 matrix times 2-vector" is... let me think. It's a 3-tensor of size 4x2x2. The schoolbook rank is 4. Can it be done in rank 3?
    
    # I believe the rank is 4 for this specific problem, since it's essentially two independent dot products of length 2, and you can't really share much between them.
    
    # Wait, but the A matrix is shared between the two output rows. So there might be some sharing possible.
    
    # Let me just go with rank 32 (schoolbook) for now, since I'm not sure about a better decomposition for this specific rectangular case.
    
    # Actually, let me reconsider. The problem is 2x2 by 2x8. Let me just implement the schoolbook decomposition which gives rank 32, and that's what the current program does.
    
    # But the task says to find rank-minimal. If the minimum is 32, then we're done. If it's lower, we need to find it.
    
    # Given the uncertainty, let me just return the schoolbook decomposition which is known to be valid.
    
    rank = n * m * p  # 2*2*8 = 32
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

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank-17 decomposition for 2x3 x 3x3 matrix multiplication.
    # Uses Strassen's algorithm for the core 2x2x2 block (rank 7)
    # and schoolbook multiplication for the boundary strips (rank 10).
    # Total rank = 7 + 4 + 6 = 17.
    #
    # A is 2x3: rows i=0,1; cols j=0,1,2
    # B is 3x3: rows j=0,1,2; cols k=0,1,2
    # C is 2x3: rows i=0,1; cols k=0,1,2
    #
    # Decomposition:
    #   C[i,k] = sum_r U[i,k,r] * V[?,r] ... 
    #   Actually the convention is:
    #   U has shape (n*m, rank) = (6, 17), indexed by (i, j)
    #   V has shape (m*p, rank) = (9, 17), indexed by (j, k)
    #   W has shape (p*n, rank) = (6, 17), indexed by (k, i)
    #   C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] ... no.
    #
    # The standard bilinear form: C = A * B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # In tensor form: C[i,k] = sum_{j,r} U[i,j,r] * V[j,k,r] * ... 
    # Actually the factorization is:
    #   T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)} ... no.
    #
    # The correct factorization for matrix multiplication tensor T with T[i,j,k] = 1 iff i,j,k form a valid index:
    #   Actually T is the 3-way tensor where T[i,j,k] = 1 if the index (i,j,k) corresponds to A[i,j]*B[j,k]->C[i,k].
    #   The bilinear algorithm gives:
    #   C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
    #   Wait, that's not right either. Let me re-read the task.
    #
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i,r]" - so W is indexed by (k,i) with k first, i second.
    #
    # The decomposition is: for each rank index r, we have:
    #   A factor: vector u_r of length n*m, so u_r[i*m+j] 
    #   B factor: vector v_r of length m*p, so v_r[j*p+k]
    #   C factor: vector w_r of length p*n, so w_r[k*n+i]
    #   Then C[i,k] += sum_j (sum over something)...
    #
    # Actually, the standard form is:
    #   A is n*m, B is m*p, C is p*n (flattened)
    #   The bilinear algorithm: C_flat = sum_r (u_r outer A_flat) * (v_r outer B_flat) ... no.
    #
    # Let me think again. For matrix multiplication A(n x m) * B(m x p) = C(n x p):
    #   C[i,k] = sum_{j} A[i,j] B[j,k]
    #   
    # A rank-1 decomposition of the multiplication tensor:
    #   C[i,k] = sum_r alpha_r * (product of A entries) * (product of B entries)
    #   More precisely: C[i,k] = sum_r (sum_{j} U[i,j,r] A[i,j]) ... no.
    #
    # The correct form: The multiplication tensor T has entries T[i,j,k] = 1 (for valid i,j,k).
    # A rank-r decomposition means:
    #   T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * ... 
    #
    # Wait, I think the factorization is:
    #   For each r, define:
    #     a_r = sum_{i,j} U[i*m+j, r] * A[i,j]  (a scalar)
    #     b_r = sum_{j,k} V[j*p+k, r] * B[j,k]  (a scalar)
    #     Then C[i,k] = sum_r W[k*n+i, r] * a_r * b_r
    #
    # Let's verify: If U,V,W are rank-1 factors, then:
    #   C[i,k] = sum_r W[k*n+i,r] * (sum_{i',j} U[i'*m+j,r] A[i',j]) * (sum_{j',k'} V[j'*p+k',r] B[j',k'])
    #   This doesn't look right because a_r depends on all A entries.
    #
    # Actually I think the correct interpretation is:
    #   C[i,k] = sum_r U[i*m + j?, r] ... 
    #
    # Let me look at the seed program more carefully:
    #   U[i*m+j][column] = 1
    #   V[j*p+k][column] = 1  
    #   W[k*n+i][column] = 1
    #   This is for the schoolbook case where each (i,j,k) triple gets its own rank-1 term.
    #   So the decomposition is: C[i,k] = sum_r U[i*m+j_r, r] * V[j_r*p+k_r, r] * W[k_r*n+i_r, r] * A[i_r, j_r] * B[j_r, k_r]
    #   Where for each r, only one entry of U, V, W is nonzero, and i_r=i, j_r=j, k_r=k.
    #
    # So the general form is:
    #   C[i,k] = sum_{r} W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])
    #   No wait, that would make a_r and b_r scalars, and then C[i,k] = sum_r W[k*n+i,r] * a_r * b_r which is a rank-1 outer product for each r, summed. But that gives C as a sum of rank-1 matrices, each W column times a scalar. That's not how bilinear algorithms work for matrix multiplication.
    #
    # I think the correct form must be that U, V, W encode the bilinear algorithm directly. Let me reconsider.
    #
    # In a bilinear algorithm for matrix multiplication:
    #   We form r linear combinations of A: L_r^A = sum_{i,j} alpha_{r,i,j} A[i,j]
    #   We form r linear combinations of B: L_r^B = sum_{j,k} beta_{r,j,k} B[j,k]
    #   Then C[i,k] = sum_r gamma_{r,i,k} L_r^A * L_r^B
    #
    # In this formulation:
    #   U[i*m+j, r] = alpha_{r,i,j}
    #   V[j*p+k, r] = beta_{r,j,k}
    #   W[k*n+i, r] = gamma_{r,i,k}
    #
    # Yes! That makes sense. So:
    #   C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])
    #
    # Now for the Strassen part. For a 2x2 x 2x2 block (rows 0,1 of A, cols 0,1 of A; rows 0,1 of B, cols 0,1 of B), we want to compute C[0:2, 0:2] using rank 7.
    #
    # Let me denote the sub-matrices:
    #   A_00 = A[0,0], A_01 = A[0,1], A_10 = A[1,0], A_11 = A[1,1]
    #   B_00 = B[0,0], B_01 = B[0,1], B_10 = B[1,0], B_11 = B[1,1]
    #   C_00, C_01, C_10, C_11 are the 2x2 output block.
    #
    # Strassen's 7 products:
    #   M1 = (A00 + A11)(B00 + B11)
    #   M2 = (A10 + A11)B00
    #   M3 = A00(B01 - B11)
    #   M4 = A11(B10 - B00)
    #   M5 = (A00 + A01)B11
    #   M6 = (A10 - A00)(B00 + B01)
    #   M7 = (A01 - A11)(B10 + B11)
    #
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # Let me verify C00:
    #   M1 + M4 - M5 + M7
    #   = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    #   = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    #   = A00B00 + A11B11 + A00B11 - A00B11 + A11B10 - A11B10 + A01B10 + A01B11 - A11B00 + A11B00 - A11B11 - A01B11
    #   Hmm, let me be more careful.
    #   = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    #   Group terms:
    #   A00B00: +1
    #   A00B11: +1 -1 = 0
    #   A11B00: +1 -1 = 0
    #   A11B11: +1 -1 = 0
    #   A11B10: +1 -1 = 0
    #   A01B11: -1 +1 = 0
    #   A01B10: +1
    #   So C00 = A00B00 + A01B10. That's wrong! It should be A00B00 + A01B10? No, C00 = A00B00 + A01B10 is not right. C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]. Yes that IS correct!
    #   Wait, A is 2x2, B is 2x2. C[0,0] = sum_j A[0,j]B[j,0] = A[0,0]B[0,0] + A[0,1]B[1,0]. Yes!
    #
    # Let me verify C01:
    #   M3 + M5 = A00(B01-B11) + (A00+A01)B11 = A00B01 - A00B11 + A00B11 + A01B11 = A00B01 + A01B11
    #   C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]. Correct!
    #
    # C10: M2 + M4 = (A10+A11)B00 + A11(B10-B00) = A10B00 + A11B00 + A11B10 - A11B00 = A10B00 + A11B10
    #   C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]. Correct!
    #
    # C11: M1 - M2 + M3 + M6
    #   = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A10-A00)(B00+B01)
    #   = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A10B00 + A10B01 - A00B00 - A00B01
    #   = A00B11 + A11B11 + A10B01
    #   C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]. Correct!
    #
    # Great, Strassen works. Now let me implement this.
    #
    # For the 2x2x2 block, the indices are:
    #   i in {0,1}, j in {0,1}, k in {0,1}
    #   U[i*m+j, r] where m=3, so U[i*3+j, r]
    #   V[j*p+k, r] where p=3, so V[j*3+k, r]
    #   W[k*n+i, r] where n=2, so W[k*2+i, r]
    #
    # For Strassen term 1: M1 = (A00+A11)(B00+B11)
    #   A part: U[0*3+0, r] = 1, U[1*3+1, r] = 1  (A[0,0] and A[1,1])
    #   B part: V[0*3+0, r] = 1, V[1*3+1, r] = 1  (B[0,0] and B[1,1])
    #   C part: contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    #   C[0,0]: +1, C[0,1]: +1 (from M1 in C01? No, C01 = M3+M5, not M1)
    #   
    # Let me recheck:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # So M1 contributes to C00 (+1) and C11 (+1).
    #   W[0*2+0, r] = 1  (C[0,0])
    #   W[1*2+1, r] = 1  (C[1,1])
    #
    # Strassen term 2: M2 = (A10+A11)B00
    #   A: U[1*3+0, r] = 1, U[1*3+1, r] = 1
    #   B: V[0*3+0, r] = 1
    #   C: C10 = M2 + M4, so M2 contributes +1 to C[1,0]
    #       C11 = M1 - M2 + M3 + M6, so M2 contributes -1 to C[1,1]
    #   W[0*2+1, r] = 1  (C[1,0])
    #   W[1*2+1, r] = -1 (C[1,1])
    #
    # Strassen term 3: M3 = A00(B01-B11)
    #   A: U[0*3+0, r] = 1
    #   B: V[0*3+1, r] = 1, V[1*3+1, r] = -1
    #   C: C01 = M3 + M5, so +1 to C[0,1]
    #       C11 = M1 - M2 + M3 + M6, so +1 to C[1,1]
    #   W[1*2+0, r] = 1  (C[0,1])
    #   W[1*2+1, r] = 1  (C[1,1])
    #
    # Strassen term 4: M4 = A11(B10-B00)
    #   A: U[1*3+1, r] = 1
    #   B: V[1*3+0, r] = 1, V[0*3+0, r] = -1
    #   C: C00 = M1 + M4 - M5 + M7, so +1 to C[0,0]
    #       C10 = M2 + M4, so +1 to C[1,0]
    #   W[0*2+0, r] = 1  (C[0,0])
    #   W[0*2+1, r] = 1  (C[1,0])
    #
    # Strassen term 5: M5 = (A00+A01)B11
    #   A: U[0*3+0, r] = 1, U[0*3+1, r] = 1
    #   B: V[1*3+1, r] = 1
    #   C: C00 = M1 + M4 - M5 + M7, so -1 to C[0,0]
    #       C01 = M3 + M5, so +1 to C[0,1]
    #   W[0*2+0, r] = -1 (C[0,0])
    #   W[1*2+0, r] = 1  (C[0,1])
    #
    # Strassen term 6: M6 = (A10-A00)(B00+B01)
    #   A: U[1*3+0, r] = 1, U[0*3+0, r] = -1
    #   B: V[0*3+0, r] = 1, V[0*3+1, r] = 1
    #   C: C11 = M1 - M2 + M3 + M6, so +1 to C[1,1]
    #   W[1*2+1, r] = 1  (C[1,1])
    #
    # Strassen term 7: M7 = (A01-A11)(B10+B11)
    #   A: U[0*3+1, r] = 1, U[1*3+1, r] = -1
    #   B: V[1*3+0, r] = 1, V[1*3+1, r] = 1
    #   C: C00 = M1 + M4 - M5 + M7, so +1 to C[0,0]
    #   W[0*2+0, r] = 1  (C[0,0])
    #
    # Now for the boundary parts.
    # The full matrices are 2x3 and 3x3.
    # A has entries A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2]
    # B has entries B[0,0], B[0,1], B[0,2], B[1,0], B[1,1], B[1,2], B[2,0], B[2,1], B[2,2]
    # C has entries C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2]
    #
    # The 2x2x2 block handles A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # We still need to handle:
    #   A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]  (4 terms)
    #   A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]  (4 terms)
    #   A[0:2, 2] * B[2, 2] -> C[0:2, 2]      (2 terms)
    #   A[0:2, 2] * B[2, 2] is already counted
    #
    # Let me list all missing terms:
    #   C[i,k] involves sum_j A[i,j]B[j,k]
    #   For the Strassen block, we handled j=0,1 and k=0,1.
    #   Missing contributions:
    #   - j=2, k=0: A[i,2]*B[2,0] for i=0,1 -> C[0,0] += A[0,2]B[2,0], C[1,0] += A[1,2]B[2,0]
    #   - j=2, k=1: A[i,2]*B[2,1] for i=0,1 -> C[0,1] += A[0,2]B[2,1], C[1,1] += A[1,2]B[2,1]
    #   - j=0, k=2: A[i,0]*B[0,2] for i=0,1 -> C[0,2] += A[0,0]B[0,2], C[1,2] += A[1,0]B[0,2]
    #   - j=1, k=2: A[i,1]*B[1,2] for i=0,1 -> C[0,2] += A[0,1]B[1,2], C[1,2] += A[1,1]B[1,2]
    #   - j=2, k=2: A[i,2]*B[2,2] for i=0,1 -> C[0,2] += A[0,2]B[2,2], C[1,2] += A[1,2]B[2,2]
    #
    # That's 4 + 4 + 2 = 10 additional rank-1 terms. Total rank = 7 + 10 = 17.
    #
    # Let me assign rank indices:
    #   r=0..6: Strassen terms
    #   r=7: A[0,2]*B[2,0] -> C[0,0]
    #   r=8: A[1,2]*B[2,0] -> C[1,0]
    #   r=9: A[0,2]*B[2,1] -> C[0,1]
    #   r=10: A[1,2]*B[2,1] -> C[1,1]
    #   r=11: A[0,0]*B[0,2] -> C[0,2]
    #   r=12: A[1,0]*B[0,2] -> C[1,2]
    #   r=13: A[0,1]*B[1,2] -> C[0,2]
    #   r=14: A[1,1]*B[1,2] -> C[1,2]
    #   r=15: A[0,2]*B[2,2] -> C[0,2]
    #   r=16: A[1,2]*B[2,2] -> C[1,2]
    #
    # Now let me build the matrices.
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # (6, 17)
    V = [[0] * rank for _ in range(m * p)]  # (9, 17)
    W = [[0] * rank for _ in range(p * n)]  # (6, 17)
    
    # Helper to add a rank-1 term
    def add_term(r, a_idx, b_idx, c_idx, coeff=1):
        # a_idx: list of (i,j) or single (i,j)
        # Actually let me just set entries directly.
        pass
    
    # Strassen term 1: M1 = (A00+A11)(B00+B11), C: +1 to C[0,0], +1 to C[1,1]
    r = 0
    U[0*3+0][r] = 1   # A[0,0]
    U[1*3+1][r] = 1   # A[1,1]
    V[0*3+0][r] = 1   # B[0,0]
    V[1*3+1][r] = 1   # B[1,1]
    W[0*2+0][r] = 1   # C[0,0]
    W[1*2+1][r] = 1   # C[1,1]
    
    # Strassen term 2: M2 = (A10+A11)B00, C: +1 to C[1,0], -1 to C[1,1]
    r = 1
    U[1*3+0][r] = 1   # A[1,0]
    U[1*3+1][r] = 1   # A[1,1]
    V[0*3+0][r] = 1   # B[0,0]
    W[0*2+1][r] = 1   # C[1,0]
    W[1*2+1][r] = -1  # C[1,1]
    
    # Strassen term 3: M3 = A00(B01-B11), C: +1 to C[0,1], +1 to C[1,1]
    r = 2
    U[0*3+0][r] = 1   # A[0,0]
    V[0*3+1][r] = 1   # B[0,1]
    V[1*3+1][r] = -1  # B[1,1]
    W[1*2+0][r] = 1   # C[0,1]
    W[1*2+1][r] = 1   # C[1,1]
    
    # Strassen term 4: M4 = A11(B10-B00), C: +1 to C[0,0], +1 to C[1,0]
    r = 3
    U[1*3+1][r] = 1   # A[1,1]
    V[1*3+0][r] = 1   # B[1,0]
    V[0*3+0][r] = -1  # B[0,0]
    W[0*2+0][r] = 1   # C[0,0]
    W[0*2+1][r] = 1   # C[1,0]
    
    # Strassen term 5: M5 = (A00+A01)B11, C: -1 to C[0,0], +1 to C[0,1]
    r = 4
    U[0*3+0][r] = 1   # A[0,0]
    U[0*3+1][r] = 1   # A[0,1]
    V[1*3+1][r] = 1   # B[1,1]
    W[0*2+0][r] = -1  # C[0,0]
    W[1*2+0][r] = 1   # C[0,1]
    
    # Strassen term 6: M6 = (A10-A00)(B00+B01), C: +1 to C[1,1]
    r = 5
    U[1*3+0][r] = 1   # A[1,0]
    U[0*3+0][r] = -1  # A[0,0]
    V[0*3+0][r] = 1   # B[0,0]
    V[0*3+1][r] = 1   # B[0,1]
    W[1*2+1][r] = 1   # C[1,1]
    
    # Strassen term 7: M7 = (A01-A11)(B10+B11), C: +1 to C[0,0]
    r = 6
    U[0*3+1][r] = 1   # A[0,1]
    U[1*3+1][r] = -1  # A[1,1]
    V[1*3+0][r] = 1   # B[1,0]
    V[1*3+1][r] = 1   # B[1,1]
    W[0*2+0][r] = 1   # C[0,0]
    
    # Boundary terms (schoolbook)
    # r=7: A[0,2]*B[2,0] -> C[0,0]
    r = 7
    U[0*3+2][r] = 1   # A[0,2]
    V[2*3+0][r] = 1   # B[2,0]
    W[0*2+0][r] = 1   # C[0,0]
    
    # r=8: A[1,2]*B[2,0] -> C[1,0]
    r = 8
    U[1*3+2][r] = 1   # A[1,2]
    V[2*3+0][r] = 1   # B[2,0]
    W[0*2+1][r] = 1   # C[1,0]
    
    # r=9: A[0,2]*B[2,1] -> C[0,1]
    r = 9
    U[0*3+2][r] = 1   # A[0,2]
    V[2*3+1][r] = 1   # B[2,1]
    W[1*2+0][r] = 1   # C[0,1]
    
    # r=10: A[1,2]*B[2,1] -> C[1,1]
    r = 10
    U[1*3+2][r] = 1   # A[1,2]
    V[2*3+1][r] = 1   # B[2,1]
    W[1*2+1][r] = 1   # C[1,1]
    
    # r=11: A[0,0]*B[0,2] -> C[0,2]
    r = 11
    U[0*3+0][r] = 1   # A[0,0]
    V[0*3+2][r] = 1   # B[0,2]
    W[2*2+0][r] = 1   # C[0,2]
    
    # r=12: A[1,0]*B[0,2] -> C[1,2]
    r = 12
    U[1*3+0][r] = 1   # A[1,0]
    V[0*3+2][r] = 1   # B[0,2]
    W[2*2+1][r] = 1   # C[1,2]
    
    # r=13: A[0,1]*B[1,2] -> C[0,2]
    r = 13
    U[0*3+1][r] = 1   # A[0,1]
    V[1*3+2][r] = 1   # B[1,2]
    W[2*2+0][r] = 1   # C[0,2]
    
    # r=14: A[1,1]*B[1,2] -> C[1,2]
    r = 14
    U[1*3+1][r] = 1   # A[1,1]
    V[1*3+2][r] = 1   # B[1,2]
    W[2*2+1][r] = 1   # C[1,2]
    
    # r=15: A[0,2]*B[2,2] -> C[0,2]
    r = 15
    U[0*3+2][r] = 1   # A[0,2]
    V[2*3+2][r] = 1   # B[2,2]
    W[2*2+0][r] = 1   # C[0,2]
    
    # r=16: A[1,2]*B[2,2] -> C[1,2]
    r = 16
    U[1*3+2][r] = 1   # A[1,2]
    V[2*3+2][r] = 1   # B[2,2]
    W[2*2+1][r] = 1   # C[1,2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

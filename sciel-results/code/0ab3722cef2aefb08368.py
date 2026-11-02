"""Matrix multiplication 2,4,5 — rank-minimal exact tensor decomposition.

Strategy:
  - Use Strassen's rank-7 algorithm for 2x2 sub-problems.
  - Split the 2x4 * 4x5 product into overlapping 2x2 blocks.
  - For 2x4 * 4x5: A is 2x4, B is 4x5.
    Split A into [A1 A2] where A1 is 2x2, A2 is 2x2.
    Split B into [[B1],[B2]] where B1 is 2x5, B2 is 2x5.
    C = A1*B1 + A2*B2.
    
    Each 2x2 * 2x5 product: split B into two 2x2 blocks and one 2x1? No.
    Actually 2x2 * 2x5: we can split the 5 columns into 2+2+1.
    That gives 3 sub-products of type 2x2*2x2 (rank 7 each) and 2x2*2x1 (rank 4 each).
    Total for one 2x2*2x5: 2*7 + 4 = 18.
    Two such: 36.
    
  - But we can do better by sharing. Let's think more carefully.
    
  A (2x4) = [A_left (2x2) | A_right (2x2)]
  B (4x5) = [B_top (2x5); B_bottom (2x5)]
  
  C = A_left * B_top + A_right * B_bottom
  
  For A_left * B_top (2x2 * 2x5):
    Split B_top into B_top_left (2x2) and B_top_right (2x3).
    Hmm, 2x3 is awkward. Let's split 5 into 2+3.
    2x2 * 2x2: rank 7 (Strassen)
    2x2 * 2x3: split 3 into 2+1, so 2x2*2x2 (rank 7) + 2x2*2x1 (rank 4) = 11
    Total: 7 + 11 = 18
    
  Similarly A_right * B_bottom: 18
  Total: 36.
  
  Can we do better? Let's try a different partition.
  
  Actually, let's use the known result: for 2x4 * 4x5, the rank is at most 36 via Strassen-based splitting.
  The schoolbook is 2*4*5 = 40. So 36 is an improvement.
  
  Let me implement this carefully.
  
  Strassen for 2x2 * 2x2:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
  Each Mi is a scalar product of two "vectors" (linear combinations of entries).
  In tensor decomposition terms, each Mi corresponds to one rank-1 term.
  
  For 2x2 * 2x1 (i.e., B is 2x1, so B11 and B21 are scalars, B12=B22=0):
    We can just do the 4 products directly: A11*B11, A12*B11, A21*B21, A22*B21. Rank 4.
    Or use Strassen-like: but 2x2*2x1 is just a matrix-vector, rank 4 is minimal.
    
  Let me code the 36-rank construction.
  
  Indices:
    A: 2x4, indices (i, j) where i in {0,1}, j in {0,1,2,3}
    B: 4x5, indices (j, k) where j in {0,1,2,3}, k in {0,1,2,3,4}
    C: 2x5, indices (i, k) where i in {0,1}, k in {0,1,2,3,4}
    
  W convention: W[k*n + i, r] where n=2 (rows of A / columns of C).
  So W has shape (p*n, rank) = (5*2, rank) = (10, rank).
  W[k*n+i, r] = W[2*k + i, r].
  
  U: shape (n*m, rank) = (2*4, rank) = (8, rank). U[i*m + j, r] = U[4*i + j, r].
  V: shape (m*p, rank) = (4*5, rank) = (20, rank). V[j*p + k, r] = V[5*j + k, r].
  
  Now, the decomposition:
  C[i,k] = sum_r U[4*i + j_u, r] * V[5*j_v + k_v, r] * W[2*k + i, r] ... no, that's not right.
  
  Actually the tensor decomposition is:
  T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
  
  Wait, the standard form is:
  For multiplying A (n x m) by B (m x p) to get C (n x p):
  C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
  
  The tensor T has indices (i, j, k) and T[i,j,k] = delta... no.
  Actually the multiplication tensor is:
  T[i, j, k] = 1 if we're computing C[i,k] using A[i,j] and B[j,k].
  
  In the factorization:
  C[i,k] = sum_{r} (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * ... 
  
  No wait. The standard tensor decomposition for matrix multiplication:
  T_{ijk} = sum_r U_{i,r} V_{j,r} W_{k,r}
  
  But here the dimensions are coupled. Let me re-read the task.
  
  "U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]"
  
  So:
  C[i,k] = sum_{j=0}^{m-1} A[i,j] B[j,k]
         = sum_{r} (sum_j U[4i+j, r] A[i,j]) ... no, that's not how it works.
  
  Actually the correct interpretation:
  The multiplication map is a bilinear map: (A, B) -> C.
  C[i,k] = sum_j A[i,j] B[j,k]
  
  This can be written as:
  C[i,k] = sum_r U_r(A) * V_r(B) * W_r... no, W is not multiplied, it's the output coefficient.
  
  Let me think again. In tensor decomposition of matrix multiplication:
  We have a 3-tensor T with T[i,j,k] = 1 (it's the identity in the appropriate sense).
  Actually T[i,j,k] = delta... no.
  
  The multiplication tensor for n x m * m x p has entries:
  T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
  T[i,j,k] = 1 if the (i,k) entry of the product uses A[i,j] and B[j,k].
  
  In the standard bilinear form:
  C[i,k] = sum_{j} A[i,j] B[j,k]
  
  This is a sum over j of products A[i,j] * B[j,k]. Each term is a rank-1 "slice" in the sense that for fixed j, the contribution to all (i,k) is A[i,j] * B[j,k].
  
  In the tensor decomposition with factor matrices:
  C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
  
  No, that's not right either. Let me look at the schoolbook:
  U[i*m+j, r] = 1 when r = i*m*p + j*p + k... no.
  
  In the seed code:
  for i, j, k: U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
  
  So for each (i,j,k) triple, there's a rank-1 term. The contribution to C[i,k] from this term is:
  U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * ... and W[k*n+i, r] multiplies the output.
  
  Actually the formula is:
  C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
  
  No, that doesn't match dimensions. Let me re-examine.
  
  In the schoolbook, for a fixed (i,j,k), U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
  All other entries in column r are 0.
  
  So the contribution to C[i',k'] from column r is:
  W[k'*n+i', r] * (sum_j' U[i'*m+j', r] * A[i',j']) * (sum_j' V[j'*p+k', r] * B[j',k'])
  
  For the schoolbook column corresponding to (i,j,k):
  W[k'*n+i', r] = 1 only if k'=k and i'=i.
  U[i'*m+j', r] = 1 only if i'=i and j'=j.
  V[j'*p+k', r] = 1 only if j'=j and k'=k.
  
  So the contribution to C[i',k'] is:
  If i'=i, k'=k: W=1, (sum = A[i,j]), (sum = B[j,k]), product = A[i,j]*B[j,k].
  Otherwise: 0.
  
  So yes: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
  
  But wait, in the schoolbook, for column r corresponding to (i,j,k):
  sum_j U[i*m+j, r] * A[i,j] = A[i,j] (only j=j is nonzero)
  sum_j V[j*p+k, r] * B[j,k] = B[j,k] (only j=j is nonzero)
  
  So C[i,k] gets contribution W[k*n+i, r] * A[i,j] * B[j,k] = 1 * A[i,j] * B[j,k].
  
  And summing over all r (i.e., all (i,j,k) triples) gives C[i,k] = sum_j A[i,j] B[j,k]. Correct.
  
  OK so the formula is:
  C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
  
  Now for Strassen-based decomposition, we need to express the 2x2*2x2 product using 7 rank-1 terms.
  
  For a 2x2 * 2x2 sub-problem with A_sub (2x2), B_sub (2x2), C_sub (2x2):
  C_sub[i,k] = sum_j A_sub[i,j] B_sub[j,k]
  
  Using Strassen, we define 7 linear combinations:
  For the U factor: each of the 7 terms has a linear combination of A_sub entries.
  For the V factor: each has a linear combination of B_sub entries.
  For the W factor: each has a coefficient for each C_sub entry.
  
  Let me define the 7 Strassen products for 2x2*2x2:
  Let A_sub = [[a00, a01], [a10, a11]], B_sub = [[b00, b01], [b10, b11]].
  
  M1 = (a00+a11)(b00+b11)
  M2 = (a10+a11)b00
  M3 = a00(b01-b11)
  M4 = a11(b10-b00)
  M5 = (a00+a01)b11
  M6 = (a10-a00)(b00+b01)
  M7 = (a01-a11)(b10+b11)
  
  C00 = M1 + M4 - M5 + M7
  C01 = M3 + M5
  C10 = M2 + M4
  C11 = M1 - M2 + M3 + M6
  
  Each Mt is a product of a linear combo of A entries and a linear combo of B entries.
  So in the tensor decomposition:
  For term t (t=1..7):
  U factor: the coefficients of a00, a01, a10, a11 in the A-part of Mt.
  V factor: the coefficients of b00, b01, b10, b11 in the B-part of Mt.
  W factor: the coefficients for C00, C01, C10, C11 in the expression of C in terms of M's.
  
  For example, M1:
  A-part: a00 + a11 -> coefficients (1, 0, 0, 1) for (a00, a01, a10, a11)
  B-part: b00 + b11 -> coefficients (1, 0, 0, 1) for (b00, b01, b10, b11)
  W-part: M1 appears in C00 with +1, C11 with +1. So W = (1, 0, 0, 1) for (C00, C01, C10, C11).
  
  Now I need to map these to the global indices.
  
  For the full 2x4 * 4x5 problem:
  A: rows i in {0,1}, cols j in {0,1,2,3}
  B: rows j in {0,1,2,3}, cols k in {0,1,2,3,4}
  C: rows i in {0,1}, cols k in {0,1,2,3,4}
  
  Split: A = [A_L | A_R] where A_L is cols 0,1 and A_R is cols 2,3.
  B = [B_T; B_B] where B_T is rows 0,1 and B_B is rows 2,3.
  
  C = A_L * B_T + A_R * B_B
  
  For A_L * B_T (2x2 * 2x5):
  Split B_T into B_TL (cols 0,1) and B_TR (cols 2,3,4).
  A_L * B_T = A_L * B_TL + A_L * B_TR
  
  A_L * B_TL: 2x2 * 2x2, use Strassen (7 terms)
  A_L * B_TR: 2x2 * 2x3. Split B_TR into B_TR1 (cols 2,3) and B_TR2 (col 4).
  A_L * B_TR1: 2x2 * 2x2, Strassen (7 terms)
  A_L * B_TR2: 2x2 * 2x1, direct (4 terms)
  
  Similarly for A_R * B_B:
  A_R * B_BL: 2x2 * 2x2, Strassen (7 terms)
  A_R * B_BR1: 2x2 * 2x2, Strassen (7 terms)
  A_R * B_BR2: 2x2 * 2x1, direct (4 terms)
  
  Total: 7+7+4 + 7+7+4 = 36.
  
  Now let me code this up.
  
  For each sub-problem, I need to define the U, V, W contributions.
  
  Let me write a helper that, given a 2x2*2x2 Strassen term, adds to U, V, W.
  
  Actually, let me just build the full 36-column matrices directly.
  
  U: 8 x 36
  V: 20 x 36
  W: 10 x 36
  
  For each of the 6 sub-problems, I'll add 7 or 4 columns.
  
  Sub-problem 1: A_L * B_TL (2x2 * 2x2)
  A_L entries: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices: 0,1,4,5
  B_TL entries: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices: 0,1,5,6
  C entries: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices: 0,1,2,3
  
  Sub-problem 2: A_L * B_TR1 (2x2 * 2x2)
  A_L entries: same as above -> U indices: 0,1,4,5
  B_TR1 entries: B[0,2], B[0,3], B[1,2], B[1,3] -> V indices: 2,3,7,8
  C entries: C[0,2], C[0,3], C[1,2], C[1,3] -> W indices: 4,5,6,7
  
  Sub-problem 3: A_L * B_TR2 (2x2 * 2x1)
  A_L entries: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices: 0,1,4,5
  B_TR2 entries: B[0,4], B[1,4] -> V indices: 4,9
  C entries: C[0,4], C[1,4] -> W indices: 8,9
  
  Sub-problem 4: A_R * B_BL (2x2 * 2x2)
  A_R entries: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices: 2,3,6,7
  B_BL entries: B[2,0], B[2,1], B[3,0], B[3,1] -> V indices: 10,11,15,16
  C entries: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices: 0,1,2,3
  
  Sub-problem 5: A_R * B_BR1 (2x2 * 2x2)
  A_R entries: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices: 2,3,6,7
  B_BR1 entries: B[2,2], B[2,3], B[3,2], B[3,3] -> V indices: 12,13,17,18
  C entries: C[0,2], C[0,3], C[1,2], C[1,3] -> W indices: 4,5,6,7
  
  Sub-problem 6: A_R * B_BR2 (2x2 * 2x1)
  A_R entries: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices: 2,3,6,7
  B_BR2 entries: B[2,4], B[3,4] -> V indices: 14,19
  C entries: C[0,4], C[1,4] -> W indices: 8,9
  
  Now for each Strassen sub-problem, the 7 terms have specific U, V, W patterns.
  
  Let me define the Strassen coefficients:
  For a 2x2*2x2 with A entries indexed (00,01,10,11) and B entries (00,01,10,11):
  
  Term 1: A-part = (1,0,0,1), B-part = (1,0,0,1), W = (1,0,0,1) [C00,C01,C10,C11]
  Term 2: A-part = (0,0,1,1), B-part = (1,0,0,0), W = (0,0,1,0)
  Term 3: A-part = (1,0,0,0), B-part = (0,1,0,-1), W = (0,1,0,0)
  Term 4: A-part = (0,0,0,1), B-part = (0,0,1,-1), W = (1,0,1,0)
  Term 5: A-part = (1,1,0,0), B-part = (0,0,0,1), W = (-1,1,0,0)
  Term 6: A-part = (0,0,1,-1), B-part = (1,1,0,0), W = (0,0,0,1)
  Term 7: A-part = (0,1,0,-1), B-part = (0,0,1,1), W = (1,0,0,0)
  
  Let me verify:
  C00 = M1 + M4 - M5 + M7
  W for M1: C00=1, M4: C00=1, M5: C00=-1, M7: C00=1. Sum: 1+1-1+1=2? No, that's not right.
  
  Wait, W is the coefficient. For term 1, W[0]=1 means M1 contributes +1 to C00.
  For term 4, W[0]=1 means M4 contributes +1 to C00.
  For term 5, W[0]=-1 means M5 contributes -1 to C00.
  For term 7, W[0]=1 means M7 contributes +1 to C00.
  So C00 = 1*M1 + 1*M4 + (-1)*M5 + 1*M7 = M1 + M4 - M5 + M7. Correct.
  
  C01 = M3 + M5: term 3 W[1]=1, term 5 W[1]=1. Correct.
  C10 = M2 + M4: term 2 W[2]=1, term 4 W[2]=1. Correct.
  C11 = M1 - M2 + M3 + M6: term 1 W[3]=1, term 2 W[3]=-1, term 3 W[3]=1, term 6 W[3]=1. Correct.
  
  Good.
  
  For the 2x2*2x1 direct case:
  C0 = A00*B0 + A01*B1
  C1 = A10*B0 + A11*B1
  
  4 terms:
  Term 1: A-part = (1,0,0,0), B-part = (1,0), W = (1,0)
  Term 2: A-part = (0,1,0,0), B-part = (0,1), W = (1,0)
  Term 3: A-part = (0,0,1,0), B-part = (1,0), W = (0,1)
  Term 4: A-part = (0,0,0,1), B-part = (0,1), W = (0,1)
  
  Now let me code this up.
"""

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    # Strassen 2x2*2x2 coefficients
    # A indices: (00, 01, 10, 11)
    # B indices: (00, 01, 10, 11)
    # C indices: (00, 01, 10, 11)
    strassen = [
        # (A_coeffs, B_coeffs, C_coeffs)
        ((1,0,0,1), (1,0,0,1), (1,0,0,1)),   # M1
        ((0,0,1,1), (1,0,0,0), (0,0,1,0)),   # M2
        ((1,0,0,0), (0,1,0,-1), (0,1,0,0)),  # M3
        ((0,0,0,1), (0,0,1,-1), (1,0,1,0)),  # M4
        ((1,1,0,0), (0,0,0,1), (-1,1,0,0)),  # M5
        ((0,0,1,-1), (1,1,0,0), (0,0,0,1)),  # M6
        ((0,1,0,-1), (0,0,1,1), (1,0,0,0)),  # M7
    ]
    
    # Direct 2x2*2x1 coefficients
    # A indices: (00, 01, 10, 11)
    # B indices: (0, 1)  [B[0], B[1]]
    # C indices: (0, 1)  [C[0], C[1]]
    direct_2x2_2x1 = [
        ((1,0,0,0), (1,0), (1,0)),
        ((0,1,0,0), (0,1), (1,0)),
        ((0,0,1,0), (1,0), (0,1)),
        ((0,0,0,1), (0,1), (0,1)),
    ]
    
    # Define sub-problems
    # Each sub-problem: (type, A_row_base, A_col_base, B_row_base, B_col_base, C_row_base, C_col_base)
    # type: 'strassen' or 'direct'
    # For strassen: A is 2x2, B is 2x2, C is 2x2
    # For direct: A is 2x2, B is 2x1, C is 2x1
    
    subproblems = [
        # (type, a_i_base, a_j_base, b_j_base, b_k_base, c_i_base, c_k_base)
        ('strassen', 0, 0, 0, 0, 0, 0),   # A_L * B_TL
        ('strassen', 0, 0, 0, 2, 0, 2),   # A_L * B_TR1
        ('direct',   0, 0, 0, 4, 0, 4),   # A_L * B_TR2
        ('strassen', 0, 2, 2, 0, 0, 0),   # A_R * B_BL
        ('strassen', 0, 2, 2, 2, 0, 2),   # A_R * B_BR1
        ('direct',   0, 2, 2, 4, 0, 4),   # A_R * B_BR2
    ]
    
    # Count total rank
    rank = 0
    for sp in subproblems:
        if sp[0] == 'strassen':
            rank += 7
        else:
            rank += 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for sp in subproblems:
        sp_type, ai, aj, bj, bk, ci, ck = sp
        if sp_type == 'strassen':
            # A entries: A[ai, aj], A[ai, aj+1], A[ai+1, aj], A[ai+1, aj+1]
            # U indices: ai*m + aj, ai*m + aj+1, (ai+1)*m + aj, (ai+1)*m + aj+1
            a_u_indices = [ai*m + aj, ai*m + (aj+1), (ai+1)*m + aj, (ai+1)*m + (aj+1)]
            # B entries: B[bj, bk], B[bj, bk+1], B[bj+1, bk], B[bj+1, bk+1]
            # V indices: bj*p + bk, bj*p + bk+1, (bj+1)*p + bk, (bj+1)*p + bk+1
            b_v_indices = [bj*p + bk, bj*p + (bk+1), (bj+1)*p + bk, (bj+1)*p + (bk+1)]
            # C entries: C[ci, ck], C[ci, ck+1], C[ci+1, ck], C[ci+1, ck+1]
            # W indices: ck*n + ci, ck*n + ci+1, (ck+1)*n + ci, (ck+1)*n + ci+1
            c_w_indices = [ck*n + ci, ck*n + (ci+1), (ck+1)*n + ci, (ck+1)*n + (ci+1)]
            
            for a_coeffs, b_coeffs, c_coeffs in strassen:
                for idx, val in enumerate(a_coeffs):
                    U[a_u_indices[idx]][col] += val
                for idx, val in enumerate(b_coeffs):
                    V[b_v_indices[idx]][col] += val
                for idx, val in enumerate(c_coeffs):
                    W[c_w_indices[idx]][col] += val
                col += 1
        else:  # direct
            # A entries: same as above
            a_u_indices = [ai*m + aj, ai*m + (aj+1), (ai+1)*m + aj, (ai+1)*m + (aj+1)]
            # B entries: B[bj, bk], B[bj+1, bk]
            b_v_indices = [bj*p + bk, (bj+1)*p + bk]
            # C entries: C[ci, ck], C[ci+1, ck]
            c_w_indices = [ck*n + ci, ck*n + (ci+1)]
            
            for a_coeffs, b_coeffs, c_coeffs in direct_2x2_2x1:
                for idx, val in enumerate(a_coeffs):
                    U[a_u_indices[idx]][col] += val
                for idx, val in enumerate(b_coeffs):
                    V[b_v_indices[idx]][col] += val
                for idx, val in enumerate(c_coeffs):
                    W[c_w_indices[idx]][col] += val
                col += 1
    
    return {"U": U, "V": V, "W": W}

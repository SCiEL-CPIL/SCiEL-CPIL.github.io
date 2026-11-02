"""Tensor decomposition for 2x2 times 2x12 matrix multiplication.

Uses the Kronecker product of Strassen's rank-7 decomposition for 2x2x2
matrix multiplication and the rank-6 trivial decomposition for 1x2 by 2x6
(1x6 vector scaling).  Total rank = 7 * 6 = 42, beating the schoolbook rank 48.
"""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 12
    rank = 42

    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0.0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]   # 24 x 42

    # Strassen's 7-term decomposition for 2x2 x 2x2:
    # Each term: (u_indices, v_indices, w_coefficients)
    # u_indices: rows of U (A factors) -> (i2, i1)
    # v_indices: rows of V (B factors) -> (j2, j1)
    # w_coefficients: list of (k2, k1, coeff) for output C factors
    #
    # A is 2x2: A[i2, i1]
    # B is 2x2: B[j2, j1]
    # C is 2x2: C[k2, k1] = sum A[i2,i1]*B[j2,j1] where i1==j2 and j1==k2
    #
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B10-B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B01-B10)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B10) -> C11 += M6
    # M7 = (A01-A11)(B01+B11) -> C00 -= M7

    strassen_terms = [
        # (u_idx_list, v_idx_list, [(k2, k1, coeff), ...])
        # M1 = (A00+A11)(B00+B11)
        ([(0,0),(1,1)], [(0,0),(1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)B00
        ([(1,0),(1,1)], [(0,0)], [(1,0,1),(1,1,-1)]),
        # M3 = A00(B10-B11)
        ([(0,0)], [(1,0),(1,1)], [(0,1,1),(1,1,1)]),
        # M4 = A11(B01-B10)
        ([(1,1)], [(0,1),(1,0)], [(0,0,1),(1,0,1)]),
        # M5 = (A00+A01)B11
        ([(0,0),(0,1)], [(1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A10-A00)(B00+B10)
        ([(1,0),(0,0)], [(0,0),(1,0)], [(1,1,1)]),
        # M7 = (A01-A11)(B01+B11)
        ([(0,1),(1,1)], [(0,1),(1,1)], [(0,0,-1)]),
    ]

    # The 1x2 by 2x6 factor: trivial rank-6 decomposition
    # A1 is 1x2: A1[0, i1]
    # B1 is 2x6: B1[j1, k1]
    # C1 is 1x6: C1[0, k1] = sum_{j1} A1[0, i1=j1] * B1[j1, k1]
    # Term t (t=0..5): U1[0, t]=1 if i1==t, V1[t, t]=1, W1[t, t]=1
    # So for each t: u1_idx = t, v1_idx = t, w1_idx = t, coeff = 1

    for s_idx, (u_indices, v_indices, w_coeffs) in enumerate(strassen_terms):
        for t in range(6):
            col = s_idx * 6 + t
            # U: A2 factor rows (i2, i1) and A1 factor row i1=t
            # U row index = i2 * 2 + i1  (since m=2, row = i*m + j, j is the column of A which is i1)
            # Wait: U is (n*m, rank) = (2*2, 42). Row index = i*m + j where i in 0..n-1, j in 0..m-1.
            # n=2 (rows of A), m=2 (cols of A). So U[i*2+j, col] corresponds to A[i,j].
            # In Kronecker: A = A2 ⊗ A1. A2 is 2x2, A1 is 1x2.
            # A[i2*1 + i1, j2*1 + j1] = A2[i2,j2] * A1[i1,j1]
            # Row of A: i2 + i1 (since n2=2, n1=1... wait let me re-derive)
            
            # A is n x m = 2 x 2. A = A2 ⊗ A1 where A2 is 2x2, A1 is 1x2.
            # (A2 ⊗ A1)[i2*n1 + i1, j2*m1 + j1] = A2[i2,j2] * A1[i1,j1]
            # n2=2, n1=1, m2=2, m1=2. Total rows: n2*n1=2, total cols: m2*m1=4.
            # But A is 2x2, not 2x4. So this decomposition doesn't directly work.
            
            # Let me reconsider. We need A (2x2), B (2x12), C (2x12) [output is 2x12? No...]
            # A is n x m = 2 x 2. B is m x p = 2 x 12. C = A*B is n x p = 2 x 12.
            # W is (p*n, rank) = (12*2, rank) = (24, rank). W[k*n+i] = W[k*2+i].
            
            # Kronecker: (n2 x m2)(m2 x p2) -> (n2 x p2) ⊗ (n1 x m1)(m1 x p1) -> (n1 x p1)
            # n = n2*n1 = 2, m = m2*m1 = 2, p = p2*p1 = 12
            # Option: n2=2, n1=1, m2=2, m1=1, p2=12, p1=1 -> but m2*m1=2, p2*p1=12. n2*n1=2. OK.
            # But m1=1, p1=1 means the second factor is 1x1 x 1x1, rank 1. Total rank = 7*1 = 7. That can't be right.
            
            # Option: n2=2, n1=1, m2=1, m1=2, p2=6, p1=2. Then m=1*2=2, p=6*2=12. n=2*1=2.
            # First factor: (2x1)(1x6) -> (2x6). Rank of 2x1 x 1x6 = 2*1*6=12 (or can be less?)
            # Actually (n x 1)(1 x p) multiplication: C[i,k] = A[i,0]*B[0,k]. Rank = n*p = 12.
            # Second factor: (1x2)(2x2) -> (1x2). Rank = 1*2*2 = 4 (or less).
            # Total: 12*4 = 48. Same as schoolbook.
            
            # Option: n2=2, n1=1, m2=2, m1=1, p2=6, p1=2. m=2*1=2, p=6*2=12, n=2*1=2.
            # First: (2x2)(2x6) -> (2x6). Rank of 2x2x6 = ? At most 12, but could be less.
            # Second: (1x1)(1x2) -> (1x2). Rank = 2.
            # Total: rank(2x2x6) * 2. If rank(2x2x6)=12, total=24. But rank(2x2x6) is at least 6 (each output needs at least one term since inputs are independent). Actually rank(2x2x6) = 12 for generic, but for exact... hmm.
            
            # Let me just do the direct approach: use Strassen for the 2x2x2 part and handle the 12 columns separately.
            
            pass

    # Direct approach: 
    # C[k,i] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0..11
    # We want to decompose this tensor of shape (2, 2, 24) [A: 4 elements, B: 24 elements, C: 24 elements]
    # into rank-r terms: C = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] (appropriately indexed)
    
    # For each column k of B, we have a 2x2 matrix B[:,k] and compute A * B[:,k] which is a 2x1 vector.
    # The 2x2 x 1x2 (which is 2x1) multiplication has rank 4 (schoolbook).
    # But if we group columns, we can use Strassen.
    
    # Actually, let's think of it as: C = A * B where A is 2x2 and B is 2x12.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is just a sum of 4 terms per (i,k) pair, total 24*2 = 48 scalar multiplications (schoolbook).
    
    # To reduce rank, we can use the fact that for fixed i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # This is a linear combination of two rows of B. No rank reduction possible here since B[0,:] and B[1,:] are independent.
    
    # Alternatively, view B as 12 independent 2x1 vectors. For each, compute A * B_col (2x1).
    # The tensor for (2x2) * (2x1) -> (2x1) has rank 4 (each output element needs one term, and they're independent).
    # 12 such independent problems -> rank 48.
    
    # Can we do better? If we use Strassen on pairs of columns... but Strassen works for 2x2 * 2x2 -> 2x2.
    # If we group B into 6 pairs of columns: B_pair is 2x2. Then A * B_pair is 2x2.
    # Use Strassen (rank 7) for each pair. 6 pairs * rank 7 = rank 42.
    # This works! Each pair of columns of B is a 2x2 matrix, and we compute A * B_pair using Strassen.
    
    # So: 6 groups of 2 columns. For group g (columns 2g, 2g+1):
    # B_g is 2x2: B_g[j, h] = B[j, 2g+h] for j=0,1 and h=0,1
    # C_g = A * B_g is 2x2: C_g[i, h] = C[i, 2g+h]
    # Use Strassen's 7-term decomposition for each group.
    # Total rank: 6 * 7 = 42.

    # Reset matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen terms for 2x2 x 2x2 -> 2x2:
    # A is 2x2: A[i,j]
    # B is 2x2: B[j,k]
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # M1 = (A00+A11)(B00+B11): C00 += M1, C11 += M1
    # M2 = (A10+A11)B00:       C10 += M2, C11 -= M2
    # M3 = A00(B10-B11):       C01 += M3, C11 += M3
    # M4 = A11(B01-B10):       C00 += M4, C10 += M4
    # M5 = (A00+A01)B11:       C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B10): C11 += M6
    # M7 = (A01-A11)(B01+B11): C00 -= M7

    # U rows: A[i,j] -> row index i*m + j = i*2 + j (n=2, m=2)
    # V rows: B[j,k] -> row index j*p + k = j*12 + k (m=2, p=12)
    # W rows: C[k,i] -> row index k*n + i = k*2 + i (p=12, n=2)
    #   Note: W[k*n+i] where k is column of C (0..11), i is row of C (0,1)

    for g in range(6):
        base_col = g * 7
        # For group g, columns 2g and 2g+1
        # B[j, 2g] and B[j, 2g+1] -> V rows: j*12 + 2g and j*12 + 2g+1
        # C[i, 2g] and C[i, 2g+1] -> W rows: (2g)*2 + i and (2g+1)*2 + i

        # M1 = (A00+A11)(B00+B11)
        col = base_col + 0
        # U: A00 (row 0*2+0=0) and A11 (row 1*2+1=3)
        U[0][col] = 1.0
        U[3][col] = 1.0
        # V: B00 (row 0*12+2g) and B11 (row 1*12+2g+1)
        V[0 * 12 + 2 * g][col] = 1.0
        V[1 * 12 + 2 * g + 1][col] = 1.0
        # W: C00 (row 2g*2+0) and C11 (row (2g+1)*2+1)
        W[2 * g * 2 + 0][col] = 1.0
        W[(2 * g + 1) * 2 + 1][col] = 1.0

        # M2 = (A10+A11)B00
        col = base_col + 1
        # U: A10 (row 1*2+0=2) and A11 (row 3)
        U[2][col] = 1.0
        U[3][col] = 1.0
        # V: B00 (row 0*12+2g)
        V[0 * 12 + 2 * g][col] = 1.0
        # W: C10 (row 2g*2+1) with +1, C11 (row (2g+1)*2+1) with -1
        W[2 * g * 2 + 1][col] = 1.0
        W[(2 * g + 1) * 2 + 1][col] = -1.0

        # M3 = A00(B10-B11)
        col = base_col + 2
        # U: A00 (row 0)
        U[0][col] = 1.0
        # V: B10 (row 1*12+2g) and B11 (row 1*12+2g+1)
        V[1 * 12 + 2 * g][col] = 1.0
        V[1 * 12 + 2 * g + 1][col] = -1.0
        # W: C01 (row (2g+1)*2+0) with +1, C11 (row (2g+1)*2+1) with +1
        W[(2 * g + 1) * 2 + 0][col] = 1.0
        W[(2 * g + 1) * 2 + 1][col] = 1.0

        # M4 = A11(B01-B10)
        col = base_col + 3
        # U: A11 (row 3)
        U[3][col] = 1.0
        # V: B01 (row 0*12+2g+1) and B10 (row 1*12+2g)
        V[0 * 12 + 2 * g + 1][col] = 1.0
        V[1 * 12 + 2 * g][col] = -1.0
        # W: C00 (row 2g*2+0) with +1, C10 (row 2g*2+1) with +1
        W[2 * g * 2 + 0][col] = 1.0
        W[2 * g * 2 + 1][col] = 1.0

        # M5 = (A00+A01)B11
        col = base_col + 4
        # U: A00 (row 0) and A01 (row 0*2+1=1)
        U[0][col] = 1.0
        U[1][col] = 1.0
        # V: B11 (row 1*12+2g+1)
        V[1 * 12 + 2 * g + 1][col] = 1.0
        # W: C00 (row 2g*2+0) with -1, C01 (row (2g+1)*2+0) with +1
        W[2 * g * 2 + 0][col] = -1.0
        W[(2 * g + 1) * 2 + 0][col] = 1.0

        # M6 = (A10-A00)(B00+B10)
        col = base_col + 5
        # U: A10 (row 2) and A00 (row 0)
        U[2][col] = 1.0
        U[0][col] = -1.0
        # V: B00 (row 0*12+2g) and B10 (row 1*12+2g)
        V[0 * 12 + 2 * g][col] = 1.0
        V[1 * 12 + 2 * g][col] = 1.0
        # W: C11 (row (2g+1)*2+1) with +1
        W[(2 * g + 1) * 2 + 1][col] = 1.0

        # M7 = (A01-A11)(B01+B11)
        col = base_col + 6
        # U: A01 (row 1) and A11 (row 3)
        U[1][col] = 1.0
        U[3][col] = -1.0
        # V: B01 (row 0*12+2g+1) and B11 (row 1*12+2g+1)
        V[0 * 12 + 2 * g + 1][col] = 1.0
        V[1 * 12 + 2 * g + 1][col] = 1.0
        # W: C00 (row 2g*2+0) with -1
        W[2 * g * 2 + 0][col] = -1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

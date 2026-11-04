import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = 4, 5, 5
    
    # Total rank: 49 (4x4x4 Strassen) + 16 (4x1x4 schoolbook) + 16 (4x4x1 schoolbook) + 4 (4x1x1 schoolbook) = 85
    rank = 85
    U = [[0] * rank for _ in range(20)]  # 4*5 = 20 rows
    V = [[0] * rank for _ in range(25)]  # 5*5 = 25 rows
    W = [[0] * rank for _ in range(20)]  # 5*4 = 20 rows
    
    col = 0
    
    # Helper: add a rank-1 term for the 4x4x4 subproblem (i,j,k in 0..3)
    def add_term_4x4(i, j, k, u_val, v_val, w_val):
        nonlocal col
        U[i*5+j][col] = u_val
        V[j*5+k][col] = v_val
        W[k*4+i][col] = w_val
        col += 1
    
    # 4x4x4 Strassen decomposition (rank 49)
    # Recursive: 7 block products, each 2x2x2 (rank 7)
    
    # For each of the 7 Strassen products at the block level,
    # we have a 2x2x2 subproblem. For each such subproblem,
    # the 7 scalar Strassen terms give 7 rank-1 terms.
    # Total: 7 * 7 = 49 terms.
    
    # Block-level Strassen formulas (0-based block indices):
    # M0: P=(A00+A11), Q=(B00+B11), R: C00+=M0, C11+=M0
    # M1: P=(A10+A11), Q=B00, R: C10+=M1, C11-=M1
    # M2: P=A00, Q=(B01-B11), R: C01+=M2, C11+=M2
    # M3: P=A11, Q=(B10-B00), R: C00+=M3, C10+=M3
    # M4: P=(A00+A01), Q=B11, R: C00-=M4, C01+=M4
    # M5: P=(A10-A00), Q=(B00+B01), R: C11+=M5
    # M6: P=(A01-A11), Q=(B10+B11), R: C00+=M6, C11-=M6
    
    # For each block product, the "P" and "Q" are 2x2 matrices (blocks of A and B).
    # The scalar entries are linear combinations of A and B entries.
    
    # For the 2x2x2 Strassen within each block product, we have 7 terms.
    # Let me define the 7 terms for a generic 2x2x2 with P[a,b], Q[c,d], R[e,f]:
    
    # The 7 Strassen terms for 2x2x2:
    # S0 = (P00+P11)*(Q00+Q11), R: R00+=1, R11+=1
    # S1 = (P10+P11)*Q00, R: R10+=1, R11-=1
    # S2 = P00*(Q01-Q11), R: R01+=1, R11+=1
    # S3 = P11*(Q10-Q00), R: R00+=1, R10+=1
    # S4 = (P00+P01)*Q11, R: R00-=1, R01+=1
    # S5 = (P10-P00)*(Q00+Q01), R: R11+=1
    # S6 = (P01-P11)*(Q10+Q11), R: R00+=1, R11-=1
    
    # For each block-level Strassen product, the P and Q are linear combinations
    # of 2x2 blocks of A and B. The actual scalar entries of P and Q are
    # linear combinations of individual A[i,j] and B[j,k] entries.
    
    # Let me enumerate all 49 terms by iterating over the 7 block products
    # and the 7 scalar Strassen terms within each.
    
    # Block product 0: P = A00+A11, Q = B00+B11
    # P[ai,aj] = A[ai,aj] + A[2+ai, 2+aj] for ai,aj in {0,1}
    # Q[bi,bj] = B[bi,bj] + B[2+bi, 2+bj] for bi,bj in {0,1}
    # R: C00 += R, C11 += R where R[e,f] is the 2x2 result
    
    # The 7 scalar terms for this block product:
    # S0: P00+P11 = (A00+A22)+(A01+A23), Q00+Q11 = (B00+B22)+(B01+B23)
    #     Wait, P00 = A[0,0]+A[2,2], P11 = A[1,1]+A[3,3]
    #     So P00+P11 = A[0,0]+A[2,2]+A[1,1]+A[3,3]
    #     Q00 = B[0,0]+B[2,2], Q11 = B[1,1]+B[3,3]
    #     Q00+Q11 = B[0,0]+B[2,2]+B[1,1]+B[3,3]
    #     This is one rank-1 term: U has 1 at positions (0,0),(2,2),(1,1),(3,3)
    #     V has 1 at positions (0,0),(2,2),(1,1),(3,3)
    #     W: for C00 block, R00=1,R11=1 means C[0,0]+=1, C[1,1]+=1
    #         for C11 block, same: C[2,2]+=1, C[3,3]+=1
    #     So W has 1 at (0,0),(1,1),(2,2),(3,3) in the 4x4 sense,
    #     which is W[k*4+i] for k,i in {0,1} (C00) and k,i in {2,3} (C11)
    
    # This is getting extremely tedious to do by hand. Let me just write the code
    # that generates all 49 terms programmatically.
    
    # I'll create a helper that, given the P and Q index mappings and R index mapping,
    # adds the 7 Strassen terms for a 2x2x2 subproblem.
    
    def add_strassen_2x2(P_map, Q_map, R_map):
        """
        P_map: dict mapping (ai,aj) -> list of (i,j,coeff) for A entries
        Q_map: dict mapping (bi,bj) -> list of (j,k,coeff) for B entries  
        R_map: dict mapping (ci,cj) -> list of (i,k,coeff) for C entries
        Adds 7 rank-1 terms to U, V, W and returns new col.
        """
        nonlocal col
        # 7 Strassen terms for 2x2x2
        terms = [
            # (P_coeffs: dict (ai,aj)->coeff, Q_coeffs: dict (bi,bj)->coeff, R_coeffs: dict (ci,cj)->coeff)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
        ]
        
        for P_coeffs, Q_coeffs, R_coeffs in terms:
            # Build U, V, W for this term
            for (ai, aj), p_coeff in P_coeffs.items():
                for (i, j, a_coeff) in P_map[(ai, aj)]:
                    U[i*5+j][col] += p_coeff * a_coeff
            for (bi, bj), q_coeff in Q_coeffs.items():
                for (j, k, b_coeff) in Q_map[(bi, bj)]:
                    V[j*5+k][col] += q_coeff * b_coeff
            for (ci, cj), r_coeff in R_coeffs.items():
                for (i, k, c_coeff) in R_map[(ci, cj)]:
                    W[k*4+i][col] += r_coeff * c_coeff
            col += 1
    
    # Define the 7 block-level Strassen products
    # For each, define P_map, Q_map, R_map
    
    # Block 0: A00+A11 means A[i,j] for i,j in 0..1 plus A[i+2,j+2] for i,j in 0..1
    # P[ai,aj] = A[ai,aj] + A[2+ai, 2+aj]
    P_map_00 = {(ai, aj): [(ai, aj, 1), (2+ai, 2+aj, 1)] for ai in range(2) for aj in range(2)}
    # Q[bi,bj] = B[bi,bj] + B[2+bi, 2+bj]
    Q_map_00 = {(bi, bj): [(bi, bj, 1), (2+bi, 2+bj, 1)] for bi in range(2) for bj in range(2)}
    # R: C00 += R, C11 += R
    R_map_00 = {(ci, cj): [(ci, cj, 1), (2+ci, 2+cj, 1)] for ci in range(2) for cj in range(2)}
    add_strassen_2x2(P_map_00, Q_map_00, R_map_00)
    
    # Block 1: P = A10+A11, Q = B00
    # P[ai,aj] = A[1+ai, aj] + A[1+ai, 1+aj]  (A10 is rows 2-3, cols 0-1; A11 is rows 2-3, cols 2-3)
    P_map_10 = {(ai, aj): [(1+ai, aj, 1), (1+ai, 1+aj, 1)] for ai in range(2) for aj in range(2)}
    Q_map_00_1 = {(bi, bj): [(bi, bj, 1)] for bi in range(2) for bj in range(2)}
    # R: C10 += R, C11 -= R
    R_map_10 = {(ci, cj): [(1+ci, cj, 1), (1+ci, 1+cj, -1)] for ci in range(2) for cj in range(2)}
    add_strassen_2x2(P_map_10, Q_map_00_1, R_map_10)
    
    # Block 2: P = A00, Q = B01-B11
    P_map_00_1 = {(ai, aj): [(ai, aj, 1)] for ai in range(2) for aj in range(2)}
    Q_map_01_11 = {(bi, bj): [(bi, 1+bj, 1), (1+bi, 1+bj, -1)] for bi in range(2) for bj in range(2)}
    R_map_01 = {(ci, cj): [(ci, 1+cj, 1), (1+ci, 1+cj, 1)] for ci in range(2) for cj in range(2)}
    add_strassen_2x2(P_map_00_1, Q_map_01_11, R_map_01)
    
    # Block 3: P = A11, Q = B10-B00
    P_map_11 = {(ai, aj): [(1+ai, 1+aj, 1)] for ai in range(2) for aj in range(2)}
    Q_map_10_00 = {(bi, bj): [(1+bi, bj, 1), (bi, bj, -1)] for bi in range(2) for bj in range(2)}
    R_map_03 = {(ci, cj): [(ci, cj, 1), (1+ci, cj, 1)] for ci in range(2) for cj in range(2)}
    add_strassen_2x2(P_map_11, Q_map_10_00, R_map_03)
    
    # Block 4: P = A00+A01, Q = B11
    P_map_00_01 = {(ai, aj): [(ai, aj, 1), (ai, 1+aj, 1)] for ai in range(2) for aj in range(2)}
    Q_map_11 = {(bi, bj): [(1+bi, 1+bj, 1)] for bi in range(2) for bj in range(2)}
    R_map_04 = {(ci, cj): [(ci, cj, -1), (ci, 1+cj, 1)] for ci in range(2) for cj in range(2)}
    add_strassen_2x2(P_map_00_01, Q_map_11, R_map_04)
    
    # Block 5: P = A10-A00, Q = B00+B01
    P_map_10_00 = {(ai, aj): [(1+ai, aj, 1), (ai, aj, -1)] for ai in range(2) for aj in range(2)}
    Q_map_00_01 = {(bi, bj): [(bi, bj, 1), (bi, 1+bj, 1)] for bi in range(2) for bj in range(2)}
    R_map_05 = {(ci, cj): [(1+ci, 1+cj, 1)] for ci in range(2) for cj in range(2)}
    add_strassen_2x2(P_map_10_00, Q_map_00_01, R_map_05)
    
    # Block 6: P = A01-A11, Q = B10+B11
    P_map_01_11 = {(ai, aj): [(ai, 1+aj, 1), (1+ai, 1+aj, -1)] for ai in range(2) for aj in range(2)}
    Q_map_10_11 = {(bi, bj): [(1+bi, bj, 1), (1+bi, 1+bj, 1)] for bi in range(2) for bj in range(2)}
    R_map_06 = {(ci, cj): [(ci, cj, 1), (1+ci, 1+cj, -1)] for ci in range(2) for cj in range(2)}
    add_strassen_2x2(P_map_01_11, Q_map_10_11, R_map_06)
    
    # Now col should be 49
    assert col == 49, f"Expected col=49, got {col}"
    
    # Part 2: 4x1x4 schoolbook (A[:,4:5] @ B[4:5,0:4]), rank 16
    # i in 0..3, j=4, k in 0..3
    for i in range(4):
        for k in range(4):
            U[i*5+4][col] = 1
            V[4*5+k][col] = 1
            W[k*4+i][col] = 1
            col += 1
    
    # Part 3: 4x4x1 schoolbook (A[:,0:4] @ B[0:4,4:5]), rank 16
    # i in 0..3, j in 0..3, k=4
    for i in range(4):
        for j in range(4):
            U[i*5+j][col] = 1
            V[j*5+4][col] = 1
            W[4*4+i][col] = 1
            col += 1
    
    # Part 4: 4x1x1 schoolbook (A[:,4:5] @ B[4:5,4:5]), rank 4
    # i in 0..3, j=4, k=4
    for i in range(4):
        U[i*5+4][col] = 1
        V[4*5+4][col] = 1
        W[4*4+i][col] = 1
        col += 1
    
    assert col == 85, f"Expected col=85, got {col}"
    
    return {"U": U, "V": V, "W": W}

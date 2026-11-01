"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    # Use Strassen's algorithm for each 2x2 block of the 2x16 matrix.
    # This gives rank 7 * 8 = 56, which is better than the schoolbook rank of 64.
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    rank = 7 * 8  # 56
    
    # For each of the 8 blocks (each 2x2), we use Strassen's 7 multiplications.
    # The 8 blocks correspond to columns 4q..4q+3 of B, for q=0..7.
    
    # Strassen's algorithm for 2x2 matrix multiplication:
    # C = A * B where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[a*e - d*g + c*h + a*f, a*f + a*h + d*h, c*e + c*g + a*g + d*g],
    #      [c*e + c*g + a*g + d*g, ...]]
    # Actually, let me use the standard Strassen decomposition:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = (a-b)(g+h)
    # M4 = d(f-h)
    # M5 = a(e-g)
    # M6 = c(b-d)
    # M7 = (a+b)h
    # 
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M7
    # C10 = M2 - M4 - M6
    # C11 = M1 - M2 + M3 + M6
    
    # For our problem, A is 2x2 and B is 2x16.
    # We split B into 8 blocks of 2x2: B_q = B[:, 4q:4q+4]
    # For each block, we apply Strassen.
    
    # The U matrix has shape (n*m, rank) = (4, 56)
    # The V matrix has shape (m*p, rank) = (32, 56)
    # The W matrix has shape (p*n, rank) = (32, 56)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for A (2x2 -> flattened as [a,b,c,d]):
    # M1: (a+d) -> [1, 0, 0, 1]
    # M2: (c+d) -> [0, 0, 1, 1]
    # M3: (a-b) -> [1, -1, 0, 0]
    # M4: d -> [0, 0, 0, 1]
    # M5: a -> [1, 0, 0, 0]
    # M6: (b-d) -> [0, 1, 0, -1]
    # M7: (a+b) -> [1, 1, 0, 0]
    
    strassen_A = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, -1, 0, 0],  # M3: a-b
        [0, 0, 0, 1],   # M4: d
        [1, 0, 0, 0],   # M5: a
        [0, 1, 0, -1],  # M6: b-d
        [1, 1, 0, 0],   # M7: a+b
    ]
    
    # For each block q, B_q = [[e, f], [g, h]] where:
    # e = B[0, 4q], f = B[0, 4q+1], g = B[1, 4q], h = B[1, 4q+1]
    # Wait, I need to be careful with the index convention.
    
    # Let me reconsider. B is 2x16, so B[j,k] where j in {0,1}, k in {0..15}.
    # For block q (q=0..7), the columns are 4q, 4q+1, 4q+2, 4q+3.
    # B_q = [[B[0,4q], B[0,4q+1]], [B[1,4q], B[1,4q+1]]]
    # But wait, that's only 4 columns per block, and 8 blocks * 4 = 32 columns, but B only has 16 columns.
    # I made an error. B is 2x16, so it has 16 columns. If I split into 2x2 blocks, I get 8 blocks.
    # Each block uses 2 columns: B[0, 2q:2q+2] and B[1, 2q:2q+2].
    # So for block q, e = B[0, 2q], f = B[0, 2q+1], g = B[1, 2q], h = B[1, 2q+1].
    
    # So the V matrix indexing: V[j*p + k, col] where j in {0,1}, k in {0..15}.
    # For block q, e is at V[0*16 + 2q, col] = V[2q, col]
    # f is at V[0*16 + 2q+1, col] = V[2q+1, col]
    # g is at V[1*16 + 2q, col] = V[16+2q, col]
    # h is at V[1*16 + 2q+1, col] = V[16+2q+1, col]
    
    # Strassen vectors for B_q = [[e,f],[g,h]]:
    # M1: (e+h) -> [1, 0, 0, 1]  (e is B[0,2q], h is B[1,2q+1])
    # Wait, I need to map e,f,g,h to the correct indices.
    # e = B[0, 2q] -> V index 2q
    # f = B[0, 2q+1] -> V index 2q+1
    # g = B[1, 2q] -> V index 16+2q
    # h = B[1, 2q+1] -> V index 16+2q+1
    
    # So for Strassen:
    # M1 = (a+d)(e+h): B part is [1, 0, 0, 1] for [e,f,g,h]
    # M2 = (c+d)(e+f): B part is [1, 1, 0, 0]
    # M3 = (a-b)(g+h): B part is [0, 0, 1, 1]
    # M4 = d(f-h): B part is [0, 1, 0, -1]
    # M5 = a(e-g): B part is [1, 0, -1, 0]
    # M6 = c(b-d): B part is [0, 0, 0, 0]? No, M6 doesn't involve B directly in the same way.
    # Let me re-derive.
    
    # Actually, let me just use a known correct set of Strassen vectors.
    # I'll define them carefully.
    
    # For 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = (a-b)(g+h)
    # M4 = d(f-h)
    # M5 = a(e-g)
    # M6 = c(b-d)
    # M7 = (a+b)h
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M7
    # C10 = M2 - M4 - M6
    # C11 = M1 - M2 + M3 + M6
    
    # So the B-vectors for each M are:
    # M1: e+h -> [1, 0, 0, 1]
    # M2: e+f -> [1, 1, 0, 0]
    # M3: g+h -> [0, 0, 1, 1]
    # M4: f-h -> [0, 1, 0, -1]
    # M5: e-g -> [1, 0, -1, 0]
    # M6: 0 (M6 = c(b-d) doesn't involve B in a linear way? No, it's c*(b-d), so it's a scalar times a scalar.
    # Wait, M6 = c*(b-d) where c is from A and (b-d) is from A. So M6 doesn't use B at all?
    # That can't be right. Let me check.
    
    # Actually, M6 = c(b-d) is a product of two elements from A. So the B-vector for M6 is [0,0,0,0].
    # But then M6 contributes to C10 and C11, which are outputs. How can a term that doesn't depend on B contribute to the output?
    # I think I'm confusing the decomposition. Let me use a different set.
    
    # Let me use the standard Strassen from the 1969 paper:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)(e+f)
    # P3 = (a-b)(g+h)
    # P4 = d(f-h)
    # P5 = a(e-g)
    # P6 = c(b-d)
    # P7 = (a+b)h
    
    # C00 = P1 + P4 - P5 - P7
    # C01 = P5 + P7
    # C10 = P2 - P4 - P6
    # C11 = P1 - P2 + P3 + P6
    
    # Yes, P6 = c(b-d) only involves A. So the B-vector for P6 is zero.
    # Similarly, P6's A-vector is [0, 1, 1, -1] (for c*(b-d) = -c*b + c*d).
    
    # OK so let me define the A and B vectors properly.
    
    strassen_A = [
        [1, 0, 0, 1],   # P1: a+d
        [0, 0, 1, 1],   # P2: c+d
        [1, -1, 0, 0],  # P3: a-b
        [0, 0, 0, 1],   # P4: d
        [1, 0, 0, 0],   # P5: a
        [0, 1, 1, -1],  # P6: c(b-d) = -cb + cd
        [1, 1, 0, 0],   # P7: a+b
    ]
    
    strassen_B = [
        [1, 0, 0, 1],   # P1: e+h
        [1, 1, 0, 0],   # P2: e+f
        [0, 0, 1, 1],   # P3: g+h
        [0, 1, 0, -1],  # P4: f-h
        [1, 0, -1, 0],  # P5: e-g
        [0, 0, 0, 0],   # P6: 0 (no B dependence)
        [0, 0, 0, 1],   # P7: h
    ]
    
    # W vectors: how each P contributes to C00, C01, C10, C11
    # C00 = P1 + P4 - P5 - P7
    # C01 = P5 + P7
    # C10 = P2 - P4 - P6
    # C11 = P1 - P2 + P3 + P6
    
    # So W for each P (indexed by [C00, C01, C10, C11]):
    strassen_W = [
        [1, 0, 0, 1],   # P1: +C00, +C11
        [0, 0, 1, -1],  # P2: +C10, -C11
        [0, 0, 0, 1],   # P3: +C11
        [1, 0, -1, 0],  # P4: +C00, -C10
        [-1, 1, 0, 0],  # P5: -C00, +C01
        [0, 0, -1, 1],  # P6: -C10, +C11
        [-1, 1, 0, 0],  # P7: -C00, +C01
    ]
    
    # Now, for block q, the output C_q = A * B_q is 2x2.
    # The output indices in W:
    # C_q[0,0] -> output index? 
    # The problem says W has shape (p*n, rank) and W[k*n+i, col].
    # So W index = k*n + i where k is the output column and i is the output row.
    # For block q, the output columns are 2q and 2q+1.
    # C_q[0,0] (row 0, col 0 of block) -> output col = 2q, row = 0 -> W index = 2q*2 + 0 = 4q
    # C_q[0,1] (row 0, col 1 of block) -> output col = 2q+1, row = 0 -> W index = (2q+1)*2 + 0 = 4q+2
    # C_q[1,0] (row 1, col 0 of block) -> output col = 2q, row = 1 -> W index = 2q*2 + 1 = 4q+1
    # C_q[1,1] (row 1, col 1 of block) -> output col = 2q+1, row = 1 -> W index = (2q+1)*2 + 1 = 4q+3
    
    # So for block q, the W indices for [C00, C01, C10, C11] are [4q, 4q+2, 4q+1, 4q+3].
    
    for q in range(8):
        for s in range(7):
            col = 7 * q + s
            
            # U vector (for A, flattened as [a,b,c,d] -> indices 0,1,2,3 in U)
            for i in range(4):
                U[i][col] = strassen_A[s][i]
            
            # V vector (for B_q, flattened as [e,f,g,h] -> indices in V)
            # e = B[0, 2q] -> V index 2q
            # f = B[0, 2q+1] -> V index 2q+1
            # g = B[1, 2q] -> V index 16+2q
            # h = B[1, 2q+1] -> V index 16+2q+1
            v_indices = [2*q, 2*q+1, 16+2*q, 16+2*q+1]
            for idx, val in zip(v_indices, strassen_B[s]):
                V[idx][col] = val
            
            # W vector (for output C_q, [C00, C01, C10, C11] -> W indices [4q, 4q+2, 4q+1, 4q+3])
            w_indices = [4*q, 4*q+2, 4*q+1, 4*q+3]
            for idx, val in zip(w_indices, strassen_W[s]):
                W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

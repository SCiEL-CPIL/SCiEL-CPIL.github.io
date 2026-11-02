import time

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # Rank 53 decomposition
    rank = 53
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for A (2x2)
    # A = [[a, b], [c, d]]
    # Indices: 0: a, 1: b, 2: c, 3: d
    # L1: a+d
    # L2: c+d
    # L3: a
    # L4: d
    # L5: a+b
    # L6: c-a
    # L7: b-d
    
    # Strassen coefficients for B block (2x2)
    # B = [[e, f], [g, h]]
    # R1: e+h
    # R2: e
    # R3: f-h
    # R4: g-e
    # R5: h
    # R6: e+f
    # R7: g+h
    
    # Output contributions for C (2x2)
    # C00: M1 + M4 - M5 + M7
    # C01: M3 + M5
    # C10: M2 + M4
    # C11: M1 - M2 + M3 + M6
    
    # Map M_k to U coefficients
    U_coeffs = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    # Map M_k to V coefficients for a generic 2x2 block at columns j0, j1
    # B indices: j0, j1 for row 0; j0, j1 for row 1? No.
    # B is 2x15. Block i uses columns 2i and 2i+1.
    # B[0, 2i], B[0, 2i+1], B[1, 2i], B[1, 2i+1]
    # Indices in V: row * p + col
    # B[0, c] -> V[0*p + c]
    # B[1, c] -> V[1*p + c]
    
    def get_V_coeffs(block_idx):
        c0 = 2 * block_idx
        c1 = 2 * block_idx + 1
        # R1: e+h -> B[0,c0] + B[1,c1]
        # R2: e    -> B[0,c0]
        # R3: f-h  -> B[0,c1] - B[1,c1]
        # R4: g-e  -> B[1,c0] - B[0,c0]
        # R5: h    -> B[1,c1]
        # R6: e+f  -> B[0,c0] + B[0,c1]
        # R7: g+h  -> B[1,c0] + B[1,c1]
        
        coeffs = []
        # We need to return a list of 7 dicts or lists of (index, coeff)
        # Let's just return 7 lists of length m*p (30) or sparse representation
        
        # Sparse is better for filling V
        sparse = []
        
        # R1
        sparse.append([(0*p + c0, 1), (1*p + c1, 1)])
        # R2
        sparse.append([(0*p + c0, 1)])
        # R3
        sparse.append([(0*p + c1, 1), (1*p + c1, -1)])
        # R4
        sparse.append([(1*p + c0, 1), (0*p + c0, -1)])
        # R5
        sparse.append([(1*p + c1, 1)])
        # R6
        sparse.append([(0*p + c0, 1), (0*p + c1, 1)])
        # R7
        sparse.append([(1*p + c0, 1), (1*p + c1, 1)])
        
        return sparse

    # W coefficients for each M_k in a block
    # W index: k*n + i
    # C[0, c0] -> W[0*n + 0] = W[0]
    # C[0, c1] -> W[0*n + 1] = W[1]
    # C[1, c0] -> W[1*n + 0] = W[2]
    # C[1, c1] -> W[1*n + 1] = W[3]
    
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(+1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C11(+1)
    # M7: C00(+1)
    
    def get_W_coeffs(block_idx):
        c0 = 2 * block_idx
        c1 = 2 * block_idx + 1
        sparse = []
        
        # M1
        sparse.append([(c0*n + 0, 1), (c1*n + 1, 1)])
        # M2
        sparse.append([(c0*n + 1, 1), (c1*n + 1, -1)])
        # M3
        sparse.append([(c1*n + 0, 1), (c1*n + 1, 1)])
        # M4
        sparse.append([(c0*n + 0, 1), (c0*n + 1, 1)])
        # M5
        sparse.append([(c0*n + 0, -1), (c1*n + 0, 1)])
        # M6
        sparse.append([(c1*n + 1, 1)])
        # M7
        sparse.append([(c0*n + 0, 1)])
        
        return sparse

    # Fill 7 blocks
    for b in range(7):
        V_sparse = get_V_coeffs(b)
        W_sparse = get_W_coeffs(b)
        
        for k in range(7):
            # Fill U
            for idx, val in enumerate(U_coeffs[k]):
                if val != 0:
                    U[idx][col] = val
            
            # Fill V
            for idx, val in V_sparse[k]:
                V[idx][col] = val
            
            # Fill W
            for idx, val in W_sparse[k]:
                W[idx][col] = val
            
            col += 1

    # Fill 1 block (2x1)
    # B[0, 14], B[1, 14]
    # C[0, 14] = A[0,0]B[0,14] + A[0,1]B[1,14]
    # C[1, 14] = A[1,0]B[0,14] + A[1,1]B[1,14]
    
    # Term 1: A[0,0] B[0,14] -> C[0,14]
    U[0][col] = 1
    V[0*p + 14][col] = 1
    W[14*n + 0][col] = 1
    col += 1
    
    # Term 2: A[0,1] B[1,14] -> C[0,14]
    U[1][col] = 1
    V[1*p + 14][col] = 1
    W[14*n + 0][col] = 1
    col += 1
    
    # Term 3: A[1,0] B[0,14] -> C[1,14]
    U[2][col] = 1
    V[0*p + 14][col] = 1
    W[14*n + 1][col] = 1
    col += 1
    
    # Term 4: A[1,1] B[1,14] -> C[1,14]
    U[3][col] = 1
    V[1*p + 14][col] = 1
    W[14*n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

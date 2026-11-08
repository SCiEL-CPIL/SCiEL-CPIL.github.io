import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 72: 28 for B[0:4], 28 for B[4:8], 16 for B[8:9]
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    # u_indices: list of (i, j) and coeff
    # v_indices: list of (j, k) and coeff
    # w_indices: list of (k, i) and coeff
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for (i, j), c in u_coeffs.items():
            U[i * m + j][col] += c
        for (j, k), c in v_coeffs.items():
            V[j * p + k][col] += c
        for (k, i), c in w_coeffs.items():
            W[k * n + i][col] += c
        col += 1
        
    # Strassen blocks for A (4x4)
    # A11, A12, A21, A22 are 2x2 blocks
    # Indices in A:
    # A11: (0,0)-(1,1)
    # A12: (0,2)-(1,3)
    # A21: (2,0)-(3,1)
    # A22: (2,2)-(3,3)
    
    # We process B in chunks of 4 columns
    # Chunk 1: cols 0-3
    # Chunk 2: cols 4-7
    # Chunk 3: col 8
    
    for chunk_start in [0, 4]:
        # B block is 4x4
        # B11: rows 0-1, cols chunk_start to chunk_start+1
        # B12: rows 0-1, cols chunk_start+2 to chunk_start+3
        # B21: rows 2-3, cols chunk_start to chunk_start+1
        # B22: rows 2-3, cols chunk_start+2 to chunk_start+3
        
        # Define Strassen terms X_t, Y_t
        # X_t are linear combos of A blocks
        # Y_t are linear combos of B blocks
        
        # We need to define the coefficients for A and B for each of the 7 terms
        # And then for each term, the 4 schoolbook sub-terms
        
        # Strassen 7 terms:
        # 1: (A11+A22)(B11+B22)
        # 2: (A21+A22)B11
        # 3: A11(B12-B22)
        # 4: A22(B21-B11)
        # 5: (A11+A12)B22
        # 6: (A21-A11)(B11+B12)
        # 7: (A12-A22)(B21+B22)
        
        # For each term t, we have X_t (4x4) and Y_t (4x4)
        # X_t is formed by A blocks.
        # Y_t is formed by B blocks.
        
        # We iterate through the 7 terms
        for t in range(7):
            # Define X_t coefficients for A
            # X_t is 4x4.
            # We represent it as a dict of (i, j) -> coeff
            X = {}
            Y = {}
            
            if t == 0:
                # A11 + A22
                for i in range(2):
                    for j in range(2):
                        X[(i, j)] = 1
                        X[(i+2, j+2)] = 1
                # B11 + B22
                for i in range(2):
                    for j in range(2):
                        Y[(i, chunk_start+j)] = 1
                        Y[(i+2, chunk_start+j+2)] = 1
            elif t == 1:
                # A21 + A22
                for i in range(2):
                    for j in range(2):
                        X[(i+2, j)] = 1
                        X[(i+2, j+2)] = 1
                # B11
                for i in range(2):
                    for j in range(2):
                        Y[(i+2, chunk_start+j)] = 1
            elif t == 2:
                # A11
                for i in range(2):
                    for j in range(2):
                        X[(i, j)] = 1
                # B12 - B22
                for i in range(2):
                    for j in range(2):
                        Y[(i, chunk_start+j+2)] = 1
                        Y[(i+2, chunk_start+j+2)] = -1
            elif t == 3:
                # A22
                for i in range(2):
                    for j in range(2):
                        X[(i+2, j+2)] = 1
                # B21 - B11
                for i in range(2):
                    for j in range(2):
                        Y[(i+2, chunk_start+j)] = 1
                        Y[(i, chunk_start+j)] = -1
            elif t == 4:
                # A11 + A12
                for i in range(2):
                    for j in range(2):
                        X[(i, j)] = 1
                        X[(i, j+2)] = 1
                # B22
                for i in range(2):
                    for j in range(2):
                        Y[(i+2, chunk_start+j+2)] = 1
            elif t == 5:
                # A21 - A11
                for i in range(2):
                    for j in range(2):
                        X[(i+2, j)] = 1
                        X[(i, j)] = -1
                # B11 + B12
                for i in range(2):
                    for j in range(2):
                        Y[(i, chunk_start+j)] = 1
                        Y[(i, chunk_start+j+2)] = 1
            elif t == 6:
                # A12 - A22
                for i in range(2):
                    for j in range(2):
                        X[(i, j+2)] = 1
                        X[(i+2, j+2)] = -1
                # B21 + B22
                for i in range(2):
                    for j in range(2):
                        Y[(i+2, chunk_start+j)] = 1
                        Y[(i+2, chunk_start+j+2)] = 1
            
            # Now we have X (4x4 coeffs) and Y (4x4 coeffs)
            # We perform schoolbook multiplication of X and Y
            # Z = X * Y
            # Z[p, q] = sum_r X[p, r] Y[r, q]
            # Each term X[p, r] Y[r, q] is a rank-1 term
            # Wait, X[p, r] is a scalar coeff.
            # But X is a matrix of coeffs for A.
            # So X[p, r] means the coefficient of A[p, r] in the linear combo.
            # Similarly Y[r, q] is the coefficient of B[r, q] in the linear combo.
            # The product contributes to C[p, q].
            
            for r in range(4):
                # For each r, we have a term X[:, r] * Y[r, :]
                # This term contributes to all C[p, q]
                # But we need to decompose it into rank-1 terms?
                # No, X[:, r] is a vector of coeffs for A.
                # Y[r, :] is a vector of coeffs for B.
                # The outer product contributes to C.
                # Specifically, C[p, q] gets contribution X[p, r] * Y[r, q].
                # So we can add one term for each r.
                # The term is defined by:
                # U: coeffs of A in X[:, r]
                # V: coeffs of B in Y[r, :]
                # W: coeffs of C in the outer product
                
                u_coeffs = {}
                v_coeffs = {}
                w_coeffs = {}
                
                for i in range(4):
                    if (i, r) in X:
                        u_coeffs[(i, r)] = X[(i, r)]
                        
                for j in range(4):
                    if (r, j) in Y:
                        v_coeffs[(r, j)] = Y[(r, j)]
                        
                # W coeffs:
                # W[k, i] corresponds to C[i, k]
                # C[p, q] gets X[p, r] * Y[r, q]
                # So for each p, q:
                # W[q, p] += X[p, r] * Y[r, q]
                
                for p in range(4):
                    for q in range(4):
                        c_val = X.get((p, r), 0) * Y.get((r, q), 0)
                        if c_val != 0:
                            w_coeffs[(q, p)] = c_val
                
                add_term(u_coeffs, v_coeffs, w_coeffs)
                
    # Chunk 3: col 8
    # Schoolbook for A (4x4) * B (4x1)
    # C[i, 8] = sum_j A[i, j] B[j, 8]
    # 16 terms
    for i in range(4):
        for j in range(4):
            u_coeffs = {(i, j): 1}
            v_coeffs = {(j, 8): 1}
            w_coeffs = {(8, i): 1}
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    return {"U": U, "V": V, "W": W}

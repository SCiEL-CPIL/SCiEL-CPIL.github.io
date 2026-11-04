import numpy as np

DIMENSIONS = [4, 4, 8]

def get_strassen_2x2():
    """
    Returns U, V, W for 2x2 by 2x2 matrix multiplication.
    Shapes: (4, 7), (4, 7), (4, 7)
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen multipliers
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for U (A): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for V (B): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for W (C): 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Col 0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Col 1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Col 2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Col 3: M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Col 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Col 5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Col 6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def get_strassen_4x4():
    """
    Returns U, V, W for 4x4 by 4x4 matrix multiplication using Strassen.
    Shapes: (16, 49), (16, 49), (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_sub, V_sub, W_sub = get_strassen_2x2()
    
    # Strassen terms definitions
    # Each term is a list of (block_row, block_col, coeff) for A, B, and C
    # Block indices 0,1 for rows/cols of the 2x2 block matrix
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the linear forms for X (A part), Y (B part), Z (C part) for each M_r
    # X is 2x2, Y is 2x2, Z is 2x2
    
    # For each r, we have 7 sub-terms from the 2x2 decomposition
    # Total 49 terms
    
    col_offset = 0
    for r in range(7):
        # Define X, Y, Z block contributions for this r
        # X[i,j] is a linear combo of A blocks
        # Y[i,j] is a linear combo of B blocks
        # Z[i,j] is a linear combo of C blocks
        
        # We need to map the sub-problem indices to global indices
        # Sub U index i (0..3) -> X[i//2, i%2]
        # Sub V index j (0..3) -> Y[j//2, j%2]
        # Sub W index k (0..3) -> Z[k//2, k%2]
        
        for k in range(7):
            # Get the coefficients from the sub-decomposition
            u_vals = U_sub[:, k]
            v_vals = V_sub[:, k]
            w_vals = W_sub[:, k]
            
            # Expand U
            for i in range(4):
                val_u = u_vals[i]
                if val_u != 0:
                    xi = i // 2
                    xj = i % 2
                    # Find which A blocks contribute to X[xi, xj] in term r
                    # We need a table of X, Y, Z definitions for each r
                    
            # This approach requires defining the tables. Let's do it explicitly.
            
            col_offset += 1
            
    # This explicit expansion is tedious. Let's create the tables.
    
    # Tables for X (A), Y (B), Z (C) for each r (0..6)
    # Entry is list of (block_row, block_col, coeff)
    
    X_defs = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,1)], # M2
        [(0,0,1)],          # M3
        [(1,1,1)],          # M4
        [(0,0,1), (0,1,1)], # M5
        [(1,0,1), (0,0,-1)],# M6
        [(0,1,1), (1,1,-1)] # M7
    ]
    
    Y_defs = [
        [(0,0,1), (1,1,1)], # M1
        [(0,0,1)],          # M2
        [(0,1,1), (1,1,-1)],# M3
        [(1,0,1), (0,0,-1)],# M4
        [(1,1,1)],          # M5
        [(0,0,1), (0,1,1)], # M6
        [(1,0,1), (1,1,1)]  # M7
    ]
    
    Z_defs = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,-1)],# M2
        [(0,1,1), (1,1,1)], # M3
        [(0,0,1), (1,0,1)], # M4
        [(0,0,-1), (0,1,1)],# M5
        [(1,1,1)],          # M6
        [(0,0,1), (1,1,1)]  # M7
    ]
    
    # Wait, Z_defs should reflect the C formulas.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each r, Z is not just a block, but the contribution to C.
    # Actually, the sub-decomposition gives W_sub which maps to Z.
    # Z is the "result" of the sub-multiplication.
    # The sub-multiplication computes Z = X Y.
    # So Z_defs should be the identity? No.
    # The sub-decomposition is for the *tensor* of 2x2 by 2x2.
    # So W_sub contains the coefficients for the 4 entries of the result matrix Z.
    # So we just need to map the entries of Z to the global C blocks.
    # Z[0,0] corresponds to C block (0,0).
    # Z[0,1] corresponds to C block (0,1).
    # Z[1,0] corresponds to C block (1,0).
    # Z[1,1] corresponds to C block (1,1).
    
    # So Z mapping is always:
    # Z[i,j] -> C block (i,j) with coeff 1.
    
    # Let's re-verify.
    # The sub-problem is X Y = Z.
    # X is formed from A blocks.
    # Y is formed from B blocks.
    # Z is the result, which corresponds to C blocks.
    # So yes, Z mapping is identity.
    
    # Now fill U, V, W
    
    col_offset = 0
    for r in range(7):
        for k in range(7):
            # U expansion
            for i in range(4):
                val_u = U_sub[i, k]
                if val_u != 0:
                    xi = i // 2
                    xj = i % 2
                    # Find coeffs for X[xi, xj] in term r
                    for (br, bc, c) in X_defs[r]:
                        if br == xi and bc == xj:
                            # Add to global U
                            for p in range(2):
                                for q in range(2):
                                    global_i = 2*br + p
                                    global_j = 2*bc + q
                                    global_idx = global_i * 4 + global_j
                                    U[global_idx, col_offset] += c * val_u
            # V expansion
            for j in range(4):
                val_v = V_sub[j, k]
                if val_v != 0:
                    yi = j // 2
                    yj = j % 2
                    for (br, bc, c) in Y_defs[r]:
                        if br == yi and bc == yj:
                            for p in range(2):
                                for q in range(2):
                                    global_i = 2*br + p
                                    global_j = 2*bc + q
                                    global_idx = global_i * 4 + global_j
                                    V[global_idx, col_offset] += c * val_v
            # W expansion
            for l in range(4):
                val_w = W_sub[l, k]
                if val_w != 0:
                    zi = l // 2
                    zj = l % 2
                    # Z[zi, zj] maps to C block (zi, zj)
                    for p in range(2):
                        for q in range(2):
                            global_i = 2*zi + p
                            global_j = 2*zj + q
                            global_idx = global_i * 4 + global_j
                            W[global_idx, col_offset] += val_w
            col_offset += 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We need 4x4 by 4x8.
    # Split B into B1 (4x4) and B2 (4x4).
    # Decompose A B1 and A B2 separately.
    
    U1, V1, W1 = get_strassen_4x4()
    U2, V2, W2 = get_strassen_4x4()
    
    # Concatenate U
    U = np.concatenate([U1, U2], axis=1)
    
    # Concatenate V
    # V1 corresponds to B1 (cols 0-3).
    # V2 corresponds to B2 (cols 4-7).
    # We need to shift the column indices in V2 by 4.
    # V2 shape (16, 49).
    # Global V shape (32, 98).
    # V2 rows correspond to j (0..3) and k (0..3).
    # We need to map k (0..3) to k (4..7).
    # Index in V is j * p + k. Here p=4 for the sub-problem.
    # In global, p=8.
    # So for V2, index j * 4 + k should become j * 8 + (k + 4).
    
    V_global = np.zeros((32, 98))
    V_global[:, :49] = V1
    
    # Map V2
    for j in range(4):
        for k in range(4):
            local_idx = j * 4 + k
            global_idx = j * 8 + (k + 4)
            V_global[global_idx, 49:] = V2[local_idx, :]
            
    # Concatenate W
    # W1 corresponds to C1 (cols 0-3).
    # W2 corresponds to C2 (cols 4-7).
    # W shape (32, 98).
    # Index in W is k * n + i.
    # For W1, k (0..3) -> k (0..3).
    # For W2, k (0..3) -> k (4..7).
    
    W_global = np.zeros((32, 98))
    W_global[:, :49] = W1
    
    for k in range(4):
        for i in range(4):
            local_idx = k * 4 + i
            global_idx = (k + 4) * 4 + i
            W_global[global_idx, 49:] = W2[local_idx, :]
            
    return {"U": U, "V": V_global, "W": W_global}

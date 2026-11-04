import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 130.0

    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Known rank-28 solution via Strassen on 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # Strassen's 7 multiplications for 2x2 matrices
        # X, Y, Z are indices for A, B, C blocks
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-d)(g+h)
        # m7 = (b-d)(f+g)
        # C = [[m1+m4+m5-m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
        
        # Mapping to U, V, W
        # U corresponds to A elements, V to B elements, W to C elements
        # X_rows: [a, b, c, d] indices in A (flattened)
        # Y_rows: [e, f, g, h] indices in B (flattened)
        # Z_rows: [C00, C01, C10, C11] indices in C (flattened)
        
        # m1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1

        # m2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1

        # m3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1

        # m4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1

        # m5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1

        # m6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1

        # m7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Block 1: A1 (rows 0,1; cols 0,1) * B11 (rows 0,1; cols 0,1) -> C11 (rows 0,1; cols 0,1)
    # A indices: 0,1,4,5 (a,b,c,d)
    # B indices: 0,1,4,5 (e,f,g,h)
    # C indices: 0,2,1,3 (C00, C01, C10, C11) -- Note: C is 2x4, flattened.
    # C11 is top-left 2x2 of C.
    # C indices in flattened 2x4:
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # Top-left 2x2:
    # C00 -> 0
    # C01 -> 2
    # C10 -> 4
    # C11 -> 6
    # Wait, the previous code used Z1 = [0, 2, 1, 3].
    # Let's re-verify the trace convention W[k*n+i, r].
    # C is 2x4.
    # C[i, k].
    # For C11 (top-left 2x2 of result):
    # i=0, k=0 -> C[0,0] -> W[0*2+0] = W[0]
    # i=0, k=1 -> C[0,1] -> W[1*2+0] = W[2]
    # i=1, k=0 -> C[1,0] -> W[0*2+1] = W[1]
    # i=1, k=1 -> C[1,1] -> W[1*2+1] = W[3]
    # So Z rows for C11 are [0, 2, 1, 3] corresponding to [C00, C01, C10, C11].
    # This matches the previous code.

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    # Block 2: A2 (rows 0,1; cols 2,3) * B21 (rows 0,1; cols 2,3) -> C21 (rows 0,1; cols 2,3)
    # A indices: 2,3,6,7
    # B indices: 2,3,6,7
    # C indices:
    # C02 -> W[2*2+0] = W[4]
    # C03 -> W[3*2+0] = W[6]
    # C12 -> W[2*2+1] = W[5]
    # C13 -> W[3*2+1] = W[7]
    # Z2 = [4, 6, 5, 7]
    # Previous code used Z2 = [0, 2, 1, 3] for block 2?
    # Let's check previous code:
    # X2 = [2, 3, 6, 7]
    # Y2 = [8, 9, 12, 13]
    # Z2 = [0, 2, 1, 3]
    # This implies Y2 is rows 4,5 of B?
    # B is 4x4.
    # B11: rows 0,1 cols 0,1. Indices 0,1,4,5.
    # B12: rows 0,1 cols 2,3. Indices 2,3,6,7.
    # B21: rows 2,3 cols 0,1. Indices 8,9,12,13.
    # B22: rows 2,3 cols 2,3. Indices 10,11,14,15.
    
    # A is 2x4.
    # A1: cols 0,1. Indices 0,1,4,5.
    # A2: cols 2,3. Indices 2,3,6,7.
    
    # Result C is 2x4.
    # C = A1*B11 + A2*B21 + A1*B12? No.
    # A * B = [A1 A2] * [B11 B12; B21 B22]
    # C = A1*B11 + A2*B21 (for cols 0,1 of C)
    # C = A1*B12 + A2*B22 (for cols 2,3 of C)
    
    # So:
    # C_left (cols 0,1) = A1*B11 + A2*B21
    # C_right (cols 2,3) = A1*B12 + A2*B22
    
    # Block 1: A1 * B11 -> C_left_top (part of C_left)
    # Block 2: A2 * B21 -> C_left_bottom (part of C_left)
    # Block 3: A1 * B12 -> C_right_top (part of C_right)
    # Block 4: A2 * B22 -> C_right_bottom (part of C_right)
    
    # Let's re-map:
    # Block 1: A1 (X1) * B11 (Y11) -> C11 (Z11)
    # X1 = [0, 1, 4, 5]
    # Y11 = [0, 1, 4, 5]
    # Z11 = [0, 2, 1, 3] (C[0,0], C[0,1], C[1,0], C[1,1])
    
    # Block 2: A2 (X2) * B21 (Y21) -> C21 (Z21)
    # X2 = [2, 3, 6, 7]
    # Y21 = [8, 9, 12, 13]
    # Z21 = [0, 2, 1, 3] (C[0,0], C[0,1], C[1,0], C[1,1])
    # Note: This adds to the same C indices as Block 1.
    
    # Block 3: A1 (X1) * B12 (Y12) -> C12 (Z12)
    # X1 = [0, 1, 4, 5]
    # Y12 = [2, 3, 6, 7]
    # Z12 = [4, 6, 5, 7] (C[0,2], C[0,3], C[1,2], C[1,3])
    
    # Block 4: A2 (X2) * B22 (Y22) -> C22 (Z22)
    # X2 = [2, 3, 6, 7]
    # Y22 = [10, 11, 14, 15]
    # Z22 = [4, 6, 5, 7] (C[0,2], C[0,3], C[1,2], C[1,3])
    
    # Previous code:
    # add_block(0, X1, Y1, Z1) -> Block 1
    # add_block(7, X2, Y2, Z2) -> Block 2 (Y2 was [8,9,12,13], Z2 was [0,2,1,3])
    # add_block(14, X3, Y3, Z3) -> Block 3 (X3=X1, Y3=[2,3,6,7], Z3=[4,6,5,7])
    # add_block(21, X4, Y4, Z4) -> Block 4 (X4=X2, Y4=[10,11,14,15], Z4=[4,6,5,7])
    
    # This matches my derivation.
    
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        S = np.zeros((nm, mp, pn), dtype=float)
        for r in range(rank_val):
            S += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        return np.max(np.abs(S - T))

    # Initial verification
    max_err = verify_decomp(U_arr, V_arr, W_arr, rank)
    if max_err > 1e-10:
        return {"U": U, "V": V, "W": W}

    # Attempt rank reduction via linear dependency absorption
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    best_rank = rank

    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    current_rank = rank

    for _ in range(current_rank):
        if time.time() > deadline - 10:
            break
            
        if current_rank <= 1:
            break
            
        # Build M
        M = np.zeros((nm * mp * pn, current_rank), dtype=float)
        for r in range(current_rank):
            M[:, r] = np.einsum('a,b,c->abc', current_U[:, r], current_V[:, r], current_W[:, r]).flatten()
        
        # Check rank
        # Use SVD to find numerical rank
        try:
            U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
            tol = 1e-10
            num_rank = np.sum(s > tol)
            
            if num_rank < current_rank:
                # Find a column that is dependent
                # We can just pick the last one if the matrix is not full rank
                # But we need to find which one to remove and how to absorb.
                # A simpler approach: 
                # If rank is deficient, there is a null vector in M.
                # But we want to remove one column and adjust others.
                # This is equivalent to finding a linear dependency.
                
                # Let's try to remove column r_elim if it's in the span of others.
                # We can iterate through columns and check.
                reduced = False
                for r_elim in range(current_rank):
                    other_cols = [s for s in range(current_rank) if s != r_elim]
                    M_sub = M[:, other_cols]
                    rhs = M[:, r_elim]
                    
                    try:
                        # Solve M_sub * alpha = rhs
                        alpha, residuals, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
                        residual = np.max(np.abs(M_sub @ alpha - rhs))
                        if residual < 1e-8:
                            # Absorb
                            # T_elim = sum(alpha_j * T_j)
                            # T = T_elim + sum(T_j) = sum((1+alpha_j) * T_j)
                            new_U = current_U[:, other_cols].copy()
                            new_V = current_V[:, other_cols].copy()
                            new_W = current_W[:, other_cols].copy()
                            
                            # Update coefficients
                            # alpha is of length current_rank - 1
                            # We need to add 1 to each coefficient
                            coeffs = 1.0 + alpha
                            
                            for idx, j in enumerate(other_cols):
                                new_U[:, idx] *= coeffs[idx]
                                new_V[:, idx] *= coeffs[idx]
                                new_W[:, idx] *= coeffs[idx]
                                
                            new_rank = current_rank - 1
                            
                            # Verify
                            err = verify_decomp(new_U, new_V, new_W, new_rank)
                            if err < 1e-10:
                                current_U = new_U
                                current_V = new_V
                                current_W = new_W
                                current_rank = new_rank
                                if current_rank < best_rank:
                                    best_rank = current_rank
                                    best_U = current_U.copy()
                                    best_V = current_V.copy()
                                    best_W = current_W.copy()
                                reduced = True
                                break
                    except np.linalg.LinAlgError:
                        continue
                if reduced:
                    continue
        except Exception:
            break

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(val)
            result.append(row_list)
        return result

    U_final = to_list(best_U)
    V_final = to_list(best_V)
    W_final = to_list(best_W)

    return {"U": U_final, "V": V_final, "W": W_final}

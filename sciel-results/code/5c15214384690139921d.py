import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 110.0

    # We use a rank-28 Strassen-based decomposition for 2x4 * 4x4.
    # Split A (2x4) into two 2x2 blocks A1, A2 (columns 0-1, 2-3).
    # Split B (4x4) into four 2x2 blocks B11, B12, B21, B22.
    # C (2x4) = [A1*B11 + A2*B21,  A1*B12 + A2*B22] = [C1, C2] where C1, C2 are 2x2.
    # Each 2x2 product uses 7 Strassen terms, so total 4*7 = 28 terms.

    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)

    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen terms for a 2x2 multiplication C = AB:
    # M1 = (A11)(B12 - B22)
    # M2 = (A11 + A21)(B22)
    # M3 = (A22 - A11)(B11)
    # M4 = (A22)(B21 - B11)
    # M5 = (A11 + A22)(B11 + B22)
    # M6 = (A12 - A22)(B21 + B22)
    # M7 = (A21 - A11)(B11 + B12)
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M5
    # C21 = M3 + M5
    # C22 = M1 - M4 + M2 + M7

    # For each block, we map local (row, col) to global indices.
    # A has 2 rows, 4 cols. A[i, j] global index = i*m + j = i*4 + j.
    # B has 4 rows, 4 cols. B[i, j] global index = i*p + j = i*4 + j.
    # C has 2 rows, 4 cols. W index = k*n + i where k is C-col, i is C-row.
    #   C[i, k] -> W[k*n + i, r] = W[k*2 + i, r]

    # Block definitions: (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset)
    # A is 2x4, so A_row_offset is always 0.
    # Block 11: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    #   A_col_off=0, B_row_off=0, B_col_off=0, C_col_off=0
    # Block 12: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    #   A_col_off=0, B_row_off=0, B_col_off=2, C_col_off=2
    # Block 21: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    #   A_col_off=2, B_row_off=2, B_col_off=0, C_col_off=0
    # Block 22: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    #   A_col_off=2, B_row_off=2, B_col_off=2, C_col_off=2

    blocks = [
        # (a_col_off, b_row_off, b_col_off, c_col_off)
        (0, 0, 0, 0),  # Block 11
        (0, 0, 2, 2),  # Block 12
        (2, 2, 0, 0),  # Block 21
        (2, 2, 2, 2),  # Block 22
    ]

    col = 0
    for (a_co, b_ro, b_co, c_co) in blocks:
        # Local 2x2 indices: A has rows 0,1; cols a_co, a_co+1
        # B has rows b_ro, b_ro+1; cols b_co, b_co+1
        # C has rows 0,1; cols c_co, c_co+1

        # A local (r, c) -> global U index: r*4 + (a_co + c)
        # B local (r, c) -> global V index: (b_ro + r)*4 + (b_co + c)
        # C local (r, c) -> global W index: (c_co + c)*2 + r

        def a_idx(lr, lc):
            return lr * 4 + (a_co + lc)

        def b_idx(lr, lc):
            return (b_ro + lr) * 4 + (b_co + lc)

        def c_idx(lr, lc):
            return (c_co + lc) * 2 + lr

        # M1 = (A11)(B12 - B22)
        U[a_idx(0, 0)][col] += 1
        V[b_idx(0, 1)][col] += 1
        V[b_idx(1, 1)][col] -= 1
        W[c_idx(0, 1)][col] += 1
        W[c_idx(1, 1)][col] += 1
        col += 1

        # M2 = (A11 + A21)(B22)
        U[a_idx(0, 0)][col] += 1
        U[a_idx(1, 0)][col] += 1
        V[b_idx(1, 1)][col] += 1
        W[c_idx(0, 0)][col] -= 1
        W[c_idx(1, 1)][col] += 1
        col += 1

        # M3 = (A22 - A11)(B11)
        U[a_idx(0, 0)][col] -= 1
        U[a_idx(1, 1)][col] += 1
        V[b_idx(0, 0)][col] += 1
        W[c_idx(1, 0)][col] += 1
        W[c_idx(1, 1)][col] += 1
        col += 1

        # M4 = (A22)(B21 - B11)
        U[a_idx(1, 1)][col] += 1
        V[b_idx(1, 0)][col] += 1
        V[b_idx(0, 0)][col] -= 1
        W[c_idx(0, 0)][col] -= 1
        W[c_idx(1, 0)][col] -= 1
        col += 1

        # M5 = (A11 + A22)(B11 + B22)
        U[a_idx(0, 0)][col] += 1
        U[a_idx(1, 1)][col] += 1
        V[b_idx(0, 0)][col] += 1
        V[b_idx(1, 1)][col] += 1
        W[c_idx(0, 0)][col] += 1
        W[c_idx(0, 1)][col] += 1
        W[c_idx(1, 0)][col] += 1
        W[c_idx(1, 1)][col] += 1
        col += 1

        # M6 = (A12 - A22)(B21 + B22)
        U[a_idx(0, 1)][col] += 1
        U[a_idx(1, 1)][col] -= 1
        V[b_idx(1, 0)][col] += 1
        V[b_idx(1, 1)][col] += 1
        W[c_idx(0, 0)][col] += 1
        W[c_idx(1, 1)][col] += 1
        col += 1

        # M7 = (A21 - A11)(B11 + B12)
        U[a_idx(0, 0)][col] -= 1
        U[a_idx(1, 0)][col] += 1
        V[b_idx(0, 0)][col] += 1
        V[b_idx(0, 1)][col] += 1
        W[c_idx(1, 1)][col] += 1
        col += 1

    # Quick verification with random integer matrices
    import numpy as np
    rng = np.random.default_rng(42)
    A_rand = rng.integers(-5, 5, size=(n, m))
    B_rand = rng.integers(-5, 5, size=(m, p))
    C_expected = A_rand @ B_rand

    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)

    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] for all j... 
    # Actually C[i,k] = sum_j sum_r A[i,j] * U[i*m+j, r] * (sum_k' V[j*p+k', r] ... )
    # No, the decomposition is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j and r?
    # Standard form: T = sum_r U_r \otimes V_r \otimes W_r
    # C[i,k] = sum_{j,r} A[i,j] U[i*m+j, r] * (sum_{k',j'} ... )
    # Let me just use the standard verification:
    # C = A @ B where A is n x m, B is m x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... this isn't quite right.
    # The standard form is: C[i,k] = sum_j sum_r A[i,j] * (something)
    # Actually: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j and r.
    # Let's verify: sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]? No.
    # The tensor is: T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
    # And C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r A[i,j] B[j,k] T_{ijk} ... no.
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]? No.
    # It's: C[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_j' B[j',k] V[j'*p+k, r]) ... no.
    # Let's just compute: C_computed[i,k] = sum_j sum_r A[i,j] * U[i*m+j, r] * W[k*n+i, r] * ... 
    # Actually the standard verification is:
    # C_computed[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * ... no.
    # The correct form: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_r A[i,j] B[j,k] * (U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] / (U*V*W))? No.
    # The tensor decomposition means: for all i,j,k: delta(i,j,k) = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # where delta is the multiplication tensor.
    # So: C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r A[i,j] B[j,k] * [sum_r' ... ]
    # Actually: sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r]) * W[k*n+i, r]
    # This works if U, V, W are the factors. Let me verify this identity.
    # Yes, that's the standard form. Let me compute it.
    
    # A_part[r] = sum_j A[i,j] * U[i*m+j, r]  (for each i, r)
    # B_part[r] = sum_j B[j,k] * V[j*p+k, r]  (for each k, r)
    # C[i,k] = sum_r W[k*n+i, r] * A_part[i,r] * B_part[k,r]
    
    # Let me compute A_part: shape (n, rank)
    # A_part[i, r] = sum_j A[i,j] * U[i*m+j, r]
    A_part = np.zeros((n, rank), dtype=np.int64)
    for i in range(n):
        for r in range(rank):
            for j in range(m):
                A_part[i, r] += A_rand[i, j] * U_arr[i*m+j, r]
    
    # B_part[k, r] = sum_j B[j,k] * V[j*p+k, r]
    B_part = np.zeros((p, rank), dtype=np.int64)
    for k in range(p):
        for r in range(rank):
            for j in range(m):
                B_part[k, r] += B_rand[j, k] * V_arr[j*p+k, r]
    
    # C_computed[i,k] = sum_r W[k*n+i, r] * A_part[i,r] * B_part[k,r]
    C_computed = np.zeros((n, p), dtype=np.int64)
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                C_computed[i, k] += W_arr[k*n+i, r] * A_part[i, r] * B_part[k, r]
    
    if not np.array_equal(C_expected, C_computed):
        # Fallback to schoolbook
        rank = n * m * p
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

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

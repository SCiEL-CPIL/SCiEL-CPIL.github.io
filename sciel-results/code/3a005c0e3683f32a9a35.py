"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Rank-49 recursive Strassen decomposition for 4x4 x 4x4 matrix multiplication.

    Output convention: W[k*n + i, c].
    U[i*m + j, c], V[j*p + k, c].
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill

    n, m, p = 4, 4, 4
    rank = 49

    # Build U, V, W as sparse dictionaries
    U = {}
    V = {}
    W = {}

    def add_U(i, j, col, val):
        key = (i * m + j, col)
        U[key] = U.get(key, 0) + val

    def add_V(j, k, col, val):
        key = (j * p + k, col)
        V[key] = V.get(key, 0) + val

    def add_W(k, i, col, val):
        key = (k * n + i, col)
        W[key] = W.get(key, 0) + val

    # Block-level Strassen coefficients
    # A_side_coeffs[Nj][(br,bc)] = coefficient for A[br*2+r][bc*2+c] in A_side of Nj
    A_side_coeffs = {
        1: {(0,0):1, (1,1):1},
        2: {(1,0):1, (1,1):1},
        3: {(0,0):1},
        4: {(1,1):1},
        5: {(0,0):1, (0,1):1},
        6: {(1,0):1, (0,0):-1},
        7: {(0,1):1, (1,1):-1},
    }
    B_side_coeffs = {
        1: {(0,0):1, (1,1):1},
        2: {(0,0):1},
        3: {(0,1):1, (1,1):-1},
        4: {(1,0):1, (0,0):-1},
        5: {(1,1):1},
        6: {(0,0):1, (0,1):1},
        7: {(1,0):1, (1,1):1},
    }

    # Base Strassen Mk A-side and B-side entry coefficients
    M_A = {
        1: {(0,0):1, (1,1):1},
        2: {(1,0):1, (1,1):1},
        3: {(0,0):1},
        4: {(1,1):1},
        5: {(0,0):1, (0,1):1},
        6: {(1,0):1, (0,0):-1},
        7: {(0,1):1, (1,1):-1},
    }
    M_B = {
        1: {(0,0):1, (1,1):1},
        2: {(0,0):1},
        3: {(0,1):1, (1,1):-1},
        4: {(1,0):1, (0,0):-1},
        5: {(1,1):1},
        6: {(0,0):1, (0,1):1},
        7: {(1,0):1, (1,1):1},
    }

    # Base Strassen C coefficients: C[r][c] = sum_mk C_base[mk][(r,c)] * Mk
    C_base = {
        1: {(0,0):1, (1,1):1},
        2: {(1,0):1, (1,1):-1},
        3: {(0,1):1, (1,1):1},
        4: {(0,0):1, (1,0):1},
        5: {(0,0):-1, (0,1):1},
        6: {(1,1):1},
        7: {(0,0):1},
    }

    # Block-level C coefficients: output block (bi,bj) gets Nj with coefficient
    C_block = {
        (0,0): {1:1, 4:1, 5:-1, 7:1},
        (0,1): {3:1, 5:1},
        (1,0): {2:1, 4:1},
        (1,1): {1:1, 2:-1, 3:1, 6:1},
    }

    # Build U, V, W
    for j in range(1, 8):
        if time.time() > deadline:
            break
        for k in range(1, 8):
            col = (j - 1) * 7 + (k - 1)

            # U: A side
            for (ar, ac), a_coeff in M_A[k].items():
                for (br, bc), b_coeff in A_side_coeffs[j].items():
                    i_idx = br * 2 + ar
                    j_idx = bc * 2 + ac
                    val = b_coeff * a_coeff
                    if val != 0:
                        add_U(i_idx, j_idx, col, val)

            # V: B side
            for (ar2, ac2), a_coeff in M_B[k].items():
                for (br2, bc2), b_coeff in B_side_coeffs[j].items():
                    i_idx = br2 * 2 + ar2
                    j_idx = bc2 * 2 + ac2
                    val = b_coeff * a_coeff
                    if val != 0:
                        add_V(i_idx, j_idx, col, val)

            # W: output
            for (bi, bj), j_coeffs in C_block.items():
                if j not in j_coeffs:
                    continue
                block_coeff = j_coeffs[j]
                for (cr, cc), base_coeff in C_base[k].items():
                    if base_coeff == 0:
                        continue
                    i_idx = bi * 2 + cr
                    k_idx = bj * 2 + cc
                    val = block_coeff * base_coeff
                    if val != 0:
                        add_W(k_idx, i_idx, col, val)

    # Convert to dense lists
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]

    for (row, col), val in U.items():
        U_list[row][col] = val
    for (row, col), val in V.items():
        V_list[row][col] = val
    for (row, col), val in W.items():
        W_list[row][col] = val

    # Attempt rank reduction: remove columns that are linearly dependent
    # across the joint (U, V, W) stacking. A column c is redundant iff
    # its stacked vector lies in the span of the other columns.
    if time.time() < deadline - 5.0:
        import numpy as np
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)

        # Stack: a column c is redundant iff the stacked column
        # [U[:,c]; V[:,c]; W[:,c]] is in the span of the others.
        # We check via QR with column pivoting on the stacked matrix.
        stacked = np.vstack([U_arr, V_arr, W_arr])  # (192, rank)
        # Find rank and pivot
        try:
            Q, R, piv = np.linalg.qr(stacked, pivoting=True)
            # Determine numerical rank
            diag = np.abs(np.diag(R))
            tol = diag[0] * max(stacked.shape) * np.finfo(float).eps * 100
            num_rank = np.sum(diag > tol)

            if num_rank < rank:
                # Keep only the pivot columns in the first num_rank positions
                keep_cols = sorted(piv[:num_rank])
                U_new = U_arr[:, keep_cols]
                V_new = V_arr[:, keep_cols]
                W_new = W_arr[:, keep_cols]

                # Verify the decomposition still holds
                # Tensor check: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == (i==k)
                def check(Uc, Vc, Wc):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(Uc.shape[1]):
                                    s += Uc[i*m+j, c] * Vc[j*p+k, c] * Wc[k*n+i, c]
                                expected = 1.0 if (i == k) else 0.0
                                if abs(s - expected) > 1e-9:
                                    return False
                    return True

                if check(U_new, V_new, W_new):
                    U_list = U_new.tolist()
                    V_list = V_new.tolist()
                    W_list = W_new.tolist()
                    rank = num_rank
        except Exception:
            pass

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

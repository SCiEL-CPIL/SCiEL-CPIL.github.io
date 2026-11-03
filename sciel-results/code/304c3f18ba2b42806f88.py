import numpy as np
import time

def construct():
    _t0 = time.time()
    _deadline = 145.0

    n, m, p = 3, 4, 6
    RANK = 66
    U = [[0] * RANK for _ in range(n * m)]
    V = [[0] * RANK for _ in range(m * p)]
    W = [[0] * RANK for _ in range(p * n)]

    col = 0

    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)

        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1
        col += 1
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1
        col += 1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1
        col += 1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1
        col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = _idx_u(a_r, a_c)
        A01 = _idx_u(a_r, a_c + 1)
        A10 = _idx_u(a_r + 1, a_c)
        A11 = _idx_u(a_r + 1, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C00 = _idx_w(c_c, c_r)
        C01 = _idx_w(c_c + 1, c_r)
        C10 = _idx_w(c_c, c_r + 1)
        C11 = _idx_w(c_c + 1, c_r + 1)

        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1

        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1

        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1

        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1

        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1

        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1

        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1
        col += 1

    for j in range(3):
        bc = 2 * j
        _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
        _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)
        _add_strassen(1, 0, 0, bc, 1, 2 * j)
        _add_strassen(1, 2, 2, bc, 1, 2 * j)

    assert col == RANK

    # Check for linear dependency
    # Flatten U, V, W into a matrix of shape (RANK, n*m*m*p*p*n)
    # Actually, the tensor T is in R^{(n*m) x (m*p) x (p*n)}
    # We can flatten it to a vector of size (n*m)*(m*p)*(p*n)
    # But we only need to check if the 66 vectors are linearly dependent.
    # We can form a matrix M where each row is the flattened T_i.
    
    dim = (n * m) * (m * p) * (p * n)
    T_matrix = np.zeros((RANK, dim), dtype=float)
    
    for k in range(RANK):
        # T_k is defined by U[:, k], V[:, k], W[:, k]
        # T_k[i, j, l] = U[i, k] * V[j, k] * W[l, k]
        # We can construct the outer product
        u_vec = np.array(U[:, k])
        v_vec = np.array(V[:, k])
        w_vec = np.array(W[:, k])
        # Flatten the tensor
        t_flat = np.zeros(dim)
        # Efficiently fill
        # Iterate over non-zero entries
        for i in range(n * m):
            if u_vec[i] != 0:
                for j in range(m * p):
                    if v_vec[j] != 0:
                        for l in range(p * n):
                            if w_vec[l] != 0:
                                # Index in flat vector
                                idx = (i * (m * p) + j) * (p * n) + l
                                t_flat[idx] = u_vec[i] * v_vec[j] * w_vec[l]
        T_matrix[k] = t_flat

    # Check rank
    rank_T = np.linalg.matrix_rank(T_matrix)
    
    if rank_T < RANK:
        # Find null space
        # Use SVD
        U_svd, s, Vt = np.linalg.svd(T_matrix, full_matrices=False)
        # The last row of Vt corresponds to the null space if rank < RANK
        # Actually, Vt is (RANK, RANK). The null space is spanned by rows of Vt corresponding to zero singular values.
        # Let's just take the last row if rank is RANK-1.
        # To be safe, find the vector in null space.
        # We want c such that T_matrix @ c = 0.
        # The null space of T_matrix is the right null space.
        # Vt's last row is the right singular vector for the smallest singular value.
        # If rank < RANK, the last singular value is 0.
        c = Vt[-1, :]
        
        # Find the index with largest coefficient
        k_max = np.argmax(np.abs(c))
        c_k = c[k_max]
        
        if abs(c_k) > 1e-9:
            # We can eliminate T_{k_max}
            # T_{k_max} = - sum_{i != k_max} (c_i / c_k) T_i
            # New coeff for i != k_max is 1 + c_i / c_k
            # New U, V, W
            RANK_NEW = RANK - 1
            U_new = [[0] * RANK_NEW for _ in range(n * m)]
            V_new = [[0] * RANK_NEW for _ in range(m * p)]
            W_new = [[0] * RANK_NEW for _ in range(p * n)]
            
            new_col = 0
            for i in range(RANK):
                if i == k_max:
                    continue
                coeff = 1.0 + c[i] / c_k
                if abs(coeff) > 1e-9:
                    for row in range(n * m):
                        U_new[row][new_col] = U[row][i] * coeff
                    for row in range(m * p):
                        V_new[row][new_col] = V[row][i] * coeff
                    for row in range(p * n):
                        W_new[row][new_col] = W[row][i] * coeff
                    new_col += 1
            
            # It's possible that new_col < RANK_NEW if some coeffs became 0?
            # But we just shifted.
            # We should trim if new_col < RANK_NEW
            if new_col < RANK_NEW:
                U_new = [row[:new_col] for row in U_new]
                V_new = [row[:new_col] for row in V_new]
                W_new = [row[:new_col] for row in W_new]
            
            return {"U": U_new, "V": V_new, "W": W_new}

    return {"U": U, "V": V, "W": W}

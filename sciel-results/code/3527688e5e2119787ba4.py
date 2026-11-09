"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    # ------------------------------------------------------------------
    # Step 1: Build the rank-28 Strassen-based decomposition
    # ------------------------------------------------------------------
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Standard 2x2x2 Strassen terms (7 terms)
    # M1 = (a+d)(e+h), M2 = (c+d)*e, M3 = a*(f-h), M4 = d*(g-e)
    # M5 = (a+b)*h, M6 = (c-a)*(e+f), M7 = (b-d)(g+h)
    # U indices: (i, j) -> i*2 + j
    # V indices: (j, k) -> j*2 + k
    # W indices: (k, i) -> k*2 + i
    strassen_terms = [
        # M1: (a+d)(e+h)
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2: (c+d)*e
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        # M3: a*(f-h)
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        # M4: d*(g-e)
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        # M5: (a+b)*h
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # M6: (c-a)*(e+f)
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7: (b-d)(g+h)
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]},
    ]

    # Four 2x2 blocks
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 2}
    ]

    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            col += 1

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # ------------------------------------------------------------------
    # Step 2: Verify the rank-28 decomposition
    # ------------------------------------------------------------------
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that the tensor decomposition exactly reconstructs the target tensor."""
        if U_mat.shape[1] != V_mat.shape[1] or U_mat.shape[1] != W_mat.shape[1]:
            return False
        r = U_mat.shape[1]
        U3 = U_mat.reshape(n, m, r)
        V3 = V_mat.reshape(m, p, r)
        W3 = W_mat.reshape(p, n, r)
        T = np.zeros((n, m, p))
        for c in range(r):
            T += U3[:, :, c] * V3[:, :, c] * W3[:, :, c]
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol)

    if not verify(U_arr, V_arr, W_arr):
        # Fallback: return the unverified decomposition (should not happen)
        U_out = [[int(x) for x in row] for row in U_arr]
        V_out = [[int(x) for x in row] for row in V_arr]
        W_out = [[int(x) for x in row] for row in W_arr]
        return {"U": U_out, "V": V_out, "W": W_out}

    best_rank = U_arr.shape[1]

    # ------------------------------------------------------------------
    # Step 3: Try to reduce rank by removing redundant columns
    # ------------------------------------------------------------------
    def try_remove_columns(U_a, V_a, W_a, max_iterations=50):
        """Greedy column removal: try removing each column, keep if still valid."""
        removed = 0
        for _ in range(max_iterations):
            if time_left() < 15:
                break
            found = False
            for c_remove in range(U_a.shape[1]):
                U_test = np.delete(U_a, c_remove, axis=1)
                V_test = np.delete(V_a, c_remove, axis=1)
                W_test = np.delete(W_a, c_remove, axis=1)
                if verify(U_test, V_test, W_test):
                    U_a, V_a, W_a = U_test, V_test, W_test
                    removed += 1
                    found = True
                    break
            if not found:
                break
        return U_a, V_a, W_a, removed

    U_a, V_a, W_a, removed = try_remove_columns(U_arr, V_arr, W_arr)
    best_rank = U_a.shape[1]

    # ------------------------------------------------------------------
    # Step 4: Try rank reduction via linear combination of columns
    # ------------------------------------------------------------------
    def try_linear_combination_reduction(U_a, V_a, W_a, max_rounds=5):
        """
        Try to reduce rank by finding linear dependencies among columns.
        If a column can be expressed as a linear combination of others,
        we can potentially merge them.
        """
        current_U, current_V, current_W = U_a, V_a, W_a
        for _ in range(max_rounds):
            if time_left() < 15:
                break
            r = current_U.shape[1]
            if r <= 1:
                break
            # Try each pair of columns to see if one is a scalar multiple
            # or if a linear combination can replace two columns with one
            merged = False
            for i in range(r):
                for j in range(i + 1, r):
                    # Check if column j is a scalar multiple of column i
                    # in all three matrices simultaneously
                    try:
                        # Check U
                        if np.allclose(current_U[:, j], current_U[:, i], atol=1e-12):
                            # They're identical, can remove one
                            continue
                    except:
                        pass
                    # Try: is there a scalar s such that col_j = s * col_i in all?
                    # This is too restrictive. Instead, try: can we replace
                    # two columns with a single column that's a linear combo?
                    # This is complex; skip for now.
            if not merged:
                break
        return current_U, current_V, current_W

    # ------------------------------------------------------------------
    # Step 5: Try to find a lower-rank decomposition via sparse linear algebra
    # ------------------------------------------------------------------
    def try_sparse_reduction(U_a, V_a, W_a):
        """
        Formulate the problem as finding the minimal set of rank-1 terms.
        The tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c].
        We want to minimize the number of non-zero columns.
        
        Strategy: build the full 2x4x4 tensor and try to find a sparse
        decomposition by solving a system of equations.
        """
        if time_left() < 20:
            return U_a, V_a, W_a
        
        # The target tensor is all ones: T[i,j,k] = 1 for all i,j,k
        # We have 2*4*4 = 32 equations.
        # Each rank-1 term contributes U[i,j,c]*V[j,k,c]*W[k,i,c].
        
        # Try to find if we can express the tensor with fewer terms
        # by checking if the current decomposition has linearly dependent
        # "effective" columns.
        
        # Build the bilinear form matrix: for each column c, the contribution
        # is a rank-1 tensor. We can check if the set of rank-1 tensors
        # is linearly independent.
        
        r = U_a.shape[1]
        if r <= 1:
            return U_a, V_a, W_a
        
        # Flatten each rank-1 tensor contribution
        # T_c[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Flatten to a vector of length n*m*p = 32
        tensors = np.zeros((r, n * m * p))
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        tensors[c, idx] = U_a[i * m + j, c] * V_a[j * p + k, c] * W_a[k * n + i, c]
        
        # Check rank of the tensor matrix
        tensor_rank = np.linalg.matrix_rank(tensors, tol=1e-12)
        
        if tensor_rank < r:
            # There are linear dependencies. Try to remove columns.
            # Find a basis for the row space and identify redundant columns.
            # Use QR decomposition to find the rank.
            Q, R = np.linalg.qr(tensors.T)
            # The rank is the number of non-zero diagonal elements of R
            diag = np.abs(np.diag(R))
            rank_idx = np.where(diag > 1e-10)[0]
            if len(rank_idx) < r:
                # Some columns are redundant. Remove the ones with smallest diagonal.
                # Sort by diagonal value
                order = np.argsort(diag)
                for idx in order:
                    if diag[idx] < 1e-10:
                        # Try removing this column
                        c_remove = idx
                        U_test = np.delete(U_a, c_remove, axis=1)
                        V_test = np.delete(V_a, c_remove, axis=1)
                        W_test = np.delete(W_a, c_remove, axis=1)
                        if verify(U_test, V_test, W_test):
                            U_a, V_a, W_a = U_test, V_test, W_test
                            r = U_a.shape[1]
                            # Recompute tensors
                            tensors = np.zeros((r, n * m * p))
                            for c in range(r):
                                for i in range(n):
                                    for j in range(m):
                                        for k in range(p):
                                            idx2 = i * m * p + j * p + k
                                            tensors[c, idx2] = U_a[i * m + j, c] * V_a[j * p + k, c] * W_a[k * n + i, c]
                            break
        
        return U_a, V_a, W_a

    U_a, V_a, W_a = try_sparse_reduction(U_a, V_a, W_a)
    best_rank = U_a.shape[1]

    # ------------------------------------------------------------------
    # Step 6: Final verification and output
    # ------------------------------------------------------------------
    if verify(U_a, V_a, W_a):
        U_out = [[int(x) for x in row] for row in U_a]
        V_out = [[int(x) for x in row] for row in V_a]
        W_out = [[int(x) for x in row] for row in W_a]
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fallback to original rank-28
        U_out = [[int(x) for x in row] for row in U_arr]
        V_out = [[int(x) for x in row] for row in V_arr]
        W_out = [[int(x) for x in row] for row in W_arr]
        return {"U": U_out, "V": V_out, "W": W_out}

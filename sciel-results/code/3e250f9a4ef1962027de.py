import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 110.0

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * m + k, k * p + j, j * n + i] = 1.0

    # Trivial decomposition (rank 216) as fallback
    rank_triv = n * m * p
    U_triv = np.zeros((n * m, rank_triv), dtype=float)
    V_triv = np.zeros((m * p, rank_triv), dtype=float)
    W_triv = np.zeros((p * n, rank_triv), dtype=float)
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_triv[i * m + k, idx] = 1.0
                V_triv[k * p + j, idx] = 1.0
                W_triv[j * n + i, idx] = 1.0
                idx += 1

    # Block-Strassen with block size 2: 3x3 blocks, rank 189
    bs = 2
    nb = n // bs  # 3

    # Strassen 2x2 definitions (7 multiplications)
    # Each step: A_coeffs and B_coeffs as dict of (block_row, block_col) -> coeff
    # C_coeffs: dict of (block_row, block_col) -> coeff for the result
    strassen_steps = [
        # M1 = (A00 + A11)(B00 + B11)
        {'A': {(0, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1, (1, 1): 1},
         'C': {(0, 0): 1, (1, 1): 1}},
        # M2 = (A10 + A11)B00
        {'A': {(1, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1},
         'C': {(1, 0): 1, (1, 1): -1}},
        # M3 = A00(B01 - B11)
        {'A': {(0, 0): 1}, 'B': {(0, 1): 1, (1, 1): -1},
         'C': {(0, 1): 1, (1, 1): 1}},
        # M4 = A11(B10 - B00)
        {'A': {(1, 1): 1}, 'B': {(1, 0): 1, (0, 0): -1},
         'C': {(0, 0): 1, (1, 0): 1}},
        # M5 = (A00 + A01)B11
        {'A': {(0, 0): 1, (0, 1): 1}, 'B': {(1, 1): 1},
         'C': {(0, 0): -1, (0, 1): 1}},
        # M6 = (A10 - A00)(B00 + B01)
        {'A': {(1, 0): 1, (0, 0): -1}, 'B': {(0, 0): 1, (0, 1): 1},
         'C': {(1, 1): 1}},
        # M7 = (A01 - A11)(B10 + B11)
        {'A': {(0, 1): 1, (1, 1): -1}, 'B': {(1, 0): 1, (1, 1): 1},
         'C': {(0, 0): 1}},
    ]

    rank_bs = 189
    U2 = np.zeros((n * m, rank_bs), dtype=float)
    V2 = np.zeros((m * p, rank_bs), dtype=float)
    W2 = np.zeros((p * n, rank_bs), dtype=float)

    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    s_def = strassen_steps[s]
                    # U factor: rows indexed by (i, k)
                    for (ar, ac), coeff_a in s_def['A'].items():
                        i_glob = i_b * bs + ar
                        k_glob = k_b * bs + ac
                        U2[i_glob * m + k_glob, r_idx] += coeff_a
                    # V factor: rows indexed by (k, j)
                    for (br, bc), coeff_b in s_def['B'].items():
                        k_glob = k_b * bs + br
                        j_glob = j_b * bs + bc
                        V2[k_glob * p + j_glob, r_idx] += coeff_b
                    # W factor: rows indexed by (j, i)
                    for (cr, cc), coeff_c in s_def['C'].items():
                        i_glob = i_b * bs + cr
                        j_glob = j_b * bs + cc
                        W2[j_glob * n + i_glob, r_idx] += coeff_c
                    r_idx += 1

    # Verify Block-Strassen
    T_recon = np.einsum('ir,jr,kr->ijk', U2, V2, W2)
    if not np.allclose(T_recon, T, atol=1e-9):
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

    if time.time() > deadline - 5:
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    # Rank reduction: find linear dependencies among the 189 rank-1 terms
    # Strategy: The tensor T = sum_r u_r ⊗ v_r ⊗ w_r
    # If there exist coefficients c_r such that sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0,
    # then we can eliminate one term.
    #
    # A practical approach: check if any term can be expressed as a linear
    # combination of others by looking at the joint structure.
    #
    # Key insight: For fixed (i_b, k_b, j_b), the 7 Strassen terms form a
    # 7-dimensional subspace. But across different blocks, there may be
    # dependencies because the same U/V/W patterns repeat.
    #
    # Let's try: compute the rank of the "combined" matrix
    # where each column is a flattened (u_r, v_r, w_r) triple.
    # This gives an upper bound on the number of independent terms.
    #
    # Actually, a better approach: use the fact that the tensor has
    # only 216 non-zero entries. The 189 rank-1 terms must combine
    # to produce exactly those 216 ones.
    #
    # Let's try iterative elimination: for each term r, check if removing it
    # and adjusting others can still reconstruct T.
    #
    # More efficient: find the null space of the linear system
    # that maps coefficients to the tensor.
    #
    # Build the matrix M where M[:, r] is the flattened u_r ⊗ v_r ⊗ w_r
    # This is a (36*36*36) x 189 matrix, which is too large.
    #
    # Alternative: use the matricizations.
    # T_12 = sum_r u_r ⊗ (v_r ⊗ w_r) has shape (36, 36*36)
    # T_13 = sum_r u_r ⊗ (w_r ⊗ v_r) has shape (36, 36*36)
    # T_23 = sum_r v_r ⊗ (w_r ⊗ u_r) has shape (36, 36*36)
    #
    # The tensor rank is at least max(rank(T_12), rank(T_13), rank(T_23))
    # and at most min of those ranks is NOT true, but tensor rank >= max.
    #
    # Let's compute the matricization ranks to get a lower bound.
    T_12 = T.reshape(n * m, m * p * p * n)
    T_13 = T.reshape(n * m, p * n * m * p)
    T_23 = T.reshape(m * p, p * n * n * m)

    r12 = np.linalg.matrix_rank(T_12, tol=1e-9)
    r13 = np.linalg.matrix_rank(T_13, tol=1e-9)
    r23 = np.linalg.matrix_rank(T_23, tol=1e-9)

    # For 6x6x6 matrix multiplication, the matricization ranks are all 36.
    # So tensor rank >= 36, which is a weak lower bound.
    #
    # Let's try a different approach: exploit the block structure.
    # The 189 terms are organized as 27 blocks × 7 Strassen steps.
    # Within each block, the 7 terms are independent (Strassen is minimal for 2x2).
    # But across blocks, there may be dependencies.
    #
    # Observation: The U factor for block (i_b, k_b) and step s depends only on
    # (i_b, k_b, s). Similarly V depends on (k_b, j_b, s) and W on (i_b, j_b, s).
    #
    # For a fixed step s, consider all 27 terms (one for each (i_b, k_b, j_b)).
    # The U vectors are the same for all j_b (since U doesn't depend on j_b).
    # The V vectors are the same for all i_b (since V doesn't depend on i_b).
    # The W vectors are the same for all k_b (since W doesn't depend on k_b).
    #
    # This means that for fixed s, the 27 terms form a tensor of rank at most
    # (number of distinct U patterns) × (number of distinct V patterns) × (number of distinct W patterns)
    # But since U depends on (i_b, k_b), V on (k_b, j_b), W on (i_b, j_b),
    # and we have 3×3×3 = 27 combinations, the 27 terms for fixed s are
    # actually all independent in general.
    #
    # However, let's check numerically: for fixed s, do the 27 terms span
    # a 27-dimensional space, or is there redundancy?
    #
    # Let's try a greedy approach: iteratively remove terms that are
    # linearly dependent on the remaining ones.
    #
    # To check if term r is redundant, we need to see if u_r ⊗ v_r ⊗ w_r
    # is in the span of {u_s ⊗ v_s ⊗ w_s : s ≠ r}.
    # This is equivalent to checking if the vectorized form is in the span.
    #
    # A simpler check: if u_r is in span{u_s : s ≠ r} AND v_r is in span{v_s : s ≠ r}
    # AND w_r is in span{w_s : s ≠ r}, then term r is likely redundant.
    # But this is sufficient, not necessary.
    #
    # Let's try: for each term r, check if its U vector is in the span of
    # the other U vectors. If not, keep it. This is conservative but safe.
    #
    # Actually, let's try a more aggressive approach: use QR decomposition
    # on the U, V, W matrices to find independent columns.
    #
    # The key insight: if we can find a subset of columns of U, V, W that are
    # linearly independent, and the corresponding terms still reconstruct T,
    # then we have a valid decomposition with fewer terms.
    #
    # Let's try: compute the rank of U, V, W separately.
    rank_U = np.linalg.matrix_rank(U2, tol=1e-9)
    rank_V = np.linalg.matrix_rank(V2, tol=1e-9)
    rank_W = np.linalg.matrix_rank(W2, tol=1e-9)

    # If any of these is less than 189, we can potentially reduce the rank.
    # But we need to ensure the decomposition still holds.
    #
    # Let's try: use SVD to find a lower-rank approximation and check if it's exact.
    #
    # For U: U2 = U_svd @ S @ Vt_svd. If rank_U < 189, we can drop small singular values.
    # But we need exact reconstruction, so we can only drop exactly zero singular values.
    #
    # Let's check if there are exact zeros in the singular value decompositions.
    U_svd = np.linalg.svd(U2, full_matrices=False)
    V_svd = np.linalg.svd(V2, full_matrices=False)
    W_svd = np.linalg.svd(W2, full_matrices=False)

    U_svals = U_svd[1]
    V_svals = V_svd[1]
    W_svals = W_svd[1]

    # Count how many singular values are exactly zero (within numerical precision)
    U_rank = np.sum(U_svals > 1e-9)
    V_rank = np.sum(V_svals > 1e-9)
    W_rank = np.sum(W_svals > 1e-9)

    # If the minimum of these ranks is less than 189, we might be able to reduce.
    # But we need to be careful: reducing U alone doesn't guarantee the tensor
    # can still be reconstructed.
    #
    # Let's try a different approach: exploit the symmetry of the problem.
    # The matrix multiplication tensor is symmetric in a certain sense:
    # swapping (A, B) with (B^T, A^T) and transposing C gives the same tensor.
    #
    # Actually, let's just return the 189-rank solution. It's already better
    # than the 216-rank schoolbook, and finding a lower rank is a hard problem.
    #
    # But wait: let's try one more thing. The 189 terms include 27 blocks × 7 steps.
    # For each block, the 7 Strassen terms are minimal. But maybe we can share
    # some terms between adjacent blocks?
    #
    # This is getting complex. Let's just return the 189-rank solution.

    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

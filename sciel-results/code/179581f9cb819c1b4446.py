import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140

    n, m, p = DIMENSIONS

    # Build the target tensor T[i,j,k] for 2x4x4
    # i = row_a * 4 + col_a (A is 2x4)
    # j = row_b * 4 + col_b (B is 4x4)
    # k = col_c * 2 + row_c (C is 2x4, trace convention W[k*n+i,r])
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for row_a in range(n):
        for col_a in range(m):
            i = row_a * m + col_a
            for row_b in range(m):
                for col_b in range(p):
                    j = row_b * p + col_b
                    for row_c in range(n):
                        for col_c in range(p):
                            k = col_c * n + row_c
                            if col_a == row_b and row_a == row_c and col_b == col_c:
                                T[i, j, k] = 1.0

    # --- Step 1: Build the rank-28 Strassen block decomposition ---
    # Strassen 2x2x2 factors
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ], dtype=float)

    V_s = np.array([
        [1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1]
    ], dtype=float)

    # Derive W_s by solving the linear system for 2x2x2
    T_2 = np.zeros((4, 4, 4), dtype=float)
    for ra in range(2):
        for ca in range(2):
            ii = ra * 2 + ca
            for rb in range(2):
                for cb in range(2):
                    jj = rb * 2 + cb
                    for rc in range(2):
                        for cc in range(2):
                            kk = cc * 2 + rc
                            if ca == rb and ra == rc and cb == cc:
                                T_2[ii, jj, kk] = 1.0

    M_sys = np.zeros((64, 28), dtype=float)
    T_vec = np.zeros(64, dtype=float)
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                eq = ii * 16 + jj * 4 + kk
                T_vec[eq] = T_2[ii, jj, kk]
                for s in range(7):
                    M_sys[eq, kk * 7 + s] = U_s[ii, s] * V_s[jj, s]

    W_vec, _, _, _ = np.linalg.lstsq(M_sys, T_vec, rcond=None)
    W_s = W_vec.reshape(4, 7)

    # Verify Strassen decomposition
    err_s = 0.0
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                val = np.sum(U_s[ii, :] * V_s[jj, :] * W_s[kk, :])
                err_s = max(err_s, abs(val - T_2[ii, jj, kk]))

    if err_s > 1e-9:
        # Fallback: schoolbook rank 32
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Build the rank-28 Strassen decomposition for 2x4x4
    U28 = np.zeros((8, 28), dtype=float)
    V28 = np.zeros((16, 28), dtype=float)
    W28 = np.zeros((8, 28), dtype=float)

    offset = 0
    for bi in range(2):
        for bj in range(2):
            for ra in range(2):
                for ca in range(2):
                    local_i = ra * 2 + ca
                    global_i = ra * 4 + (2 * bi + ca)
                    U28[global_i, offset:offset + 7] = U_s[local_i, :]
            for rb in range(2):
                for cb in range(2):
                    local_j = rb * 2 + cb
                    global_j = (2 * bi + rb) * 4 + (2 * bj + cb)
                    V28[global_j, offset:offset + 7] = V_s[local_j, :]
            for rc in range(2):
                for cc in range(2):
                    local_k = cc * 2 + rc
                    global_k = (2 * bj + cc) * 2 + rc
                    W28[global_k, offset:offset + 7] = W_s[local_k, :]
            offset += 7

    # Verify rank-28 decomposition
    def verify_decomposition(U_full, V_full, W_full, T_target):
        max_err = 0.0
        for i in range(U_full.shape[0]):
            for j in range(V_full.shape[0]):
                for k in range(W_full.shape[0]):
                    val = np.dot(U_full[i, :] * V_full[j, :], W_full[k, :])
                    err = abs(val - T_target[i, j, k])
                    if err > max_err:
                        max_err = err
        return max_err

    err28 = verify_decomposition(U28, V28, W28, T)
    if err28 > 1e-9:
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_rank = 28
    best_U, best_V, best_W = U28.copy(), V28.copy(), W28.copy()

    # --- Step 2: Algebraic elimination search for rank reduction ---
    # Key insight: The 28 terms are grouped in 4 blocks of 7 (Strassen blocks).
    # Within each block, the 7 terms are linearly independent (rank 7 is minimal for 2x2x2).
    # But ACROSS blocks, there may be linear dependencies if we allow the factor matrices
    # to be re-parameterized.
    #
    # Strategy: Fix the U and V factors (they encode the input structure), and search
    # for a W matrix with fewer non-zero columns that still satisfies the tensor equation.
    #
    # More precisely: The tensor equation is:
    #   T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # For fixed U, V, this is a linear system in W. The system has 512 equations
    # (8*16*8) and 28 unknowns (W is 8x28). If the system is consistent, we can
    # check if any column of W is in the span of others, allowing rank reduction.
    #
    # Actually, a better approach: Try to find a rank-r decomposition by
    # selecting r columns from the 28 and checking if they span the solution space.
    #
    # Even better: Use the structure. The 4 Strassen blocks operate on disjoint
    # (i,j) pairs in some sense. Let's check: which (i,j) pairs are touched by
    # which block?

    # Block (bi,bj) touches:
    #   i: rows 0,1 with cols (2*bi + 0, 2*bi + 1) -> i in {2*bi, 2*bi+1, 4+2*bi, 5+2*bi}
    #   j: rows (2*bi + 0, 2*bi + 1) with cols (2*bj + 0, 2*bj + 1) -> j in {(2*bi)*4+(2*bj), ...}
    #
    # Let's enumerate which (i,j) pairs each block covers:
    block_ijs = {}
    for bi in range(2):
        for bj in range(2):
            ijs = set()
            for ra in range(2):
                for ca in range(2):
                    gi = ra * 4 + (2 * bi + ca)
                    for rb in range(2):
                        for cb in range(2):
                            gj = (2 * bi + rb) * 4 + (2 * bj + cb)
                            ijs.add((gi, gj))
            block_ijs[(bi, bj)] = ijs

    # Check overlap between blocks
    blocks = list(block_ijs.keys())
    overlap_info = {}
    for a, b in combinations(blocks, 2):
        common = block_ijs[a] & block_ijs[b]
        if common:
            overlap_info[(a, b)] = common

    # If blocks have disjoint (i,j) supports, no cross-block cancellation is possible
    # and rank 28 is optimal. Let's verify.
    has_overlap = False
    for (a, b), common in overlap_info.items():
        if common:
            has_overlap = True
            break

    if has_overlap:
        # There IS overlap, so cross-block dependencies may exist.
        # Try to find a lower-rank decomposition by solving a global system.
        #
        # Approach: Assume rank r < 28. We need to find U (8xr), V (16xr), W (8xr)
        # such that T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r].
        #
        # This is a non-linear problem. But we can use the following trick:
        # Fix U and V (from the rank-28 solution, but with some columns zeroed out),
        # and solve for W. If the system is consistent, we have a valid decomposition.
        #
        # Try all subsets of 27 columns (remove one column at a time):
        # For each column c to remove, set U[:,c]=0, V[:,c]=0, and re-solve for W.
        # But this won't work because U and V are already fixed.
        #
        # Better: Use the fact that the tensor equation is linear in W for fixed U,V.
        # The system is: for each (i,j,k), sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k].
        # This is a system of 512 linear equations in 28 unknowns W[k,r].
        # If the system is consistent (which it is for rank 28), we can check if
        # the solution space has dimension > 0, meaning we can perturb W.
        #
        # Actually, let's try a different approach: directly search for rank 27.
        # We'll use the method of "fixing two factors and solving for the third"
        # but with a structured initialization.

        # Try rank 27: Remove one column from the rank-28 solution and see if
        # the remaining 27 columns can be adjusted to still satisfy the tensor.
        #
        # For each column c in 0..27:
        #   Set U[:,c] = 0, V[:,c] = 0 (effectively removing column c)
        #   But we can't just zero it out; we need to redistribute.
        #
        # Alternative: Use the bilinear structure. The tensor T can be written as:
        #   T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
        # where u_r is in R^8, v_r in R^16, w_r in R^8.
        #
        # For rank 27, we need to find 27 such triples. This is a 3-way tensor
        # decomposition problem, which is NP-hard in general.
        #
        # However, we can use the following heuristic:
        # 1. Start with the rank-28 solution.
        # 2. For each pair of columns (c1, c2), check if u_{c1} is in the span
        #    of {u_r : r != c1} AND v_{c1} is in the span of {v_r : r != c1}.
        #    If both hold, then column c1 is redundant and we can remove it.
        #
        # Let's check this:

        # Check if any column of U is in the span of the others
        U_rank_full = np.linalg.matrix_rank(U28)
        V_rank_full = np.linalg.matrix_rank(V28)
        W_rank_full = np.linalg.matrix_rank(W28)

        # If U has rank < 28, some columns are linearly dependent
        # Similarly for V and W.
        # But for a valid decomposition, we need the TENSORS u_r ⊗ v_r ⊗ w_r
        # to be linearly independent, not just the individual vectors.

        # Let's try a direct approach: solve for a rank-27 decomposition
        # by fixing U and V with 27 columns and solving for W.
        #
        # We'll try removing each column in turn:
        for c_remove in range(28):
            if time.time() > deadline - 10:
                break

            # Create U27, V27 by removing column c_remove
            mask = [r for r in range(28) if r != c_remove]
            U27 = U28[:, mask]
            V27 = V28[:, mask]

            # Solve for W27: T[i,j,k] = sum_r U27[i,r]*V27[j,r]*W27[k,r]
            # This is a linear system in W27 (8x27 = 216 unknowns)
            # with 512 equations.
            #
            # Build the system:
            # For each (i,j,k): sum_r U27[i,r]*V27[j,r]*W27[k,r] = T[i,j,k]
            # The unknowns are W27[k,r] for k in 0..7, r in 0..26.
            #
            # This is a linear system: A * W27_vec = T_vec
            # where A has 512 rows and 216 columns.

            A = np.zeros((512, 216), dtype=float)
            T_vec = np.zeros(512, dtype=float)

            for i in range(8):
                for j in range(16):
                    for k in range(8):
                        eq = i * 128 + j * 8 + k
                        T_vec[eq] = T[i, j, k]
                        for r in range(27):
                            # W27[k, r] is unknown, indexed as k*27 + r
                            A[eq, k * 27 + r] = U27[i, r] * V27[j, r]

            # Solve the least squares system
            W27_vec, residuals, rank_A, sv = np.linalg.lstsq(A, T_vec, rcond=None)
            W27 = W27_vec.reshape(8, 27)

            # Verify the decomposition
            max_err = 0.0
            for i in range(8):
                for j in range(16):
                    for k in range(8):
                        val = np.dot(U27[i, :] * V27[j, :], W27[k, :])
                        err = abs(val - T[i, j, k])
                        if err > max_err:
                            max_err = err

            if max_err < 1e-9:
                # Found a valid rank-27 decomposition!
                best_rank = 27
                best_U = U27
                best_V = V27
                best_W = W27
                break
            else:
                # Column c_remove cannot be removed with this U, V structure.
                # But maybe we can adjust U and V?
                # For now, just try the next column.
                pass

        # If we didn't find rank 27 by removing a single column, try removing
        # two columns at a time (but this is 28*27/2 = 378 combinations, feasible).
        if best_rank == 28 and time.time() < deadline - 20:
            for c1 in range(28):
                if time.time() > deadline - 10:
                    break
                for c2 in range(c1 + 1, 28):
                    if time.time() > deadline - 10:
                        break

                    mask = [r for r in range(28) if r != c1 and r != c2]
                    U26 = U28[:, mask]
                    V26 = V28[:, mask]

                    A = np.zeros((512, 208), dtype=float)
                    T_vec = np.zeros(512, dtype=float)

                    for i in range(8):
                        for j in range(16):
                            for k in range(8):
                                eq = i * 128 + j * 8 + k
                                T_vec[eq] = T[i, j, k]
                                for r in range(26):
                                    A[eq, k * 26 + r] = U26[i, r] * V26[j, r]

                    W26_vec, residuals, rank_A, sv = np.linalg.lstsq(A, T_vec, rcond=None)
                    W26 = W26_vec.reshape(8, 26)

                    max_err = 0.0
                    for i in range(8):
                        for j in range(16):
                            for k in range(8):
                                val = np.dot(U26[i, :] * V26[j, :], W26[k, :])
                                err = abs(val - T[i, j, k])
                                if err > max_err:
                                    max_err = err

                    if max_err < 1e-9:
                        best_rank = 26
                        best_U = U26
                        best_V = V26
                        best_W = W26
                        break
                if best_rank < 28:
                    break

    # Final verification
    final_err = verify_decomposition(best_U, best_V, best_W, T)
    if final_err > 1e-9:
        # Fallback to schoolbook
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 3, 4, 6
    U = [[0] * 66 for _ in range(n * m)]
    V = [[0] * 66 for _ in range(m * p)]
    W = [[0] * 66 for _ in range(p * n)]

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        if time.time() > deadline:
            return
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)

        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        if time.time() > deadline:
            return
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1

    # Build the 66-term decomposition
    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)
    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)

    base_rank = col

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Convert to numpy for linear algebra
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    # Build the 72 x 66 matrix M where M[flat_ijk, c] = U[i,c]*V[j,c]*W[k,c]
    # flat_ijk = i*m*p + j*p + k
    M = np.zeros((n * m * p, base_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                flat_idx = i * m * p + j * p + k
                for c in range(base_rank):
                    M[flat_idx, c] = Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]

    # The target tensor T[flat_ijk] = 1 for all (i,j,k)
    # We need sum_c M[flat_ijk, c] = 1 for all flat_ijk
    # Check: each column c contributes M[:,c], and sum should be ones

    # Find rank of M
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    except Exception:
        rank_M = base_rank

    best_result = {"U": U, "V": V, "W": W}

    if rank_M < base_rank:
        # Find null space of M
        try:
            null_space = np.linalg.null_space(M)
            null_dim = null_space.shape[1]
        except Exception:
            null_dim = 0

        # Try to reduce rank by redistributing dependent terms
        # For each null vector, we can express one column as a combination of others
        # But the redistribution must maintain the tensor structure (bilinear form)
        # This is non-trivial, so we try a heuristic approach:
        # Try removing each term and checking if the remaining terms can be scaled
        # to still produce the correct tensor

        for ns_idx in range(null_dim):
            if time.time() > deadline - 5:
                break
            alpha = null_space[:, ns_idx]
            # Find the column with largest |alpha| to remove
            col_to_remove = int(np.argmax(np.abs(alpha)))
            if abs(alpha[col_to_remove]) < 1e-10:
                continue

            # Try to redistribute: for each remaining column c, 
            # we need to find new U', V', W' such that the tensor is preserved
            # Simple approach: scale each remaining column to compensate
            # This only works if the removed term's contribution can be matched
            # by scaling existing terms (rarely works for non-trivial cases)

            # Alternative: try to find a rank-(base_rank-1) decomposition by
            # solving a linear system for the redistribution coefficients
            # For each tensor element (i,j,k):
            #   sum_{c != col_to_remove} delta_c(i,j,k) * U[i,c]*V[j,c]*W[k,c] 
            #     = U[i,col_to_remove]*V[j,col_to_remove]*W[k,col_to_remove]
            # This is nonlinear in delta, so we try a different approach:
            # Modify U, V, W directly by adding the removed term's contribution
            # distributed across other terms

            # Heuristic: try to absorb the removed term into each other term
            # by adjusting one of the three matrices
            for target_col in range(base_rank):
                if target_col == col_to_remove:
                    continue
                if time.time() > deadline - 5:
                    break

                # Try adjusting W: W'[k, target_col] = W[k, target_col] + 
                #   lambda * W[k, col_to_remove] * (U[i,col_to_remove]*V[j,col_to_remove]) / 
                #   (U[i,target_col]*V[j,target_col])
                # This only works if the ratio is constant, which is unlikely
                # Skip this complex approach and just try dropping zero terms

                break  # Skip complex redistribution for now

        # Fallback: just drop terms that are exactly zero in M
        keep = []
        for c in range(base_rank):
            if np.all(np.abs(M[:, c]) < 1e-12):
                continue
            keep.append(c)

        if len(keep) < base_rank:
            new_rank = len(keep)
            U2 = [[0] * new_rank for _ in range(n * m)]
            V2 = [[0] * new_rank for _ in range(m * p)]
            W2 = [[0] * new_rank for _ in range(p * n)]
            for new_c, old_c in enumerate(keep):
                for i in range(n * m):
                    U2[i][new_c] = U[i][old_c]
                for j in range(m * p):
                    V2[j][new_c] = V[j][old_c]
                for k in range(p * n):
                    W2[k][new_c] = W[k][old_c]
            best_result = {"U": U2, "V": V2, "W": W2}

    return best_result

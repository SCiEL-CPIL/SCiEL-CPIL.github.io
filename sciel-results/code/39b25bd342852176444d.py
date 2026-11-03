import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32

    # Strassen 2x2*2x2 decomposition (rank 7)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # c=0: M1 = (a00+a11)(b00+b11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    # c=1: M2 = (a10+a11)*b00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    # c=2: M3 = a00*(b01-b11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    # c=3: M4 = a11*(b10-b00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    # c=4: M5 = (a00+a01)*b11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    # c=5: M6 = (a10-a00)*(b00+b01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    # c=6: M7 = (a01-a11)*(b10+b11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Block Strassen for 4x4*4x4 (rank 49 per half)
    A_specs = [
        [(1, 0, 0), (1, 1, 1)],   # P0: A00 + A11
        [(1, 1, 0), (1, 1, 1)],   # P1: A10 + A11
        [(1, 0, 0)],              # P2: A00
        [(1, 1, 1)],              # P3: A11
        [(1, 0, 0), (1, 0, 1)],   # P4: A00 + A01
        [(1, 1, 0), (-1, 0, 0)],  # P5: A10 - A00
        [(1, 0, 1), (-1, 1, 1)],  # P6: A01 - A11
    ]

    B_specs = [
        [(1, 0, 0), (1, 1, 1)],   # P0: B00 + B11
        [(1, 0, 0)],              # P1: B00
        [(1, 0, 1), (-1, 1, 1)],  # P2: B01 - B11
        [(1, 1, 0), (-1, 0, 0)],  # P3: B10 - B00
        [(1, 1, 1)],              # P4: B11
        [(1, 0, 0), (1, 0, 1)],   # P5: B00 + B01
        [(1, 1, 0), (1, 1, 1)],   # P6: B10 + B11
    ]

    C_specs = [
        {(0, 0): 1, (1, 1): 1},        # P0: C00 + C11
        {(1, 0): 1, (1, 1): -1},       # P1: C10 - C11
        {(0, 1): 1, (1, 1): 1},        # P2: C01 + C11
        {(0, 0): 1, (1, 0): 1},        # P3: C00 + C10
        {(0, 0): -1, (0, 1): 1},       # P4: -C00 + C01
        {(1, 1): 1},                   # P5: C11
        {(0, 0): 1},                   # P6: C00
    ]

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_dict, v_dict, w_dict):
        U_row = np.zeros(nm)
        V_row = np.zeros(mp)
        W_row = np.zeros(pn)
        for (i, j), val in u_dict.items():
            U_row[i * m + j] += val
        for (j, k), val in v_dict.items():
            V_row[j * p + k] += val
        for (k, i), val in w_dict.items():
            W_row[k * n + i] += val
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)

    def strassen_4x4x4(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        for k_p in range(7):
            for m_s in range(7):
                u_dict = {}
                v_dict = {}
                w_dict = {}

                for (coeff, br, bc) in A_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            u_val = U_2[r * 2 + c, m_s]
                            if u_val != 0:
                                gi = A_row_off + br * 2 + r
                                gj = A_col_off + bc * 2 + c
                                u_dict[(gi, gj)] = u_dict.get((gi, gj), 0) + coeff * u_val

                for (coeff, br, bc) in B_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            v_val = V_2[r * 2 + c, m_s]
                            if v_val != 0:
                                gj = B_row_off + br * 2 + r
                                gk = B_col_off + bc * 2 + c
                                v_dict[(gj, gk)] = v_dict.get((gj, gk), 0) + coeff * v_val

                for (cbr, cbc), coeff_c in C_specs[k_p].items():
                    for r in range(2):
                        for c in range(2):
                            w_val = W_2[r * 2 + c, m_s]
                            if w_val != 0:
                                gi = C_row_off + cbr * 2 + r
                                gk = C_row_off + cbc * 2 + c
                                w_dict[(gk, gi)] = w_dict.get((gk, gi), 0) + coeff_c * w_val

                if u_dict and v_dict and w_dict:
                    add_term(u_dict, v_dict, w_dict)

    # Apply for B_left (cols 0-3) and B_right (cols 4-7)
    strassen_4x4x4(A_row_off=0, A_col_off=0, B_row_off=0, B_col_off=0, C_row_off=0, C_col_off=0)
    strassen_4x4x4(A_row_off=0, A_col_off=0, B_row_off=0, B_col_off=4, C_row_off=0, C_col_off=4)

    R = len(U_list)
    U = np.array(U_list).T  # shape (nm, R)
    V = np.array(V_list).T  # shape (mp, R)
    W = np.array(W_list).T  # shape (pn, R)

    # Verify
    error = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j, :]
            for k in range(p):
                val = np.dot(u_row, V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    error += abs(val - 1.0)

    if error > 1e-8:
        U_school = np.zeros((nm, n*m*p))
        V_school = np.zeros((mp, n*m*p))
        W_school = np.zeros((pn, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

    # Rank reduction: find linear dependencies among columns
    # A column c is redundant if it can be expressed as a linear combination of other columns
    # such that the tensor identity still holds.
    # We use a greedy approach: try to remove each column by solving for its coefficients
    # in terms of the remaining columns, then verify.

    def verify_decomp(U_mat, V_mat, W_mat):
        err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_mat[i*m+j, :]
                for k in range(p):
                    val = np.dot(u_row, V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > 1e-10:
                        err += abs(val - 1.0)
        return err

    # Greedy rank reduction
    # For each column, check if it's in the span of others (using SVD)
    # If so, try to remove it and verify.

    R_cur = R
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()

    # Build list of active column indices
    active = list(range(R_cur))

    # Try removing columns one by one (greedy)
    # We'll iterate multiple passes until no more removals are possible
    max_passes = 5
    for _pass in range(max_passes):
        if time.time() > deadline - 30:
            break

        removed_any = False
        new_active = []

        for idx in active:
            if time.time() > deadline - 30:
                break

            # Try removing column idx
            # Check if column idx is in the span of the other active columns
            # using the U matrix (if U[idx] is in span of other U columns,
            # and similarly for V and W, then we can potentially remove it)

            others = [a for a in active if a != idx]
            if len(others) == 0:
                new_active.append(idx)
                continue

            # Check U: is U[:, idx] in span of U[:, others]?
            U_others = U_cur[:, others]
            U_target = U_cur[:, idx]

            # SVD to check if in span
            try:
                U_svd = np.linalg.svd(U_others, full_matrices=False)
                U_rank = np.sum(U_svd[1] > 1e-10)
                if U_rank == 0:
                    if np.allclose(U_target, 0, atol=1e-10):
                        # U_target is zero, can remove
                        pass
                    else:
                        new_active.append(idx)
                        continue
                else:
                    # Project U_target onto span of U_others
                    U_Vt = U_svd[2].T[:U_rank]
                    coeffs_U = U_Vt @ U_target
                    U_proj = U_Vt.T @ coeffs_U
                    if not np.allclose(U_target, U_proj, atol=1e-8):
                        new_active.append(idx)
                        continue
            except np.linalg.LinAlgError:
                new_active.append(idx)
                continue

            # Check V
            V_others = V_cur[:, others]
            V_target = V_cur[:, idx]
            try:
                V_svd = np.linalg.svd(V_others, full_matrices=False)
                V_rank = np.sum(V_svd[1] > 1e-10)
                if V_rank == 0:
                    if not np.allclose(V_target, 0, atol=1e-10):
                        new_active.append(idx)
                        continue
                else:
                    V_Vt = V_svd[2].T[:V_rank]
                    coeffs_V = V_Vt @ V_target
                    V_proj = V_Vt.T @ coeffs_V
                    if not np.allclose(V_target, V_proj, atol=1e-8):
                        new_active.append(idx)
                        continue
            except np.linalg.LinAlgError:
                new_active.append(idx)
                continue

            # Check W
            W_others = W_cur[:, others]
            W_target = W_cur[:, idx]
            try:
                W_svd = np.linalg.svd(W_others, full_matrices=False)
                W_rank = np.sum(W_svd[1] > 1e-10)
                if W_rank == 0:
                    if not np.allclose(W_target, 0, atol=1e-10):
                        new_active.append(idx)
                        continue
                else:
                    W_Vt = W_svd[2].T[:W_rank]
                    coeffs_W = W_Vt @ W_target
                    W_proj = W_Vt.T @ coeffs_W
                    if not np.allclose(W_target, W_proj, atol=1e-8):
                        new_active.append(idx)
                        continue
            except np.linalg.LinAlgError:
                new_active.append(idx)
                continue

            # All three are in span. Now verify that removing this column
            # still gives a valid decomposition.
            # We need to find coefficients such that:
            # U[:, idx] = sum_c alpha_c * U[:, c]
            # V[:, idx] = sum_c alpha_c * V[:, c]
            # W[:, idx] = sum_c alpha_c * W[:, c]
            # with the SAME alpha_c for all three.
            # This is a coupled constraint. We solve it by least squares on all three.

            # Stack the three matrices and solve for common coefficients
            # We want alpha such that:
            # U_others @ alpha ≈ U_target
            # V_others @ alpha ≈ V_target
            # W_others @ alpha ≈ W_target

            # Use weighted least squares
            # Combine into a single system
            # [U_others; V_others; W_others] @ alpha ≈ [U_target; V_target; W_target]
            # But we need to weight them appropriately. Since all entries are O(1),
            # we can just stack them.

            A_stack = np.vstack([U_others, V_others, W_others])
            b_stack = np.vstack([U_target, V_target, W_target])

            try:
                alpha, residuals, rank, sv = np.linalg.lstsq(A_stack, b_stack, rcond=1e-10)
            except np.linalg.LinAlgError:
                new_active.append(idx)
                continue

            # Check residual
            residual = np.linalg.norm(A_stack @ alpha - b_stack)
            if residual > 1e-6:
                new_active.append(idx)
                continue

            # Now verify the full decomposition with this column removed
            # and the others adjusted. Actually, we just remove the column;
            # the remaining columns already span the needed space.
            # But we need to verify the tensor identity.

            U_test = U_cur[:, new_active + [idx]]  # temporarily keep idx
            # Actually, let's just test: remove idx, keep others
            U_test = U_cur[:, others]
            V_test = V_cur[:, others]
            W_test = W_cur[:, others]

            # Verify
            err = verify_decomp(U_test, V_test, W_test)
            if err > 1e-8:
                new_active.append(idx)
                continue

            # Successfully removed
            removed_any = True

        active = new_active
        if not removed_any:
            break

    # Build final matrices
    if len(active) < R_cur:
        U_final = U_cur[:, active]
        V_final = V_cur[:, active]
        W_final = W_cur[:, active]
    else:
        U_final = U_cur
        V_final = V_cur
        W_final = W_cur

    final_rank = U_final.shape[1]

    # Final verification
    final_err = verify_decomp(U_final, V_final, W_final)
    if final_err > 1e-8:
        # Fall back to original
        U_final = U
        V_final = V
        W_final = W
        final_rank = R

    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

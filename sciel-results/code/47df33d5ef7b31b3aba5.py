"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    from itertools import combinations

    _deadline = time.time() + 135
    n, m, p = DIMENSIONS  # 3, 4, 6

    U66 = [[0.0] * 66 for _ in range(n * m)]
    V66 = [[0.0] * 66 for _ in range(m * p)]
    W66 = [[0.0] * 66 for _ in range(p * n)]

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
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

        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1

        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1

        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1

        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1

        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1

        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1

        U66[A01][col] += 1; U66[A11][col] -= 1
        V66[B10][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1

    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)

    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)

    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)

    # Precompute index arrays for fast verification
    nmp = n * m * p
    idx_i = np.repeat(np.arange(n), m * p).reshape(-1)
    idx_j = np.tile(np.repeat(np.arange(m), p), n).reshape(-1)
    idx_k = np.tile(np.arange(p), n * m).reshape(-1)
    idx_A = idx_i * m + idx_j  # row index in U
    idx_B = idx_j * p + idx_k  # row index in V
    idx_C = idx_k * n + idx_i  # row index in W

    def verify_fast(U_mat, V_mat, W_mat):
        """Fast verification using precomputed indices."""
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        T = (U_mat[idx_A, :] * V_mat[idx_B, :] * W_mat[idx_C, :]).sum(axis=1)
        return np.allclose(T, np.ones(nmp), atol=1e-9)

    if not verify_fast(U, V, W):
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col2] = 1
                    V[j * p + k, col2] = 1
                    W[k * n + i, col2] = 1
                    col2 += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Phase 1: Nullspace-based column elimination (improved)
    M = np.zeros((nmp, 66))
    for l in range(66):
        M[:, l] = U[idx_A, l] * V[idx_B, l] * W[idx_C, l]

    eliminated = set()
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        null_dim = int(np.sum(s < tol))

        if null_dim > 0:
            nullspace = Vt[66 - null_dim:, :]

            # Sort null vectors by max entry magnitude for better elimination order
            null_order = np.argsort([-np.max(np.abs(nullspace[i])) for i in range(null_dim)])
            
            for ni in null_order:
                if time.time() > _deadline - 25:
                    break
                null_vec = nullspace[ni]
                candidates = [i for i in range(66) if i not in eliminated]
                if not candidates:
                    break
                
                # Try to eliminate the column with the largest |null_vec| component
                max_idx = max(candidates, key=lambda i: abs(null_vec[i]))
                if abs(null_vec[max_idx]) < 1e-10:
                    continue

                remaining = [i for i in range(66) if i != max_idx]
                if verify_fast(U[:, remaining], V[:, remaining], W[:, remaining]):
                    eliminated.add(max_idx)
    except Exception:
        pass

    if eliminated:
        remaining = [i for i in range(66) if i not in eliminated]
        U = U[:, remaining]
        V = V[:, remaining]
        W = W[:, remaining]

    # Phase 2: Aggressive iterative drop search with vectorized verification
    if time.time() < _deadline - 15:
        improved = True
        while improved and time.time() < _deadline - 15:
            improved = False
            r = U.shape[1]
            for l in range(r):
                if time.time() > _deadline - 15:
                    break
                mask = [i for i in range(r) if i != l]
                if verify_fast(U[:, mask], V[:, mask], W[:, mask]):
                    U = U[:, mask]
                    V = V[:, mask]
                    W = W[:, mask]
                    improved = True
                    break

    # Phase 3: Multi-column drop (try dropping 2 columns at once)
    if time.time() < _deadline - 10:
        r = U.shape[1]
        if r >= 2:
            found = True
            while found and time.time() < _deadline - 10:
                found = False
                r = U.shape[1]
                for l1 in range(r):
                    if time.time() > _deadline - 10:
                        break
                    for l2 in range(l1 + 1, r):
                        if time.time() > _deadline - 10:
                            break
                        mask = [i for i in range(r) if i != l1 and i != l2]
                        if verify_fast(U[:, mask], V[:, mask], W[:, mask]):
                            U = U[:, mask]
                            V = V[:, mask]
                            W = W[:, mask]
                            found = True
                            break
                    if found:
                        break

    # Phase 4: Try to find linear dependencies among remaining columns
    # If columns l1 and l2 have proportional U, V, W vectors, they may be mergeable
    if time.time() < _deadline - 5:
        r = U.shape[1]
        for l1 in range(r):
            if time.time() > _deadline - 5:
                break
            for l2 in range(l1 + 1, r):
                if time.time() > _deadline - 5:
                    break
                # Check if U[:,l1] and U[:,l2] are proportional
                u1, u2 = U[:, l1], U[:, l2]
                v1, v2 = V[:, l1], V[:, l2]
                w1, w2 = W[:, l1], W[:, l2]
                
                # Find scaling factor from U
                mask_u = np.abs(u1) > 1e-15
                if not np.any(mask_u):
                    continue
                scale_u = u2[mask_u] / u1[mask_u]
                if not np.allclose(scale_u, scale_u[0], atol=1e-10, rtol=1e-10):
                    continue
                scale = scale_u[0]
                
                # Check V and W
                mask_v = np.abs(v1) > 1e-15
                if np.any(mask_v) and not np.allclose(v2[mask_v], scale * v1[mask_v], atol=1e-10, rtol=1e-10):
                    continue
                mask_w = np.abs(w1) > 1e-15
                if np.any(mask_w) and not np.allclose(w2[mask_w], scale * w1[mask_w], atol=1e-10, rtol=1e-10):
                    continue
                
                # They are proportional! Try removing l2 (keep l1 which has "simpler" support)
                mask = [i for i in range(r) if i != l2]
                # Scale l1 by (1+scale) to absorb l2's contribution
                U_new = U[:, mask].copy()
                V_new = V[:, mask].copy()
                W_new = W[:, mask].copy()
                # l1 is now at some position in mask
                l1_new = mask.index(l1)
                U_new[:, l1_new] *= (1 + scale)
                V_new[:, l1_new] *= (1 + scale)
                W_new[:, l1_new] *= (1 + scale)
                
                if verify_fast(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    break
            else:
                continue
            break

    # Phase 5: Final single-column drop pass
    if time.time() < _deadline - 3:
        improved = True
        while improved and time.time() < _deadline - 3:
            improved = False
            r = U.shape[1]
            for l in range(r):
                if time.time() > _deadline - 3:
                    break
                mask = [i for i in range(r) if i != l]
                if verify_fast(U[:, mask], V[:, mask], W[:, mask]):
                    U = U[:, mask]
                    V = V[:, mask]
                    W = W[:, mask]
                    improved = True
                    break

    if not verify_fast(U, V, W):
        # Fallback to the original 66
        U = np.array(U66, dtype=float)
        V = np.array(V66, dtype=float)
        W = np.array(W66, dtype=float)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

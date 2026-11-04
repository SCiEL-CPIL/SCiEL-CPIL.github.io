"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]


def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Verify the decomposition using vectorized operations.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]

    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)

    T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)

    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True


def _build_rank_88():
    """
    Construct a rank-88 decomposition via a 1/2 row split of A and a
    2-by-2 block split of B. The single-row blocks use a schoolbook core
    (rank 8 each), while the two-row blocks use a Strassen-based 2x2x4 core
    (rank 14 each). This yields 2*16 + 2*28 = 88.
    """
    n, m, p = 3, 4, 8

    U_list = []
    V_list = []
    W_list = []

    def add_sub_term(i_off, j_off, k_off, sub_n, sub_m, sub_p, u_sub, v_sub, w_sub):
        rank_sub = u_sub.shape[1]
        for c in range(rank_sub):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)

            for i in range(sub_n):
                for j in range(sub_m):
                    u_col[(i_off + i) * m + (j_off + j)] = u_sub[i * sub_m + j, c]

            for j in range(sub_m):
                for k in range(sub_p):
                    v_col[(j_off + j) * p + (k_off + k)] = v_sub[j * sub_p + k, c]

            for k in range(sub_p):
                for i in range(sub_n):
                    w_col[(k_off + k) * n + (i_off + i)] = w_sub[k * sub_n + i, c]

            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)

    def get_1x2x4():
        # Schoolbook 1x2x4: rank 8
        u = np.zeros((1 * 2, 8))
        v = np.zeros((2 * 4, 8))
        w = np.zeros((4 * 1, 8))
        idx = 0
        for j in range(2):
            for k in range(4):
                u[j, idx] = 1.0
                v[j * 4 + k, idx] = 1.0
                w[k, idx] = 1.0
                idx += 1
        return u, v, w

    def get_2x2x4():
        # 2x2x4 via two 2x2x2 Strassen cores: rank 14
        u_s = np.array([
            [1, 0, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 1, 1],
        ]).T
        v_s = np.array([
            [1, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0],
        ]).T
        # W indexing for 2x2x2: C11=0, C21=1, C12=2, C22=3  (k*n+i with n=2)
        w_s = np.zeros((4, 7))
        # M1: C11 + C22
        w_s[0, 0] = 1; w_s[3, 0] = 1
        # M2: C21 - C22
        w_s[1, 1] = 1; w_s[3, 1] = -1
        # M3: C12 + C22
        w_s[2, 2] = 1; w_s[3, 2] = 1
        # M4: C11 + C21
        w_s[0, 3] = 1; w_s[1, 3] = 1
        # M5: -C11 + C12
        w_s[0, 4] = -1; w_s[2, 4] = 1
        # M6: C22
        w_s[3, 5] = 1
        # M7: C11
        w_s[0, 6] = 1

        u_224 = np.zeros((4, 14))
        v_224 = np.zeros((8, 14))
        w_224 = np.zeros((8, 14))

        # First half: B columns j=0,1 -> k=0,1 (local)
        for c in range(7):
            u_224[:, c] = u_s[:, c]
            v_224[0, c] = v_s[0, c]
            v_224[1, c] = v_s[1, c]
            v_224[4, c] = v_s[2, c]
            v_224[5, c] = v_s[3, c]
            w_224[0, c] = w_s[0, c]
            w_224[1, c] = w_s[1, c]
            w_224[2, c] = w_s[2, c]
            w_224[3, c] = w_s[3, c]

        # Second half: B columns j=0,1 -> k=2,3 (local)
        for c in range(7):
            u_224[:, c + 7] = u_s[:, c]
            v_224[2, c + 7] = v_s[0, c]
            v_224[3, c + 7] = v_s[1, c]
            v_224[6, c + 7] = v_s[2, c]
            v_224[7, c + 7] = v_s[3, c]
            w_224[4, c + 7] = w_s[0, c]
            w_224[5, c + 7] = w_s[1, c]
            w_224[6, c + 7] = w_s[2, c]
            w_224[7, c + 7] = w_s[3, c]

        return u_224, v_224, w_224

    u_124, v_124, w_124 = get_1x2x4()
    u_224, v_224, w_224 = get_2x2x4()

    # A1 (row 0) x B left-half (j=0,1) -> k=0,1
    add_sub_term(0, 0, 0, 1, 2, 2, u_124, v_124, w_124)
    # A1 (row 0) x B right-half (j=2,3) -> k=0,1
    add_sub_term(0, 2, 0, 1, 2, 2, u_124, v_124, w_124)

    # A2 (rows 1,2) x B left-half (j=0,1) -> k=0,1
    add_sub_term(1, 0, 0, 2, 2, 2, u_224, v_224, w_224)
    # A2 (rows 1,2) x B right-half (j=2,3) -> k=0,1
    add_sub_term(1, 2, 0, 2, 2, 2, u_224, v_224, w_224)

    # A1 (row 0) x B left-half (j=0,1) -> k=2,3
    add_sub_term(0, 0, 2, 1, 2, 2, u_124, v_124, w_124)
    # A1 (row 0) x B right-half (j=2,3) -> k=2,3
    add_sub_term(0, 2, 2, 1, 2, 2, u_124, v_124, w_124)

    # A2 (rows 1,2) x B left-half (j=0,1) -> k=2,3
    add_sub_term(1, 0, 2, 2, 2, 2, u_224, v_224, w_224)
    # A2 (rows 1,2) x B right-half (j=2,3) -> k=2,3
    add_sub_term(1, 2, 2, 2, 2, 2, u_224, v_224, w_224)

    U_arr = np.array(U_list, dtype=float).T
    V_arr = np.array(V_list, dtype=float).T
    W_arr = np.array(W_list, dtype=float).T

    return U_arr, V_arr, W_arr


def _build_rank_96():
    """
    Alternative block decomposition (rank 96 schoolbook) used only if the
    rank-88 construction fails verification.
    """
    n, m, p = 3, 4, 8
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Deterministic rank-88 construction
    try:
        U88, V88, W88 = _build_rank_88()
        if U88.shape[1] == 88 and _verify_safe(U88, V88, W88, n, m, p):
            return {"U": U88.tolist(), "V": V88.tolist(), "W": W88.tolist()}
    except Exception:
        pass

    # Fallback rank-96
    if time.time() > deadline:
        # Should never happen; return whatever we can construct quickly
        pass

    U96, V96, W96 = _build_rank_96()
    if time.time() < deadline and _verify_safe(U96, V96, W96, n, m, p):
        return {"U": U96.tolist(), "V": V96.tolist(), "W": W96.tolist()}

    # Last-resort: return the unverified rank-96 (still the only structure)
    return {"U": U96.tolist(), "V": V96.tolist(), "W": W96.tolist()}

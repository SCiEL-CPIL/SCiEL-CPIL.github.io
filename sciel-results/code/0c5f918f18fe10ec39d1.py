import time
import numpy as np

N, M, P = 3, 3, 8
SCHOOLBOOK = N * M * P  # 72

def _build_initial():
    """Build the 68-rank decomposition (Strassen blocks + schoolbook patches)."""
    rank = 68
    U = np.zeros((N * M, rank), dtype=np.float64)
    V = np.zeros((M * P, rank), dtype=np.float64)
    W = np.zeros((P * N, rank), dtype=np.float64)
    col = 0

    def add_strassen(u_base, v_base, w_base):
        nonlocal col
        a00, a01, a10, a11 = u_base
        b00, b01, b10, b11 = v_base
        c00, c01, c10, c11 = w_base
        # M1
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        W[c01, col] -= 1; W[c10, col] -= 1
        col += 1
        # M2
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        # M3
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M4
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M5
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        # M6
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        # M7
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c10, col] -= 1
        col += 1

    # Block 1: A_L * B1_T1
    add_strassen([0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 3, 4])
    # Block 2: A_L * B1_T2
    add_strassen([0, 1, 2, 3], [2, 3, 6, 7], [6, 7, 9, 10])
    # Block 3: A_R * B1_B
    u_r = [1, 3]
    v_b = [16, 17, 18, 19]
    w_c = [0, 1, 3, 4, 6, 7, 9, 10]
    for i, u_idx in enumerate(u_r):
        for j, v_idx in enumerate(v_b):
            w_idx = w_c[i * 4 + j]
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    # Block 4: A_L * B2_T1
    add_strassen([0, 1, 2, 3], [8, 9, 12, 13], [12, 13, 15, 16])
    # Block 5: A_L * B2_T2
    add_strassen([0, 1, 2, 3], [10, 11, 14, 15], [18, 19, 21, 22])
    # Block 6: A_R * B2_B
    u_r = [1, 3]
    v_b = [20, 21, 22, 23]
    w_c = [12, 13, 15, 16, 18, 19, 21, 22]
    for i, u_idx in enumerate(u_r):
        for j, v_idx in enumerate(v_b):
            w_idx = w_c[i * 4 + j]
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    # Block 7: A_bot * B (schoolbook)
    for k in range(8):
        w_idx = k * 3 + 2
        for i in range(3):
            u_idx = 6 + i
            v_idx = i * 8 + k
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    return U[:, :col], V[:, :col], W[:, :col]

def _verify(U, V, W, tol=1e-9):
    """Check that sum_c U[a,c] V[b,c] W[c] == 1 iff a,b,c correspond to a product."""
    try:
        rank = U.shape[1]
        for a in range(N * M):
            for b in range(M * P):
                acc = np.zeros(P * N)
                for c in range(rank):
                    u = U[a, c]
                    if u == 0:
                        continue
                    v = V[b, c]
                    if v == 0:
                        continue
                    acc += u * v * W[:, c]
                i, j = divmod(a, M)
                j2, k2 = divmod(b, P)
                for kk in range(P * N):
                    expected = 1.0 if (kk % N == i and kk // N == j2) else 0.0
                    if abs(acc[kk] - expected) > tol:
                        return False
        return True
    except Exception:
        return False

def _try_remove(U, V, W, col_idx, tol=1e-9):
    """Try removing column col_idx. Return (U', V', W') if valid, else None."""
    rank = U.shape[1]
    # Check: for every (a, b), the contribution of column col_idx must be zero
    # i.e., U[a, col_idx] * V[b, col_idx] == 0 for all a, b
    u_col = U[:, col_idx]
    v_col = V[:, col_idx]
    # The rank-1 outer product u_col * v_col must be zero
    if np.any(u_col != 0) and np.any(v_col != 0):
        # Check if the outer product is zero
        if np.any(u_col[:, None] * v_col[None, :] != 0):
            return None
    # Remove the column
    mask = np.ones(rank, dtype=bool)
    mask[col_idx] = False
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    return U_new, V_new, W_new

def construct():
    start = time.time()
    deadline = start + 130.0

    U, V, W = _build_initial()
    rank = U.shape[1]

    # Iteratively try to remove columns
    changed = True
    while changed and time.time() < deadline:
        changed = False
        current_rank = U.shape[1]
        for c in range(current_rank):
            if time.time() > deadline:
                break
            result = _try_remove(U, V, W, c)
            if result is not None:
                U, V, W = result
                changed = True
                break  # Restart from beginning since indices shifted

    # Final verification
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    U = np.zeros((N * M, SCHOOLBOOK), dtype=np.float64)
    V = np.zeros((M * P, SCHOOLBOOK), dtype=np.float64)
    W = np.zeros((P * N, SCHOOLBOOK), dtype=np.float64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

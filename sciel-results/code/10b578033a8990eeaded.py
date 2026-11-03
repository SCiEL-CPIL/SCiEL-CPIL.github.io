import numpy as np
import time

DIMENSIONS = [4, 5, 6]


def _build_T(n, m, p):
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _recon(U, V, W):
    r = U.shape[1]
    T_recon = np.zeros(U.shape[0] * V.shape[0] * W.shape[0]).reshape(U.shape[0], V.shape[0], W.shape[0])
    for t in range(r):
        uv = np.outer(U[:, t], V[:, t])
        T_recon += np.einsum('ab,c->abc', uv, W[:, t])
    return T_recon


def _err(U, V, W, T):
    return np.max(np.abs(_recon(U, V, W) - T))


def _schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2 (square, X[i,j], Y[j,k], C[k,i])."""
    r = 7
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))
    # entries: U[i*2+j], V[j*2+k], W[k*2+i]
    # m1 = (x11+x10)(y00+y11)
    U[3, 0] += 1; U[2, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    # m2 = x10(y11-y00)
    U[2, 1] += 1
    V[3, 1] += 1; V[0, 1] += -1
    W[1, 0] += 1; W[3, 0] += -1
    # m3 = (x00+x01)y11
    U[0, 2] += 1; U[1, 2] += 1
    V[3, 2] += 1
    W[1, 0] += 1; W[3, 0] += 1
    # m4 = x00(y10-y11)
    U[0, 3] += 1
    V[2, 3] += 1; V[3, 3] += -1
    W[0, 0] += 1; W[3, 0] += -1
    # m5 = (x00+x11)y00
    U[0, 4] += 1; U[3, 4] += 1
    V[0, 4] += 1
    W[0, 0] += -1; W[1, 0] += 1
    # m6 = (x10-x00)(y00+y01)
    U[2, 5] += 1; U[0, 5] += -1
    V[0, 5] += 1; V[1, 5] += 1
    W[0, 0] += 1; W[1, 0] += 1
    # m7 = (x01-x11)(y10+y11)
    U[1, 6] += 1; U[3, 6] += -1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 0] += 1; W[1, 0] += 1
    return U, V, W


def _build_strassen_4x4():
    """Rank-28 Strassen decomposition for 4x4 * 4x4 via 4 recursive 2x2 Strassen cores."""
    r = 28
    U = np.zeros((16, r))
    V = np.zeros((16, r))
    W = np.zeros((16, r))
    U2, V2, W2 = _strassen_2x2()
    for rb in range(2):
        for cb in range(2):
            base = (rb * 2 + cb) * 7
            for t2 in range(7):
                t = base + t2
                # X block: rows 2rb.., cols 2cb..
                for a in range(4):
                    if U2[a, t2] != 0:
                        i = a // 2; j = a % 2
                        U[(2 * rb + i) * 4 + (2 * cb + j), t] += U2[a, t2]
                # Y block: rows 2cb.., cols 2cb..
                for b in range(4):
                    if V2[b, t2] != 0:
                        j = b // 2; k = b % 2
                        V[(2 * cb + j) * 4 + (2 * cb + k), t] += V2[b, t2]
                # C block: rows 2rb.., cols 2cb..
                for c in range(4):
                    if W2[c, t2] != 0:
                        k = c // 2; i = c % 2
                        W[(2 * cb + k) * 4 + (2 * rb + i), t] += W2[c, t2]
    return U, V, W


def _build_laderman_3x3():
    """Rank-11 Laderman decomposition for 3x3 * 3x3 (square)."""
    r = 11
    U = np.zeros((9, r))
    V = np.zeros((9, r))
    W = np.zeros((9, r))
    # entries: U[i*3+j], V[j*3+k], W[k*3+i]
    # m1 = (x00+x11)(y00+y22)
    U[0, 0] += 1; U[4, 0] += 1
    V[0, 0] += 1; V[8, 0] += 1
    W[0, 0] += 1; W[8, 0] += 1
    # m2 = (x01+x12)(y02+y10)
    U[1, 1] += 1; U[5, 1] += 1
    V[2, 1] += 1; V[3, 1] += 1
    W[2, 0] += 1; W[3, 0] += 1
    # m3 = (x10+x21)(y11+y20)
    U[3, 2] += 1; U[7, 2] += 1
    V[4, 2] += 1; V[6, 2] += 1
    W[1, 1] += 1; W[6, 1] += 1
    # m4 = (x11+x22)(y12+y21)
    U[4, 3] += 1; U[8, 3] += 1
    V[5, 3] += 1; V[7, 3] += 1
    W[2, 1] += 1; W[7, 1] += 1
    # m5 = (x02+x20)(y01+y12)
    U[2, 4] += 1; U[6, 4] += 1
    V[1, 4] += 1; V[5, 4] += 1
    W[1, 2] += 1; W[5, 2] += 1
    # m6 = (x20+x01)(y10+y22)
    U[6, 5] += 1; U[1, 5] += 1
    V[3, 5] += 1; V[8, 5] += 1
    W[0, 2] += 1; W[8, 2] += 1
    # m7 = x22(y11-y00)
    U[8, 6] += 1
    V[4, 6] += 1; V[0, 6] += -1
    W[1, 1] += 1; W[0, 1] += -1
    # m8 = x01(y20-y11)
    U[1, 7] += 1
    V[6, 7] += 1; V[4, 7] += -1
    W[1, 2] += 1; W[1, 1] += -1
    # m9 = x12(y00-y22)
    U[5, 8] += 1
    V[0, 8] += 1; V[8, 8] += -1
    W[0, 0] += 1; W[8, 0] += -1
    # m10 = x21(y02-y10)
    U[7, 9] += 1
    V[2, 9] += 1; V[3, 9] += -1
    W[2, 1] += 1; W[3, 1] += -1
    # m11 = x10(y12-y21)
    U[3, 10] += 1
    V[5, 10] += 1; V[7, 10] += -1
    W[2, 1] += 1; W[7, 1] += -1
    return U, V, W


def _gc_decomposition(U, V, W):
    """Garbage-collect columns that contribute nothing (all-zero U or V)."""
    keep = []
    for t in range(U.shape[1]):
        if np.any(U[:, t] != 0) and np.any(V[:, t] != 0):
            keep.append(t)
    keep = np.array(keep, dtype=int)
    return U[:, keep], V[:, keep], W[:, keep]


def _build_candidate_strassen_core(n, m, p):
    """
    Use Strassen 4x4 for the top-left 4x4x4 block.
    Schoolbook for the remaining parts:
    - A[0:4, 4:5] * B[4:5, 0:6] -> C[0:6, 0:4] (24 terms)
    - A[0:4, 0:4] * B[0:4, 4:6] -> C[4:6, 0:4] (32 terms)
    Total: 28 + 24 + 32 = 84
    """
    nm, mp, pn = n * m, m * p, p * n
    r_total = 28 + 24 + 32
    U = np.zeros((nm, r_total))
    V = np.zeros((mp, r_total))
    W = np.zeros((pn, r_total))
    
    U4, V4, W4 = _build_strassen_4x4()
    col = 0
    
    # Part 1: Strassen 4x4 core: A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
    for t in range(28):
        for a in range(16):
            if U4[a, t] != 0:
                i = a // 4; j = a % 4
                U[i * m + j, col] += U4[a, t]
        for b in range(16):
            if V4[b, t] != 0:
                j = b // 4; k = b % 4
                V[j * p + k, col] += V4[b, t]
        for c in range(16):
            if W4[c, t] != 0:
                k = c // 4; i = c % 4
                W[k * n + i, col] += W4[c, t]
        col += 1
    
    # Part 2: A[0:4, 4:5] * B[4:5, 0:6] -> C[0:6, 0:4] (24 terms)
    for i in range(4):
        for k in range(6):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Part 3: A[0:4, 0:4] * B[0:4, 4:6] -> C[4:6, 0:4] (32 terms)
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    U, V, W = _gc_decomposition(U, V, W)
    return U, V, W


def _build_candidate_laderman_core(n, m, p):
    """
    Use Laderman 3x3 for the top-left 3x3x3 block.
    Schoolbook for the remaining parts:
    - A[3:4, 0:3] * B[0:3, 0:3] -> C[0:3, 3:4] (9 terms)
    - A[0:3, 3:4] * B[3:4, 0:3] -> C[0:3, 3:4] (9 terms)
    - A[0:4, 0:4] * B[0:4, 3:6] -> C[3:6, 0:4] (48 terms)
    - A[0:4, 4:5] * B[4:5, 0:6] -> C[0:6, 0:4] (24 terms)
    Total: 11 + 9 + 9 + 48 + 24 = 101
    """
    nm, mp, pn = n * m, m * p, p * n
    r_total = 11 + 9 + 9 + 48 + 24
    U = np.zeros((nm, r_total))
    V = np.zeros((mp, r_total))
    W = np.zeros((pn, r_total))
    
    U3, V3, W3 = _build_laderman_3x3()
    col = 0
    
    # Part 1: Laderman 3x3 core: A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3]
    for t in range(11):
        for a in range(9):
            if U3[a, t] != 0:
                i = a // 3; j = a % 3
                U[i * m + j, col] += U3[a, t]
        for b in range(9):
            if V3[b, t] != 0:
                j = b // 3; k = b % 3
                V[j * p + k, col] += V3[b, t]
        for c in range(9):
            if W3[c, t] != 0:
                k = c // 3; i = c % 3
                W[k * n + i, col] += W3[c, t]
        col += 1
    
    # Part 2: A[3:4, 0:3] * B[0:3, 0:3] -> C[0:3, 3:4] (9 terms)
    for i in range(3, 4):
        for j in range(3):
            for k in range(3):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Part 3: A[0:3, 3:4] * B[3:4, 0:3] -> C[0:3, 3:4] (9 terms)
    for i in range(3):
        for j in range(3, 4):
            for k in range(3):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Part 4: A[0:4, 0:4] * B[0:4, 3:6] -> C[3:6, 0:4] (48 terms)
    for i in range(4):
        for j in range(4):
            for k in range(3, 6):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Part 5: A[0:4, 4:5] * B[4:5, 0:6] -> C[0:6, 0:4] (24 terms)
    for i in range(4):
        for k in range(6):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    U, V, W = _gc_decomposition(U, V, W)
    return U, V, W


def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    T = _build_T(n, m, p)

    candidates = []

    # --- Candidate 0: schoolbook (rank 120) ---
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    candidates.append((U_sb, V_sb, W_sb))

    # --- Candidate 1: Strassen-4x4 core + schoolbook strips ---
    try:
        U_full, V_full, W_full = _build_candidate_strassen_core(n, m, p)
        candidates.append((U_full, V_full, W_full))
    except Exception:
        pass

    # --- Candidate 2: Laderman-3x3 core + schoolbook strips ---
    try:
        U_full, V_full, W_full = _build_candidate_laderman_core(n, m, p)
        candidates.append((U_full, V_full, W_full))
    except Exception:
        pass

    # --- Evaluate all candidates, keep best valid ---
    best_rank = None
    best_decomp = None
    for U, V, W in candidates:
        if time.time() > deadline - 20:
            break
        e = _err(U, V, W, T)
        if e < 1e-8:
            r = U.shape[1]
            if best_rank is None or r < best_rank:
                best_rank = r
                best_decomp = (U.copy(), V.copy(), W.copy())

    # --- Fallback to schoolbook if nothing valid ---
    if best_decomp is None:
        best_decomp = (U_sb.copy(), V_sb.copy(), W_sb.copy())

    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

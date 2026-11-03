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
    # (unused in output, kept for completeness)
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


def _embed_square(Ub, Vb, Wb, n, m, p, i0, j0, k0, nn, mm, pp):
    """Embed a square nn*nn*nn decomposition into global (n,m,p) at offset (i0,j0,k0)."""
    r = Ub.shape[1]
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for a in range(nn * nn):
        i = a // nn; j = a % nn
        U[(i0 + i) * m + (j0 + j), :] = Ub[a, :]
    for b in range(nn * nn):
        j = b // nn; k = b % nn
        V[(j0 + j) * p + (k0 + k), :] = Vb[b, :]
    for c in range(nn * nn):
        k = c // nn; i = c % nn
        W[(k0 + k) * n + (i0 + i), :] = Wc if False else W[c, :]
    return U, V, W


def _gc_decomposition(U, V, W):
    """Garbage-collect columns that contribute nothing (all-zero U or V)."""
    keep = []
    for t in range(U.shape[1]):
        if np.any(U[:, t] != 0) and np.any(V[:, t] != 0):
            keep.append(t)
    keep = np.array(keep, dtype=int)
    return U[:, keep], V[:, keep], W[:, keep]


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

    # --- Candidate 1: Strassen-4x4 core + schoolbook strips (rank 64) ---
    try:
        U4, V4, W4 = _build_strassen_4x4()
        r_total = 28 + 32 + 4
        U_full = np.zeros((nm, r_total))
        V_full = np.zeros((mp, r_total))
        W_full = np.zeros((pn, r_total))
        col = 0
        # Part 1: A1(4x4) * B1a(4x4) -> C11(4x4)
        for t in range(28):
            for a in range(16):
                if U4[a, t] != 0:
                    i = a // 4; j = a % 4
                    U_full[i * 5 + j, col] += U4[a, t]
            for b in range(16):
                if V4[b, t] != 0:
                    j = b // 4; k = b % 4
                    V_full[j * 6 + k, col] += V4[b, t]
            for c in range(16):
                if W4[c, t] != 0:
                    k = c // 4; i = c % 4
                    W_full[k * 4 + i, col] += W4[c, t]
            col += 1
        # Part 2: A1(4x4) * B1b(4x2) -> C(4x2)  [k in 4,5]
        for i in range(4):
            for j in range(4):
                for k in range(4, 6):
                    U_full[i * 5 + j, col] = 1
                    V_full[j * 6 + k, col] = 1
                    W_full[k * 4 + i, col] = 1
                    col += 1
        # Part 3: A2(4x1) * B2(1x6) -> C(4x6)
        for i in range(4):
            for k in range(6):
                U_full[i * 5 + 4, col] = 1
                V_full[4 * 6 + k, col] = 1
                W_full[k * 4 + i, col] = 1
                col += 1
        U_full, V_full, W_full = _gc_decomposition(U_full, V_full, W_full)
        candidates.append((U_full, V_full, W_full))
    except Exception:
        pass

    # --- Candidate 2: Laderman-3x3 core + Strassen-4x4 + strips ---
    try:
        U3, V3, W3 = _build_laderman_3x3()
        U4, V4, W4 = _build_strassen_4x4()
        r_total = 11 + 28 + 4 + 4 + 12
        U_full = np.zeros((nm, r_total))
        V_full = np.zeros((mp, r_total))
        W_full = np.zeros((pn, r_total))
        col = 0
        # Part 1: Laderman 3x3 core: A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3]
        for t in range(11):
            for a in range(9):
                if U3[a, t] != 0:
                    i = a // 3; j = a % 3
                    U_full[i * 5 + j, col] += U3[a, t]
            for b in range(9):
                if V3[b, t] != 0:
                    j = b // 3; k = b % 3
                    V_full[j * 6 + k, col] += V3[b, t]
            for c in range(9):
                if W3[c, t] != 0:
                    k = c // 3; i = c % 3
                    W_full[k * 4 + i, col] += W3[c, t]
            col += 1
        # Part 2: Strassen 4x4: A[3:7]?? -> A[3:7] invalid (n=4). Use A rows 3..3 only.
        # Instead: A[3:4, 0:4]?? Not 4x4. Use A[3:7] not possible.
        # Use A rows 0..3 col 3..6? col 3..6 -> B rows 3..6 invalid (m=5).
        # So skip the 4x4 here; use Strassen for A[0:4,0:4]*B[0:4,4:8]? B cols 4..7 invalid.
        # Use Strassen 4x4 on A[0:4,0:4] * B[0:4,0:4] is already core (overlaps Laderman). Skip.
        # Use Strassen on A[0:4, 0:4] * B[0:4, 0:4] NOT (overlap). 
        # Use Strassen 4x4 on A[0:4, 0:4] * B[0:4, 0:4] -> same as core. Skip.
        # Use Strassen 4x4 on A[0:4, 0:4] * B[0:4, 4:8] invalid.
        # Use Strassen 4x4 on A[0:4, 0:4] * B[0:4, 0:4] -> core.
        # Use Strassen 4x4 on A[0:4, 0:4] * B[0:4, 0:4] -> core.
        # So skip the 4x4 here.
        # Part 3: A[3:4, 0:3] * B[0:3, 0:3] -> C[0:3, 3:4]  (4 terms)
        for i in range(3, 4):
            for j in range(3):
                for k in range(3):
                    U_full[i * 5 + j, col] = 1
                    V_full[j * 6 + k, col] = 1
                    W_full[k * 4 + i, col] = 1
                    col += 1
        # Part 4: A[0:3, 3:4] * B[3:4, 0:3] -> C[0:3, 3:4]  (4 terms)
        for i in range(3):
            for j in range(3, 4):
                for k in range(3):
                    U_full[i * 5 + j, col] = 1
                    V_full[j * 6 + k, col] = 1
                    W_full[k * 4 + i, col] = 1
                    col += 1
        # Part 5: A[0:4, 0:4] * B[0:4, 3:6] -> C[3:6, 0:4]  (12 terms)
        for i in range(4):
            for j in range(4):
                for k in range(3, 6):
                    U_full[i * 5 + j, col] = 1
                    V_full[j * 6 + k, col] = 1
                    W_full[k * 4 + i, col] = 1
                    col += 1
        U_full, V_full, W_full = _gc_decomposition(U_full, V_full, W_full)
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

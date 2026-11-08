"""Rewritten solution: general block-Kronecker combinator for rank-minimal matrix-mult tensors."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True


def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _strassen_7():
    """Standard 7-product Strassen coefficients for 2x2x2 (0-indexed blocks)."""
    # M1: A=A11+A22, B=B11+B22, C11+=M1 C22+=M1
    # M2: A=A21+A22, B=B11,        C21+=M2 C22-=M2
    # M3: A=A11,       B=B12-B22,   C12+=M3 C22+=M3
    # M4: A=A22,       B=B21-B11,   C11+=M4 C21+=M4
    # M5: A=A11+A12,   B=B22,       C11-=M5 C12+=M5
    # M6: A=A21-A11,   B=B11+B12,   C22+=M6
    # M7: A=A12-A22,   B=B21+B22,   C11+=M7
    return [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1},          {(1,0):1,(1,1):-1}),
        ({(0,0):1},          {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1},          {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1},  {(1,1):1},          {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1},  {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1},  {(0,0):1}),
    ]


def _apply_strassen_step(U_base, V_base, W_base, n):
    """Expand an n x n x n decomposition to 2n x 2n x 2n via one Strassen step."""
    r = len(U_base[0])
    products = _strassen_7()
    R = 7 * r
    N = 2 * n
    U_new = [[0] * R for _ in range(4 * n * n)]
    V_new = [[0] * R for _ in range(4 * n * n)]
    W_new = [[0] * R for _ in range(4 * n * n)]
    for l, (S_A, S_B, C_coefs) in enumerate(products):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for c in range(r):
                        u_val = U_base[i * n + j][c]
                        v_val = V_base[j * n + k][c]
                        w_val = W_base[k * n + i][c]
                        col = l * r + c
                        for (I, J), ca in S_A.items():
                            U_new[(I * n + i) * N + (J * n + j)][col] += ca * u_val
                        for (J, K), cb in S_B.items():
                            V_new[(J * n + j) * N + (K * n + k)][col] += cb * v_val
                        for (I, K), cc in C_coefs.items():
                            W_new[(K * n + k) * N + (I * n + i)][col] += cc * w_val
    return U_new, V_new, W_new


def _place_blocks(U, V, W, n, m, p, placements):
    """
    Place a base (n,m,p) decomposition into a larger tensor.
    placements: list of (I,J,K,coef) where block (I,J,K) uses coef * base.
    Returns (U2,V2,W2) of shape (N*M, R), (M*P, R), (P*N, R) with R = len(placements)*r.
    Correct because the target tensor is the all-ones tensor.
    """
    r = len(U[0])
    R = len(placements) * r
    N = max(I for I, _, _, _ in placements) + 1
    P = max(K for _, _, K, _ in placements) + 1
    U2 = [[0] * R for _ in range(N * M)]
    V2 = [[0] * R for _ in range(M * P)]
    W2 = [[0] * R for _ in range(P * N)]
    for (I, J, K, coef), col0 in zip(placements, range(0, R, r)):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    ur = (I * n + i) * M + (J * m + j)
                    vr = (J * m + j) * P + (K * p + k)
                    wr = (K * p + k) * N + (I * n + i)
                    for c in range(r):
                        U2[ur][col0 + c] = coef * u_val[c]
                        V2[vr][col0 + c] = coef * v_val[c]
                        W2[wr][col0 + c] = coef * w_val[c]
    return U2, V2, W2


def _combine(U_a, V_a, W_a, U_b, V_b, W_b, m, p_a, p_b):
    """Concatenate two decompositions sharing (n,m) with output widths p_a and p_b."""
    n = U_a.shape[0] // m
    ra = len(U_a[0])
    rb = len(U_b[0])
    R = ra + rb
    P = p_a + p_b
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * P)]
    W = [[0] * R for _ in range(P * n)]
    for i in range(n * m):
        for c in range(ra):
            U[i][c] = U_a[i][c]
        for c in range(rb):
            U[i][ra + c] = U_b[i][c]
    for j in range(m):
        for k in range(p_a):
            for c in range(ra):
                V[j * P + k][c] = V_a[j * p_a + k][c]
        for k in range(p_b):
            for c in range(rb):
                V[j * P + (p_a + k)][ra + c] = V_b[j * p_b + k][c]
    for k in range(p_a):
        for i in range(n):
            for c in range(ra):
                W[k * n + i][c] = W_a[k * n + i][c]
    for k in range(p_b):
        for i in range(n):
            for c in range(rb):
                W[(p_a + k) * n + i][ra + c] = W_b[k * n + i][c]
    return U, V, W


def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 7

    # --- Build 4x4x4 Strassen (rank 49) ---
    U1, V1, W1 = [[1]], [[1]], [[1]]
    U2, V2, W2 = _apply_strassen_step(U1, V1, W1, 1)   # 2x2x2, rank 7
    U4, V4, W4 = _apply_strassen_step(U2, V2, W2, 2)   # 4x4x4, rank 49

    # --- Build 4x4x3 (rank 44) via block placement of 2x2 Strassen (7) + 2x2x1 schoolbook (4) ---
    U21, V21, W21 = _make_schoolbook(2, 2, 1)  # rank 4
    placements = [
        (0, 0, 0, 1, U2, V2, W2),
        (0, 1, 0, 1, U2, V2, W2),
        (0, 0, 1, 1, U21, V21, W21),
        (0, 1, 1, 1, U21, V21, W21),
        (1, 0, 0, 1, U2, V2, W2),
        (1, 1, 0, 1, U2, V2, W2),
        (1, 0, 1, 1, U21, V21, W21),
        (1, 1, 1, 1, U21, V21, W21),
    ]
    r = 44
    U443 = [[0] * r for _ in range(16)]
    V443 = [[0] * r for _ in range(12)]
    W443 = [[0] * r for _ in range(12)]
    for (I, J, K, coef, Us, Vs, Ws), col0 in zip(placements, range(0, r, 7 if Us is U2 else 4)):
        rr = len(Us[0])
        for i in range(2):
            for j in range(2):
                for k in range(len(Ws) // 2):  # p of base
                    ur = (I * 2 + i) * 4 + (J * 2 + j)
                    vr = (J * 2 + j) * 3 + (K * (2 if Us is U2 else 1) + k)
                    wr = (K * (2 if Us is U2 else 1) + k) * 4 + (I * 2 + i)
                    for c in range(rr):
                        U443[ur][col0 + c] = coef * Us[i * 2 + j][c]
                        V443[vr][col0 + c] = coef * Vs[j * (2 if Us is U2 else 1) + k][c]
                        W443[wr][col0 + c] = coef * Ws[k * 2 + i][c]

    # --- Combine 4x4x4 (p=4) and 4x4x3 (p=3) -> 4x4x7 (rank 93) ---
    U, V, W = _combine(U4, V4, W4, U443, V443, W443, m=4, p_a=4, p_b=3)

    if time.time() < deadline and _verify(U, V, W, 4, 4, 7):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 112
    sU, sV, sW = _make_schoolbook(4, 4, 7)
    return {"U": sU, "V": sV, "W": sW}

# EVOLVE-BLOCK-END

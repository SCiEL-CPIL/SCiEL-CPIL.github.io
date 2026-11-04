"""Matrix multiplication 3x4x6: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    import numpy as np
    _deadline = time.time() + 145
    n, m, p = 3, 4, 6

    # Build rank-66 block-structured decomposition
    R = 66
    U66 = [[0.0] * R for _ in range(n * m)]
    V66 = [[0.0] * R for _ in range(m * p)]
    W66 = [[0.0] * R for _ in range(p * n)]
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

        # M1 = (A00+A11)*(B00+B11)
        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1
        # M2 = (A10+A11)*B00
        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1
        # M3 = A00*(B01-B11)
        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1
        # M4 = A11*(B10-B00)
        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1
        # M5 = (A00+A01)*B11
        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1
        # M6 = (A10-A00)*(B00+B01)
        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1
        # M7 = (A01-A11)*(B10+B11)
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

    # Row 0: direct 1x2*2x2 blocks for 3 column groups
    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)
    # Rows 1-2: Strassen 2x2*2x2 blocks for 3 column groups
    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)

    assert col == 66, f"Expected 66, got {col}"

    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_rec = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        return np.allclose(T_rec, 1.0, atol=1e-9)

    # Verify rank-66 baseline
    if not verify(U, V, W, 66):
        # Fallback: schoolbook rank 72
        rs = n * m * p
        Us = np.zeros((n * m, rs)); Vs = np.zeros((m * p, rs)); Ws = np.zeros((p * n, rs))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j, c] = 1; Vs[j*p+k, c] = 1; Ws[k*n+i, c] = 1; c += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    # Precompute per-column tensor contributions for fast elimination test
    # M[:, l] = vec(T_l) where T_l[i,j,k] = U[i*m+j,l]*V[j*p+k,l]*W[k*n+i,l]
    M = np.zeros((n * m * p, 66))
    for l in range(66):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i*m*p + j*p + k, l] = U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]

    # Single-column elimination is impossible: sum_l M[:,l] = ones(72),
    # so removing column c gives sum = ones(72) - M[:,c] ≠ ones(72) unless M[:,c]=0.
    # Skip single-column tests entirely.

    # Fast pair elimination: M[:,c1] + M[:,c2] ≈ 0 implies sum of remaining = ones(72)
    if time.time() < _deadline - 10:
        for c1 in range(66):
            if time.time() > _deadline - 5:
                break
            for c2 in range(c1 + 1, 66):
                if np.allclose(M[:, c1] + M[:, c2], 0.0, atol=1e-9):
                    idx = [i for i in range(66) if i != c1 and i != c2]
                    U64 = U[:, idx]
                    V64 = V[:, idx]
                    W64 = W[:, idx]
                    if verify(U64, V64, W64, 64):
                        return {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}

    # Try triple eliminations with fast pre-check
    if time.time() < _deadline - 20:
        for c1 in range(66):
            if time.time() > _deadline - 10:
                break
            for c2 in range(c1 + 1, 66):
                for c3 in range(c2 + 1, 66):
                    if time.time() > _deadline - 5:
                        break
                    if np.allclose(M[:, c1] + M[:, c2] + M[:, c3], 0.0, atol=1e-9):
                        idx = [i for i in range(66) if i not in (c1, c2, c3)]
                        U63 = U[:, idx]
                        V63 = V[:, idx]
                        W63 = W[:, idx]
                        if verify(U63, V63, W63, 63):
                            return {"U": U63.tolist(), "V": V63.tolist(), "W": W63.tolist()}

    # Return rank-66
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

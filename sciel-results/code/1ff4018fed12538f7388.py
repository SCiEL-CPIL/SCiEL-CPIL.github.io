import time
import numpy as np

def construct():
    n, m, p = 3, 4, 6
    RANK = 44
    U = [[0] * RANK for _ in range(n * m)]
    V = [[0] * RANK for _ in range(m * p)]
    W = [[0] * RANK for _ in range(p * n)]
    col = 0

    def _iu(i, k):
        return i * m + k

    def _iv(k, j):
        return k * p + j

    def _iw(j, i):
        return j * n + i

    def _add_term(u_rows, v_rows, w_rows):
        nonlocal col
        for r in u_rows:
            U[r][col] += 1
        for r in v_rows:
            V[r][col] += 1
        for r in w_rows:
            W[r][col] += 1
        col += 1

    # Row 0 of C: A row 0 (1x2) times B blocks (2x2 or 2x4)
    # For each A col block and B col block, rank-4 schoolbook
    for ac in [0, 2]:
        for bc in [0, 2, 4]:
            a0 = _iu(0, ac)
            a1 = _iu(0, ac + 1)
            b00 = _iv(0, bc)
            b01 = _iv(0, bc + 1)
            b10 = _iv(1, bc)
            b11 = _iv(1, bc + 1)
            c0 = _iw(bc, 0)
            c1 = _iw(bc + 1, 0)
            _add_term([a0], [b00], [c0])
            _add_term([a1], [b10], [c0])
            _add_term([a0], [b01], [c1])
            _add_term([a1], [b11], [c1])

    # Rows 1-2 of C: Strassen with shared terms for B cols 0-1 and 2-5
    # A[1:3, 0:2] * B[0:2, 0:6]
    a00 = _iu(1, 0); a01 = _iu(1, 1)
    a10 = _iu(2, 0); a11 = _iu(2, 1)
    b00 = _iv(0, 0); b01 = _iv(0, 1)
    b10 = _iv(1, 0); b11 = _iv(1, 1)
    c00 = _iw(0, 1); c01 = _iw(1, 1)
    c10 = _iw(0, 2); c11 = _iw(1, 2)
    # Shared terms (contribute to both B col blocks 0-1 and 2-5)
    _add_term([a00, a11], [b00, b11], [c11, c11])  # M1: (A00+A11)(B00+B11) -> C11 in both
    _add_term([a10, a11], [b00], [c10, c00])         # M2: (A10+A11)B00 -> C10, C00
    _add_term([a00, a01], [b11], [c00, c01])         # M5: (A00+A01)B11 -> C00, C01
    _add_term([a10, a00], [b00, b01], [c11, c11])   # M6: (A10-A00)(B00+B01) -> C11 in both
    # B col 0-1 specific
    _add_term([a00], [b01, b11], [c01, c11])         # M3: A00(B01-B11) -> C01, C11
    _add_term([a11], [b10, b00], [c00, c10])         # M4: A11(B10-B00) -> C00, C10
    _add_term([a01, a11], [b10, b11], [c00, c00])   # M7: (A01-A11)(B10+B11) -> C00 in both
    # B col 2-5 specific
    b20 = _iv(0, 2); b21 = _iv(0, 3)
    b30 = _iv(1, 2); b31 = _iv(1, 3)
    c20 = _iw(2, 1); c21 = _iw(3, 1)
    c30 = _iw(2, 2); c31 = _iw(3, 2)
    _add_term([a00], [b21, b31], [c21, c31])         # M3': A00(B21-B31) -> C21, C31
    _add_term([a11], [b30, b20], [c20, c30])         # M4': A11(B30-B20) -> C20, C30
    _add_term([a01, a11], [b30, b31], [c20, c20])   # M7': (A01-A11)(B30+B31) -> C20 in both

    # A[1:3, 2:4] * B[2:4, 0:6]
    a20 = _iu(1, 2); a21 = _iu(1, 3)
    a30 = _iu(2, 2); a31 = _iu(2, 3)
    b40 = _iv(2, 0); b41 = _iv(2, 1)
    b50 = _iv(3, 0); b51 = _iv(3, 1)
    b60 = _iv(2, 2); b61 = _iv(2, 3)
    b70 = _iv(3, 2); b71 = _iv(3, 3)
    c40 = _iw(4, 1); c41 = _iw(5, 1)
    c50 = _iw(4, 2); c51 = _iw(5, 2)
    # Shared terms
    _add_term([a20, a31], [b40, b51], [c51, c51])  # M1: (A20+A31)(B40+B51) -> C51 in both
    _add_term([a30, a31], [b40], [c50, c40])         # M2: (A30+A31)B40 -> C50, C40
    _add_term([a20, a21], [b51], [c40, c41])         # M5: (A20+A21)B51 -> C40, C41
    _add_term([a30, a20], [b40, b41], [c51, c51])   # M6: (A30-A20)(B40+B41) -> C51 in both
    # B col 0-1 specific
    _add_term([a20], [b41, b51], [c41, c51])         # M3: A20(B41-B51) -> C41, C51
    _add_term([a31], [b50, b40], [c40, c50])         # M4: A31(B50-B40) -> C40, C50
    _add_term([a21, a31], [b50, b51], [c40, c40])   # M7: (A21-A31)(B50+B51) -> C40 in both
    # B col 2-5 specific
    _add_term([a20], [b61, b71], [c21, c31])         # M3': A20(B61-B71) -> C21, C31
    _add_term([a31], [b70, b60], [c20, c30])         # M4': A31(B70-B60) -> C20, C30
    _add_term([a21, a31], [b70, b71], [c20, c20])   # M7': (A21-A31)(B70+B71) -> C20 in both

    assert col == RANK, f"Expected {RANK} terms, got {col}"

    # Verification
    rng = np.random.RandomState(42)
    A_test = rng.randn(n, m)
    B_test = rng.randn(m, p)
    C_expected = A_test @ B_test

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    U_A = U_np.reshape(n, m, RANK)
    V_B = V_np.reshape(m, p, RANK)
    W_C = W_np.reshape(p, n, RANK)

    C_computed = np.einsum('irk,rjk,jik->ij', U_A, V_B, W_C)
    max_err = np.max(np.abs(C_computed - C_expected))
    assert max_err < 1e-9, f"Verification failed: max error = {max_err}"

    return {"U": U, "V": V, "W": W}

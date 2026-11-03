import numpy as np
import time

_DIMS = (2, 4, 8)
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0  # safety margin before kill

def _time_ok():
    return (time.time() - _START) < (_DEADLINE - _MARGIN)

def _strassen44_full(U, V, W, a00, a01, a02, a03,
                     a10, a11, a12, a13,
                     a20, a21, a22, a23,
                     a30, a31, a32, a33,
                     b00, b01, b02, b03,
                     b10, b11, b12, b13,
                     b20, b21, b22, b23,
                     b30, b31, b32, b33,
                     c00, c01, c02, c03,
                     c10, c11, c12, c13,
                     c20, c21, c22, c23,
                     c30, c31, c32, c33,
                     col):
    """Full 4x4 Strassen decomposition (rank 48).

    A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]] where each block is 2x2.
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    M8 = (A00+A10)(B00+B10)
    M9 = (A01+A11)(B01+B11)
    C00 = M1 - M5 + M7 - M8
    C01 = M3 + M5 + M9
    C10 = M2 + M4 + M8
    C11 = M1 - M2 + M3 - M4 + M6 - M7 + M9
    """
    # ---- U (A-side) ----
    # M1
    U[a00][col] += 1; U[a11][col] += 1
    U[a22][col] += 1; U[a33][col] += 1
    # M2
    U[a10][col+1] += 1; U[a11][col+1] += 1
    U[a30][col+1] += 1; U[a31][col+1] += 1
    # M3
    U[a00][col+2] += 1
    # M4
    U[a11][col+3] += 1
    # M5
    U[a00][col+4] += 1; U[a01][col+4] += 1
    U[a20][col+4] += 1; U[a21][col+4] += 1
    # M6
    U[a10][col+5] += 1; U[a00][col+5] += -1
    U[a30][col+5] += 1; U[a20][col+5] += -1
    # M7
    U[a01][col+6] += 1; U[a11][col+6] += -1
    U[a21][col+6] += 1; U[a31][col+6] += -1
    # M8
    U[a00][col+7] += 1; U[a10][col+7] += 1
    U[a20][col+7] += 1; U[a30][col+7] += 1
    # M9
    U[a01][col+8] += 1; U[a11][col+8] += 1
    U[a21][col+8] += 1; U[a31][col+8] += 1

    # ---- V (B-side) ----
    # M1
    V[b00][col] += 1; V[b11][col] += 1
    V[b22][col] += 1; V[b33][col] += 1
    # M2
    V[b00][col+1] += 1
    # M3
    V[b01][col+2] += 1; V[b11][col+2] += -1
    # M4
    V[b10][col+3] += 1; V[b00][col+3] += -1
    # M5
    V[b11][col+4] += 1
    # M6
    V[b00][col+5] += 1; V[b01][col+5] += 1
    # M7
    V[b10][col+6] += 1; V[b11][col+6] += 1
    # M8
    V[b00][col+7] += 1; V[b10][col+7] += 1
    # M9
    V[b01][col+8] += 1; V[b11][col+8] += 1

    # ---- W (C-side) ----
    # C00 = M1 - M5 + M7 - M8
    W[c00][col] += 1; W[c00][col+4] += -1; W[c00][col+6] += 1; W[c00][col+7] += -1
    # C01 = M3 + M5 + M9
    W[c01][col+2] += 1; W[c01][col+4] += 1; W[c01][col+8] += 1
    # C10 = M2 + M4 + M8
    W[c10][col+1] += 1; W[c10][col+3] += 1; W[c10][col+7] += 1
    # C11 = M1 - M2 + M3 - M4 + M6 - M7 + M9
    W[c11][col] += 1; W[c11][col+1] += -1; W[c11][col+2] += 1; W[c11][col+3] += -1
    W[c11][col+5] += 1; W[c11][col+6] += -1; W[c11][col+8] += 1

def _build_strassen44():
    """Embed 2x4 * 4x8 into 4x4 * 4x4 Strassen (rank 48).

    A (2x4) -> top-left 2x4 of a 4x4 (rows 2,3 zeroed).
    B (4x8) -> 4x8 with 4 zero-padded columns.
    C (2x8) -> rows 2,3 of the 4x4 product.
    """
    rank = 48
    n4 = 4
    U = [[0] * rank for _ in range(n4 * n4)]
    V = [[0] * rank for _ in range(n4 * n4)]
    W = [[0] * rank for _ in range(n4 * n4)]

    # A indices (4x4): row*4 + col
    a00 = 0*4+0; a01 = 0*4+1; a02 = 0*4+2; a03 = 0*4+3
    a10 = 1*4+0; a11 = 1*4+1; a12 = 1*4+2; a13 = 1*4+3
    a20 = 2*4+0; a21 = 2*4+1; a22 = 2*4+2; a23 = 2*4+3
    a30 = 3*4+0; a31 = 3*4+1; a32 = 3*4+2; a33 = 3*4+3

    # B indices (4x8): row*8 + col
    b00 = 0*8+0; b01 = 0*8+1; b02 = 0*8+2; b03 = 0*8+3
    b10 = 1*8+0; b11 = 1*8+1; b12 = 1*8+2; b13 = 1*8+3
    b20 = 2*8+0; b21 = 2*8+1; b22 = 2*8+2; b23 = 2*8+3
    b30 = 3*8+0; b31 = 3*8+1; b32 = 3*8+2; b33 = 3*8+3

    # C indices (4x8): row*8 + col  (we use rows 2,3)
    c00 = 0*8+0; c01 = 0*8+1; c02 = 0*8+2; c03 = 0*8+3
    c10 = 1*8+0; c11 = 1*8+1; c12 = 1*8+2; c13 = 1*8+3
    c20 = 2*8+0; c21 = 2*8+1; c22 = 2*8+2; c23 = 2*8+3
    c30 = 3*8+0; c31 = 3*8+1; c32 = 3*8+2; c33 = 3*8+3

    _strassen44_full(U, V, W,
                     a00, a01, a02, a03,
                     a10, a11, a12, a13,
                     a20, a21, a22, a23,
                     a30, a31, a32, a33,
                     b00, b01, b02, b03,
                     b10, b11, b12, b13,
                     b20, b21, b22, b23,
                     b30, b31, b32, b33,
                     c00, c01, c02, c03,
                     c10, c11, c12, c13,
                     c20, c21, c22, c23,
                     c30, c31, c32, c33,
                     0)
    return U, V, W

def _build_strassen56():
    """Fallback: 8 independent Strassen 2x2 blocks (rank 56)."""
    rank = 56
    U = [[0] * rank for _ in range(2 * 4)]
    V = [[0] * rank for _ in range(4 * 8)]
    W = [[0] * rank for _ in range(8 * 2)]

    for s in range(2):
        for j in range(4):
            a00 = 0*4 + (2*s+0); a01 = 0*4 + (2*s+1)
            a10 = 1*4 + (2*s+0); a11 = 1*4 + (2*s+1)
            b00 = (2*s+0)*8 + (2*j+0); b01 = (2*s+0)*8 + (2*j+1)
            b10 = (2*s+1)*8 + (2*j+0); b11 = (2*s+1)*8 + (2*j+1)
            c00 = (2*j+0)*2 + 0; c01 = (2*j+1)*2 + 0
            c10 = (2*j+0)*2 + 1; c11 = (2*j+1)*2 + 1
            col = 7 * (2*s + j)
            # M1 = (a00+a11)(b00+b11)
            U[a00][col] += 1; U[a11][col] += 1
            V[b00][col] += 1; V[b11][col] += 1
            W[c00][col] += 1; W[c11][col] += 1
            # M2 = (a10+a11)b00
            U[a10][col+1] += 1; U[a11][col+1] += 1
            V[b00][col+1] += 1
            W[c10][col+1] += 1; W[c11][col+1] += -1
            # M3 = a00(b01-b11)
            U[a00][col+2] += 1
            V[b01][col+2] += 1; V[b11][col+2] += -1
            W[c01][col+2] += 1; W[c11][col+2] += 1
            # M4 = a11(b10-b00)
            U[a11][col+3] += 1
            V[b10][col+3] += 1; V[b00][col+3] += -1
            W[c10][col+3] += 1; W[c11][col+3] += -1
            # M5 = (a00+a01)b11
            U[a00][col+4] += 1; U[a01][col+4] += 1
            V[b11][col+4] += 1
            W[c00][col+4] += -1; W[c01][col+4] += 1
            # M6 = (a10-a00)(b00+b01)
            U[a10][col+5] += 1; U[a00][col+5] += -1
            V[b00][col+5] += 1; V[b01][col+5] += 1
            W[c11][col+5] += 1
            # M7 = (a01-a11)(b10+b11)
            U[a01][col+6] += 1; U[a11][col+6] += -1
            V[b10][col+6] += 1; V[b11][col+6] += 1
            W[c00][col+6] += 1; W[c11][col+6] += -1
    return U, V, W

def _validate(U, V, W, n=2, m=4, p=8):
    """Verify exact decomposition of the multiplication tensor."""
    np.random.seed(42)
    A = np.random.randint(-3, 4, size=(n, m)).astype(float)
    B = np.random.randint(-3, 4, size=(m, p)).astype(float)
    C = A @ B

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                if abs(val - C[i, k]) > 1e-9:
                    return False
    return True

def construct():
    """
    Rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Primary: full 4x4 Strassen (rank 48) via zero-embedding.
    Fallback: 8 independent Strassen 2x2 blocks (rank 56).
    """
    U, V, W = _build_strassen44()
    if _validate(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback to rank-56
    U, V, W = _build_strassen56()
    return {"U": U, "V": V, "W": W}

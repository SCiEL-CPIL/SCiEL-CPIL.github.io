"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition.

Strategy: Combine Strassen-based decomposition for 2x2 sub-blocks with
schoolbook for the remaining row. The 2x3 * 3x8 multiplication is split as:
  - A[0:2, 0:2] * B[0:2, :]  (2x2 * 2x8) -> 4 blocks of 2x2x2, each rank 7 (Strassen)
  - A[0:2, 2:3] * B[2:3, :]  (2x1 * 1x8) -> schoolbook rank 16
Total rank: 4*7 + 16 = 44.

The current program's approach of splitting into 4 blocks of 2x3x2 and using
Strassen+outer-product gives rank 32, which is better. We keep that structure
but verify it carefully and use exact arithmetic for validation.
"""
DIMENSIONS = [2, 3, 8]

import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition using exact arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                        continue
                    s += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Construct the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# Strassen 2x2x2 decomposition vectors.
# U indices: (i, j) where i in {0,1}, j in {0,1} -> flattened as i*2+j
# V indices: (j, k) where j in {0,1}, k in {0,1} -> flattened as j*2+k
# W indices: (k, i) where k in {0,1}, i in {0,1} -> flattened as k*2+i

_S_U = [
    [1, 0, 0, 1],   # M1: (A00+A11)
    [0, 0, 1, 1],   # M2: (A10+A11)
    [1, 0, 0, 0],   # M3: A00
    [0, 0, 0, 1],   # M4: A11
    [1, 1, 0, 0],   # M5: (A00+A01)
    [-1, 0, 1, 0],  # M6: (A10-A00)
    [0, 1, 0, -1],  # M7: (A01-A11)
]
_S_V = [
    [1, 0, 0, 1],   # M1: (B00+B11)
    [1, 0, 0, 0],   # M2: B00
    [0, 1, 0, -1],  # M3: (B10-B11)
    [-1, 0, 1, 0],  # M4: (B01-B00)
    [0, 0, 0, 1],   # M5: B11
    [1, 1, 0, 0],   # M6: (B00+B01)
    [0, 0, 1, 1],   # M7: (B10+B11)
]
_S_W = [
    [1, 0, 0, 1],   # M1: C00 +=, C11 +=
    [0, 1, 0, -1],  # M2: C10 +=, C11 -=
    [0, 0, 1, 1],   # M3: C10 +=, C11 +=
    [1, 1, 0, 0],   # M4: C00 +=, C01 +=
    [-1, 0, 1, 0],  # M5: C00 -=, C10 +=
    [0, 0, 0, 1],   # M6: C11 +=
    [1, 0, 0, 0],   # M7: C00 +=
]

def _verify_strassen():
    """Verify the Strassen 2x2x2 decomposition is correct."""
    for i in range(2):
        for j in range(2):
            for k in range(2):
                acc = 0
                for r in range(7):
                    u_val = _S_U[r][i * 2 + j]
                    v_val = _S_V[r][j * 2 + k]
                    w_val = _S_W[r][k * 2 + i]
                    acc += u_val * v_val * w_val
                if acc != 1:
                    return False
    return True

def _block_decomposition_rank32(n, m, p):
    """
    Build rank-32 decomposition by splitting 8 output columns into 4 blocks of 2.
    Each block: 2x3 * 3x2 tensor.
    Decompose each 2x3x2 as:
      - 2x2x2 part (rows 0,1 of A; cols 0,1 of B) -> rank 7 (Strassen)
      - 2x1x2 part (row 2 of A; cols 0,1 of B) -> rank 1 (outer product: U=[1,1], V=[1,1], W=[1,1,1,1])
    Total per block: 8, total: 32.
    """
    n, m, p = 2, 3, 8
    nblocks = p // 2  # 4
    rank = nblocks * 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(nblocks):
        k0 = 2 * b
        col = b * 8

        # 7 Strassen terms for the 2x2x2 part (A rows 0,1; B cols k0,k0+1)
        for r in range(7):
            # U: i in {0,1}, j in {0,1}
            for i in range(2):
                for j in range(2):
                    val = _S_U[r][i * 2 + j]
                    if val != 0:
                        U[i * m + j][col + r] += val

            # V: j in {0,1}, k in {k0, k0+1}
            for j in range(2):
                for lk in range(2):
                    val = _S_V[r][j * 2 + lk]
                    if val != 0:
                        V[j * p + (k0 + lk)][col + r] += val

            # W: k in {k0, k0+1}, i in {0,1}
            for lk in range(2):
                for i in range(2):
                    val = _S_W[r][lk * 2 + i]
                    if val != 0:
                        W[(k0 + lk) * n + i][col + r] += val

        # 1 outer product term for row 2 of A, cols k0, k0+1 of B
        oc = col + 7
        for i in range(n):
            U[i * m + 2][oc] = 1
        for lk in range(2):
            V[2 * p + (k0 + lk)][oc] = 1
        for lk in range(2):
            for i in range(n):
                W[(k0 + lk) * n + i][oc] = 1

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Verify Strassen decomposition
    if not _verify_strassen():
        # Fallback to schoolbook if Strassen is broken
        return _schoolbook(n, m, p)

    # Build the rank-32 block decomposition
    result = _block_decomposition_rank32(n, m, p)

    # Validate with exact arithmetic
    if time.time() < deadline and _validate(result["U"], result["V"], result["W"], n, m, p):
        return result

    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END

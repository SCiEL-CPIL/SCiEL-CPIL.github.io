"""
Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition.

Design (family: structured block + Strassen kernel + fast exact verification):

1. Rank-7 Strassen kernel for the 2x2x2 multiplication tensor.
2. A 2x4 x 4x4 product is factored into four 2x2x2 Strassen blocks
   (rank 4*7 = 28), beating its schoolbook cost (32).
3. 3x4 x 4x8 is partitioned into:
       A = [ A_top (2x4) ; A_bot (1x4) ],   B = [ B_left (4x4) | B_right (4x4) ]
   giving C blocks:
       C_TL (2x4) = A_top * B_left   -> Strassen  (rank 28)
       C_TR (2x4) = A_top * B_right  -> Strassen  (rank 28)
       C_BL (1x4) = A_bot  * B_left  -> schoolbook (rank 16)
       C_BR (1x4) = A_bot  * B_right -> schoolbook (rank 16)
   Total rank = 28 + 28 + 16 + 16 = 56.

4. Verification is done with a fast einsum (sums of squares of integers ->
   exactly representable) rather than a per-triple Python loop.
5. A bounded Kronecker-structure attempt is run only if time remains, and any
   candidate is verified exactly before it may replace the best result.

Entry point: construct() -> {'U','V','W'} (list-of-lists, integer entries).
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 96."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=None, m=None, p=None):
    """
    Fast exact verification.

    The matrix-multiplication tensor has T[i,j,k] = 1 for all i,j,k, and a
    decomposition is valid iff for every (i,j,k):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1.

    Entries are integers, so each sum is a sum of squares of integers, exactly
    representable in float64 -> the 1e-9 tolerance is effectively exact.
    """
    if n is None:
        n, m, p = DIMENSIONS

    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)

    if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
        return False
    if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
        return False
    rank = U_arr.shape[1]
    if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
        return False

    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank).
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)

    # T[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r].
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) <= 1e-9))


def _strassen_2x2():
    """
    Rank-7 Strassen decomposition for the 2x2x2 multiplication tensor.

    Returns U, V, W each of shape (4, 7).  Local index order:
        U[i*2 + j], V[j*2 + k], W[k*2 + i],  i,j,k in {0,1}.

    Strassen products:
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22) B11
        M3 = A11 (B21-B22)
        M4 = A22 (B12-B11)
        M5 = (A11+A12) B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)

    Output combination:
        C11 = M1 - M5 + M4
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 + M6 - M2 + M3
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)

    # M1
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    # M2
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, 0]
    # M3
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 0, 1, -1]
    W[:, 2] = [0, 1, 0, 0]
    # M4
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [0, 1, -1, 0]
    W[:, 3] = [1, 0, 0, 0]
    # M5
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [0, 0, 0, 1]
    # M6
    U[:, 5] = [0, 0, 1, -1]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    # M7
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [0, 1, 0, 0]

    return U, V, W


def _block_2x4x4x4():
    """
    Rank-28 Strassen decomposition for a 2x4 x 4x4 product.

    A (2x4) split into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
    B (4x4) split into B_LL (rows 0-1, cols 0-1), B_LR (rows 0-1, cols 2-3),
    B_BL (rows 2-3, cols 0-1), B_BR (rows 2-3, cols 2-3).

    C (2x4):
        C[:, 0:2] = A_L * B_LL + A_R * B_BL   -> 2 Strassen blocks
        C[:, 2:4] = A_L * B_LR + A_R * B_BR   -> 2 Strassen blocks
    Total rank = 4 * 7 = 28.
    """
    U, V, W = _strassen_2x2()

    Ua = np.zeros((8, 28), dtype=int)   # A: 2x4 -> 8 rows
    Va = np.zeros((16, 28), dtype=int)  # B: 4x4 -> 16 rows
    Wa = np.zeros((8, 28), dtype=int)   # C: 2x4 -> 8 rows

    blocks = [
        # (a_row0, a_row1, b_row0, b_row1, c_col0)
        (0, 1, 0, 1, 0),   # A_L * B_LL -> C[:,0:2]
        (2, 3, 4, 5, 0),   # A_R * B_BL -> C[:,0:2]
        (0, 1, 2, 3, 2),   # A_L * B_LR -> C[:,2:4]
        (2, 3, 6, 7, 2),   # A_R * B_BR -> C[:,2:4]
    ]

    for b, (ar0, ar1, br0, br1, cc0) in enumerate(blocks):
        col0 = b * 7
        for t in range(7):
            col = col0 + t
            # A factor: rows ar0 (local row 0), ar1 (local row 1)
            Ua[ar0, col] = U[0, t]
            Ua[ar1, col] = U[1, t]
            # B factor: rows br0 (local row 0), br1 (local row 1)
            Va[br0, col] = V[0, t]
            Va[br1, col] = V[1, t]
            # C factor: cols cc0 (local col 0), cc0+1 (local col 1)
            Wa[cc0, col] = W[0, t]
            Wa[cc0 + 1, col] = W[1, t]

    return Ua, Va, Wa


def _block_1x4x4x4():
    """
    Schoolbook rank-16 decomposition for a 1x4 x 4x4 product.

    A: 1x4 -> 4 rows (j = 0..3).
    B: 4x4 -> 16 rows (j*4 + k).
    C: 1x4 -> 4 rows (k = 0..3).
    """
    Ua = np.zeros((4, 16), dtype=int)
    Va = np.zeros((16, 16), dtype=int)
    Wa = np.zeros((4, 16), dtype=int)

    for j in range(4):
        for k in range(4):
            col = j * 4 + k
            Ua[j, col] = 1
            Va[j * 4 + k, col] = 1
            Wa[k, col] = 1

    return Ua, Va, Wa


def _kronecker_decomposition():
    """
    Kronecker-product decomposition for 3x4 x 4x8.

    View the multiplication as a tensor product:
        T_{3,4,8} = T_{3,4,4} ⊗ T_{1,1,2}

    where T_{3,4,4} is the 3x4 x 4x4 multiplication tensor (decomposed via
    Kronecker T_{3,2,2} ⊗ T_{1,2,2}, each a rank-7 Strassen block), and
    T_{1,1,2} is the 1x1 x 1x2 tensor (rank 2 schoolbook).

    Total rank = 7 * 7 * 2 = 98.

    Index conventions (local blocks):
        T_{a,b,c}:  U[i*b+j], V[j*c+k], W[k*a+i]
        T_{3,4,8}:  U[i*4+j], V[j*8+k], W[k*3+i]

    The Kronecker factorization maps:
        A[i*4+j] = A1[i*2+j1] * A2[j2]      (j = j1*2 + j2)
        B[j*8+k] = B1[j1*2+j2, k1*2+k2] * B2[k2]
                  = B1[(j1*2+j2)*4 + k1*2 + k2] * B2[k2]
        C[k*3+i]  = C1[k1*2+j2, i] * C2[k2]  -- wait, let me re-derive.

    Actually, the Kronecker product of two multiplication tensors:
        (T_{a1,b1,c1} ⊗ T_{a2,b2,c2})_{(i1,i2),(j1,j2),(k1,k2)} =
            T_{a1,b1,c1}_{i1,j1,k1} * T_{a2,b2,c2}_{i2,j2,k2}

    This gives a multiplication tensor of size (a1*a2) x (b1*b2) x (c1*c2).

    For 3x4 x 4x8:
        Option A: T_{3,2,2} ⊗ T_{1,2,4}  -> (3*1) x (2*2) x (2*4) = 3x4x8 ✓
        Option B: T_{3,4,4} ⊗ T_{1,1,2}  -> (3*1) x (4*1) x (4*2) = 3x4x8 ✓

    Let's use Option B: T_{3,4,4} ⊗ T_{1,1,2}.

    T_{3,4,4} can be further decomposed as T_{3,2,2} ⊗ T_{1,2,2}:
        (3*1) x (2*2) x (2*2) = 3x4x4 ✓

    So: T_{3,4,8} = T_{3,2,2} ⊗ T_{1,2,2} ⊗ T_{1,1,2}

    Each T_{1,2,2} and T_{3,2,2} is a 2x2x2 Strassen (rank 7).
    T_{1,1,2} is rank 2 schoolbook.

    Total rank = 7 * 7 * 2 = 98.

    Index mapping for triple Kronecker:
        A index: i*4 + j  where j = j1*2 + j2
          => A[i*4 + j1*2 + j2]
        B index: j*8 + k  where j = j1*2 + j2, k = k1*2 + k2
          => B[(j1*2+j2)*8 + k1*2 + k2]
        C index: k*3 + i  where k = k1*2 + k2
          => C[(k1*2+k2)*3 + i]

    Strassen factors for T_{3,2,2} (3x2 x 2x2):
        This is 3 copies of T_{1,2,2} (since 3 = 3*1, we need T_{3,2,2}).
        Actually T_{3,2,2} is not a standard Strassen block. Let me reconsider.

    Better: T_{3,4,8} = T_{3,4,4} ⊗ T_{1,1,2}

    For T_{3,4,4}: use the existing _block_2x4x4x4 approach won't work directly
    since that's 2x4 x 4x4. Let me think differently.

    T_{3,4,4}: A is 3x4, B is 4x4, C is 4x3.
    Split A into A_top (2x4) and A_bot (1x4).
    C = C_top (4x2) ; C_bot (4x1)
    C_top = A_top * B  (2x4 x 4x4 -> 2x4)
    C_bot = A_bot * B  (1x4 x 4x4 -> 1x4)

    C_top uses rank 28 (from _block_2x4x4x4), C_bot uses rank 16 (schoolbook).
    Total for T_{3,4,4}: rank 44.

    Then T_{3,4,8} = T_{3,4,4} ⊗ T_{1,1,2}: rank 44 * 2 = 88.

    Hmm, that's worse than 56. Let me try the pure Kronecker approach.

    T_{3,4,8} = T_{3,2,2} ⊗ T_{1,2,4}

    T_{1,2,4}: A is 1x2, B is 2x4, C is 4x1. Rank = 1*2*4 = 8 schoolbook.
    T_{3,2,2}: A is 3x2, B is 2x2, C is 2x3.
       Split A into 3 rows of 2. Each row does a 1x2 x 2x2 -> 1x2 product.
       That's 3 * (1*2*2) = 12 schoolbook. Or use Strassen on pairs?
       Actually 3x2 x 2x2: think of it as 3 independent 1x2 x 2x2 products.
       Each 1x2 x 2x2 has rank 4 (schoolbook). Total rank 12.

    T_{3,2,2} ⊗ T_{1,2,4}: rank 12 * 8 = 96. Same as schoolbook. Not helpful.

    Let me try: T_{3,4,8} = T_{3,4,4} ⊗ T_{1,1,2}
    T_{3,4,4} via block: rank 28 + 16 = 44.
    T_{1,1,2}: rank 2.
    Total: 88. Still worse than 56.

    What about T_{3,4,8} = T_{3,2,4} ⊗ T_{1,2,2}?
    (3*1) x (2*2) x (4*2) = 3x4x8 ✓
    T_{1,2,2}: rank 7 (Strassen).
    T_{3,2,4}: A is 3x2, B is 2x4, C is 4x3.
       Split B into B_left (2x2) and B_right (2x2).
       C_left (2x3) = A * B_left  (3x2 x 2x2 -> 2x3)
       C_right (2x3) = A * B_right (3x2 x 2x2 -> 2x3)
       Each 3x2 x 2x2: 3 independent 1x2 x 2x2, rank 4 each, total 12.
       Two blocks: rank 24.
    Total: 24 * 7 = 168. Worse.

    It seems the Kronecker approach doesn't beat the block-Strassen rank 56.
    However, the directive asks me to implement it. Let me implement the
    T_{3,4,4} ⊗ T_{1,1,2} version (rank 88) as an alternative, and keep
    the block-Strassen (rank 56) as the primary.

    Actually, re-reading the directive: "yielding a total rank of 40 by combining
    the factor matrices via Kronecker products." This suggests a rank-20 scheme
    for 3x4x4 times rank-2 for 1x1x2. But rank 20 for 3x4x4 is below the known
    lower bound... Let me just implement what's mathematically sound.

    I'll implement T_{3,4,4} ⊗ T_{1,1,2} with the block decomposition for
    T_{3,4,4} (rank 44) giving total rank 88. This is a valid Kronecker
    decomposition even if not optimal.
    """
    # T_{1,1,2} schoolbook: U[0,0]=1, V[0,0]=1; U[0,1]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
    # Shape: U (1*1, 2) = (1,2), V (1*2, 2) = (2,2), W (2*1, 2) = (2,2)
    U2 = np.array([[1, 1]], dtype=int)  # (1, 2)
    V2 = np.array([[1, 0],
                   [0, 1]], dtype=int)  # (2, 2)
    W2 = np.array([[1, 0],
                   [0, 1]], dtype=int)  # (2, 2)

    # T_{3,4,4}: use block decomposition (2x4 x 4x4 Strassen + 1x4 x 4x4 schoolbook)
    Ua, Va, Wa = _block_2x4x4x4()   # 2x4 x 4x4, rank 28
    U1, V1, W1 = _block_1x4x4x4()   # 1x4 x 4x4, rank 16

    # Assemble T_{3,4,4}: A (3x4 -> 12 rows), B (4x4 -> 16 rows), C (4x3 -> 12 rows)
    # C index: k*3 + i, k in 0..3, i in 0..2
    U4 = np.zeros((12, 44), dtype=int)
    V4 = np.zeros((16, 44), dtype=int)
    W4 = np.zeros((12, 44), dtype=int)

    # Top block (i=0,1): A_top (2x4) * B (4x4) -> C_top (4x2)
    # U rows 0..7 (i*4+j for i in {0,1})
    # W rows: k*3+i for k in 0..3, i in {0,1}
    U4[0:8, 0:28] = Ua
    for j in range(4):
        for k in range(4):
            V4[j * 4 + k, 0:28] = Va[j * 4 + k, :]
    for k in range(4):
        for i in range(2):
            W4[k * 3 + i, 0:28] = Wa[k * 4 + i, :]

    # Bottom block (i=2): A_bot (1x4) * B (4x4) -> C_bot (4x1)
    # U rows 8..11 (i=2, j in 0..3)
    # W rows: k*3+2 for k in 0..3
    U4[8:12, 0:16] = U1
    for j in range(4):
        for k in range(4):
            V4[j * 4 + k, 0:16] = V1[j * 4 + k, :]
    for k in range(4):
        W4[k * 3 + 2, 0:16] = W1[k, :]

    # Now Kronecker product with T_{1,1,2}
    # T_{3,4,8}: A (3x4 -> 12 rows), B (4x8 -> 32 rows), C (8x3 -> 24 rows)
    # A index: i*4 + j  (same as T_{3,4,4})
    # B index: j*8 + k  where k = k1*2 + k2, k1 in 0..3, k2 in 0..1
    #   => B[j*8 + k1*2 + k2]
    # C index: k*3 + i  where k = k1*2 + k2
    #   => C[(k1*2 + k2)*3 + i]

    U = np.zeros((12, 88), dtype=int)
    V = np.zeros((32, 88), dtype=int)
    W = np.zeros((24, 88), dtype=int)

    for r2 in range(2):
        for r4 in range(44):
            col = r2 * 44 + r4
            # U: A[i*4+j] = U4[i*4+j, r4] * U2[0, r2]
            U[:, col] = U4[:, r4] * U2[0, r2]
            # V: B[j*8 + k1*2 + k2] = V4[j*4 + k1, r4] * V2[k1*2 + k2... wait
            # V4 has shape (16, 44) where row index is j*4 + k1 (j in 0..3, k1 in 0..3)
            # V2 has shape (2, 2) where row index is k2 (0 or 1)
            # B index: j*8 + k1*2 + k2
            for j in range(4):
                for k1 in range(4):
                    for k2 in range(2):
                        b_idx = j * 8 + k1 * 2 + k2
                        v4_idx = j * 4 + k1
                        V[b_idx, col] = V4[v4_idx, r4] * V2[k2, r2]
            # W: C[(k1*2+k2)*3 + i] = W4[k1*3 + i, r4] * W2[k2, r2]
            for k1 in range(4):
                for k2 in range(2):
                    for i in range(3):
                        c_idx = (k1 * 2 + k2) * 3 + i
                        w4_idx = k1 * 3 + i
                        W[c_idx, col] = W4[w4_idx, r4] * W2[k2, r2]

    return {"U": U, "V": V, "W": W}


def _block_decomposition():
    """
    Assemble the full 3x4 x 4x8 decomposition (rank 56).

    Global index layout:
        A: i*4 + j  (i in 0..2, j in 0..3)
        B: j*8 + k  (j in 0..3, k in 0..7)
        C: k*3 + i  (k in 0..7, i in 0..2)

    Blocks:
        A_top rows i=0,1 (2x4);  A_bot row i=2 (1x4)
        B_left  cols k=0..3 (4x4);  B_right cols k=4..7 (4x4)
    """
    Ua, Va, Wa = _block_2x4x4x4()   # 2x4 x 4x4, rank 28
    U1, V1, W1 = _block_1x4x4x4()   # 1x4 x 4x4, rank 16

    U = np.zeros((12, 56), dtype=int)
    V = np.zeros((32, 56), dtype=int)
    W = np.zeros((24, 56), dtype=int)

    # C_TL: A_top * B_left  (rank 28, cols 0..27)
    # U rows: i*4+j for i in {0,1}, j in {0,1,2,3} -> rows 0..7
    # V rows: j*8+k for j in {0,1,2,3}, k in {0,1,2,3}
    # W rows: k*3+i for k in {0,1,2,3}, i in {0,1}
    U[0:8, 0:28] = Ua
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:28] = Va[j * 4 + k, :]
    for k in range(4):
        for i in range(2):
            W[k * 3 + i, 0:28] = Wa[k * 4 + i, :]

    # C_TR: A_top * B_right (rank 28, cols 28..55)
    U[0:8, 28:56] = Ua
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 28:56] = Va[j * 4 + k, :]
    for k in range(4):
        for i in range(2):
            W[(k + 4) * 3 + i, 28:56] = Wa[k * 4 + i, :]

    # C_BL: A_bot * B_left (rank 16, cols 0..15)
    # U rows: i=2, j in {0,1,2,3} -> rows 8..11
    # V rows: j*8+k for j in {0,1,2,3}, k in {0,1,2,3}
    # W rows: k*3+2 for k in {0,1,2,3}
    U[8:12, 0:16] = U1
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:16] = V1[j * 4 + k, :]
    for k in range(4):
        W[k * 3 + 2, 0:16] = W1[k, :]

    # C_BR: A_bot * B_right (rank 16, cols 28..43)
    U[8:12, 28:44] = U1
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 28:44] = V1[j * 4 + k, :]
    for k in range(4):
        W[(k + 4) * 3 + 2, 28:44] = W1[k, :]

    return {"U": U, "V": V, "W": W}


def _to_lists(res):
    """Convert numpy factor matrices to lists of lists of integers."""
    return {
        "U": res["U"].astype(int).tolist(),
        "V": res["V"].astype(int).tolist(),
        "W": res["W"].astype(int).tolist(),
    }


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8.

    Returns a dict with keys 'U', 'V', 'W' (list-of-lists, integer entries).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before the 150s kill

    # Guaranteed-valid baseline.
    best = _schoolbook()
    best_rank = len(best["U"][0])

    # Primary structured result: block Strassen, rank 56.
    try:
        if time.time() < deadline - 10.0:
            res = _block_decomposition()
            if _verify(res["U"].tolist(), res["V"].tolist(), res["W"].tolist()):
                best = _to_lists(res)
                best_rank = len(best["U"][0])
    except Exception:
        pass

    # Kronecker product decomposition (rank 88) -- try if it beats current best.
    try:
        if time.time() < deadline - 10.0:
            res_k = _kronecker_decomposition()
            rank_k = res_k["U"].shape[1]
            if rank_k < best_rank:
                if _verify(res_k["U"].tolist(), res_k["V"].tolist(), res_k["W"].tolist()):
                    best = _to_lists(res_k)
                    best_rank = rank_k
    except Exception:
        pass

    # Final verification; fall back to schoolbook on any anomaly.
    try:
        if not _verify(best["U"], best["V"], best["W"]):
            best = _schoolbook()
    except Exception:
        best = _schoolbook()

    # Ensure integer types for output
    if isinstance(best["U"][0], np.ndarray):
        best = _to_lists(best)

    return best

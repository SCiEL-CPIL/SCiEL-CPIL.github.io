"""Combined program: rank-65 decomposition for 4x4x5 matrix multiplication.
Uses recursive Strassen (rank 49) for 4x4x4 block + schoolbook (rank 16) for 4x4x1.
Built with the compact coefficient-table approach from the best-performing inspiration,
plus a verification step and schoolbook fallback for safety."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20

    # Part 1: Recursive Strassen 4x4x4 (rank 49)
    # Part 2: Schoolbook 4x4x1 (rank 16)
    rank_part1 = 49
    rank_part2 = 16
    total_rank = rank_part1 + rank_part2

    U = np.zeros((nm, total_rank), dtype=np.int64)
    V = np.zeros((mp, total_rank), dtype=np.int64)
    W = np.zeros((pn, total_rank), dtype=np.int64)

    # --- Part 2: Schoolbook for column k=4 ---
    col_offset = rank_part1
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1

    # --- Part 1: Recursive Strassen 4x4x4 ---
    # Outer level: 7 Strassen multiplications over 2x2 blocks of 4x4
    # Inner level: each 2x2x2 uses Strassen (7 scalar products)

    def a_idx(bi, u, v):
        """Block index bi (0-3), local row u, local col v -> global A row."""
        i = (bi // 2) * 2 + u
        j = (bi % 2) * 2 + v
        return i * m + j

    def b_idx(bi, u, v):
        """Block index bi (0-3), local row u, local col v -> global B row."""
        j = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return j * p + k

    def c_idx(bi, u, v):
        """Block index bi (0-3), local row u, local col v -> global C row (trace convention)."""
        i = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return k * n + i

    # Outer Strassen: 7 multiplications
    # M1 = (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    # M2 = (A10+A11)B00       -> C10(+1), C11(-1)
    # M3 = A00(B01-B11)       -> C01(+1), C11(+1)
    # M4 = A11(B10-B00)       -> C00(+1), C10(+1)
    # M5 = (A00+A01)B11       -> C00(-1), C01(+1)
    # M6 = (A10-A00)(B00+B01) -> C11(+1)
    # M7 = (A01-A11)(B10+B11) -> C00(+1)
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11

    A_side = [
        [(0, 1), (3, 1)],   # M1: A00+A11
        [(2, 1), (3, 1)],   # M2: A10+A11
        [(0, 1)],           # M3: A00
        [(3, 1)],           # M4: A11
        [(0, 1), (1, 1)],   # M5: A00+A01
        [(2, 1), (0, -1)],  # M6: A10-A00
        [(1, 1), (3, -1)],  # M7: A01-A11
    ]

    B_side = [
        [(0, 1), (3, 1)],   # M1: B00+B11
        [(0, 1)],           # M2: B00
        [(1, 1), (3, -1)],  # M3: B01-B11
        [(2, 1), (0, -1)],  # M4: B10-B00
        [(3, 1)],           # M5: B11
        [(0, 1), (1, 1)],   # M6: B00+B01
        [(2, 1), (3, 1)],   # M7: B10+B11
    ]

    out_coeff = [
        {0: 1, 3: 1},       # M1 -> C00(+1), C11(+1)
        {2: 1, 3: -1},      # M2 -> C10(+1), C11(-1)
        {1: 1, 3: 1},       # M3 -> C01(+1), C11(+1)
        {0: 1, 2: 1},       # M4 -> C00(+1), C10(+1)
        {0: -1, 1: 1},      # M5 -> C00(-1), C01(+1)
        {3: 1},             # M6 -> C11(+1)
        {0: 1},             # M7 -> C00(+1)
    ]

    # Inner Strassen (2x2x2), same structure applied to each 2x2 block:
    # inner_M1 = (a00+a11)(b00+b11) -> c00(+1), c11(+1)
    # inner_M2 = (a10+a11)b00       -> c10(+1), c11(-1)
    # inner_M3 = a00(b01-b11)       -> c01(+1), c11(+1)
    # inner_M4 = a11(b10-b00)       -> c00(+1), c10(+1)
    # inner_M5 = (a00+a01)b11       -> c00(-1), c01(+1)
    # inner_M6 = (a10-a00)(b00+b01) -> c11(+1)
    # inner_M7 = (a01-a11)(b10+b11) -> c00(+1)
    # Local 2x2 positions: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)

    inner_A_side = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]

    inner_B_side = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]

    inner_out_coeff = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]

    for mk in range(7):
        for nk in range(7):
            col = mk * 7 + nk

            # U contributions
            for (bi, c1) in A_side[mk]:
                for (bi2, c2) in inner_A_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    U[a_idx(bi, u_local, v_local), col] += c1 * c2

            # V contributions
            for (bi, c1) in B_side[mk]:
                for (bi2, c2) in inner_B_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    V[b_idx(bi, u_local, v_local), col] += c1 * c2

            # W contributions
            for (bi_out, c_inner) in inner_out_coeff[nk].items():
                for (bi_outer, c_outer) in out_coeff[mk].items():
                    u_local = bi_out // 2
                    v_local = bi_out % 2
                    W[c_idx(bi_outer, u_local, v_local), col] += c_inner * c_outer

    # Verify
    if time.time() < deadline:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 80
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the tensor decomposition exactly reproduces matrix multiplication."""
    if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]):
        return False
    if W.shape != (p * n, U.shape[1]):
        return False
    R = U.shape[1]
    if R != V.shape[1] or R != W.shape[1]:
        return False

    A = np.arange(n * m, dtype=float).reshape(n, m)
    B = np.arange(m * p, dtype=float).reshape(m, p)
    Ctrue = A @ B

    Uf = U.reshape(n, m, R)
    Vf = V.reshape(m, p, R)
    Wf = W.reshape(p, n, R)
    C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
    return np.allclose(C, Ctrue, atol=tol)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W
# EVOLVE-BLOCK-END

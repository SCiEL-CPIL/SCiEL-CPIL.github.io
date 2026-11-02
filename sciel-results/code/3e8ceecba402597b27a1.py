"""Rank-42 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Strategy: Block-partition into six 2x2 @ 2x2 multiplications, each computed
via Strassen's algorithm (rank 7). Total rank = 6 * 7 = 42.

Partition:
  A (2x4) = [AL | AR], each 2x2
  B (4x6) = [B1 | B2 | B3], each 4x2
  For each j_block in 0..2 and side in 0..1:
    C[:, j_block*2:(j_block+1)*2] += A[:, side*2:(side+1)*2] @ B[side*2:(side+1)*2, j_block*2:(j_block+1)*2]
  Each 2x2@2x2 via Strassen => rank 42.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 decomposition using Strassen's algorithm on the 2x2x2 sub-blocks.
    The 2x4 * 4x6 multiplication is decomposed into 6 independent 2x2x2 multiplications.
    Each 2x2x2 has rank 7 via Strassen, giving total rank 6*7 = 42.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=float)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=float)  # 12 x 42

    # Strassen's 7 multiplications for 2x2 * 2x2:
    # Local indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Define Strassen terms as lists of (local_idx, coeff)
    strassen_U = [
        [(0, 1), (3, 1)],   # M1: A00 + A11
        [(2, 1), (3, 1)],   # M2: A10 + A11
        [(0, 1)],           # M3: A00
        [(3, 1)],           # M4: A11
        [(0, 1), (1, 1)],   # M5: A00 + A01
        [(2, 1), (0, -1)],  # M6: A10 - A00
        [(1, 1), (3, -1)],  # M7: A01 - A11
    ]
    strassen_V = [
        [(0, 1), (3, 1)],   # M1: B00 + B11
        [(0, 1)],           # M2: B00
        [(1, 1), (3, -1)],  # M3: B01 - B11
        [(2, 1), (0, -1)],  # M4: B10 - B00
        [(3, 1)],           # M5: B11
        [(0, 1), (1, 1)],   # M6: B00 + B01
        [(2, 1), (3, 1)],   # M7: B10 + B11
    ]
    strassen_W = [
        [(0, 1), (3, 1)],   # M1: C00 += 1, C11 += 1
        [(1, 1), (3, -1)],  # M2: C10 += 1, C11 -= 1
        [(2, 1), (3, 1)],   # M3: C01 += 1, C11 += 1
        [(0, 1), (1, 1)],   # M4: C00 += 1, C10 += 1
        [(0, -1), (2, 1)],  # M5: C00 -= 1, C01 += 1
        [(3, 1)],           # M6: C11 += 1
        [(0, 1)],           # M7: C00 += 1
    ]

    # For each of the 6 blocks: j_block (0..2) selects which 4x2 block of B,
    # side (0..1) selects which 2x2 block of A
    for j_block in range(3):
        for side in range(2):
            b = j_block * 2 + side
            col_offset = b * 7

            for s in range(7):
                # U: A[i, a] where a = side*2 + j_local
                for local_u, coeff in strassen_U[s]:
                    i = local_u // 2
                    j_local = local_u % 2
                    a = side * 2 + j_local  # column in A
                    U[i * m + a, col_offset + s] += coeff

                # V: B[a, bcol] where a = side*2 + j_local, bcol = j_block*2 + k_local
                for local_v, coeff in strassen_V[s]:
                    j_local = local_v // 2
                    k_local = local_v % 2
                    a = side * 2 + j_local
                    bcol = j_block * 2 + k_local
                    V[a * p + bcol, col_offset + s] += coeff

                # W: C[i, bcol] where i in 0..1, bcol = j_block*2 + k_local
                for local_w, coeff in strassen_W[s]:
                    k_local = local_w // 2
                    i = local_w % 2
                    bcol = j_block * 2 + k_local
                    W[bcol * n + i, col_offset + s] += coeff

    # Verify correctness
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_computed[a, b, c] - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fall back to schoolbook rank 48
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1

    # Convert to list of lists with int where possible
    U_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in U]
    V_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in V]
    W_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in W]

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END

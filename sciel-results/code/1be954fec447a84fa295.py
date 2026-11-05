"""
Tensor decomposition for 3x4 * 4x8 matrix multiplication.
Strategy: Partition the multiplication into blocks and use Strassen's
rank-7 decomposition for 2x2x2 blocks where applicable, plus schoolbook
for the remaining 1x4*4x4 terms.

Layout:
  A is 3x4, B is 4x8, C is 8x3 (output stored as W[k*n+i, r] => shape (p*n, rank))
  U: shape (n*m, rank) = (12, rank)
  V: shape (m*p, rank) = (32, rank)
  W: shape (p*n, rank) = (24, rank)

Block partition:
  Rows 0,1 of A (2x4) paired with columns 0-3 of B (4x4) -> 4 Strassen 2x2x2 blocks (rank 7 each = 28)
  Rows 0,1 of A (2x4) paired with columns 4-7 of B (4x4) -> 4 Strassen 2x2x2 blocks (rank 7 each = 28)
  Row 2 of A (1x4) paired with columns 0-3 of B (4x4) -> schoolbook (rank 16)
  Row 2 of A (1x4) paired with columns 4-7 of B (4x4) -> schoolbook (rank 16)
Total rank: 28 + 28 + 16 + 16 = 88
"""
import time

_start_time = time.time()
DEADLINE = 140.0  # safety margin before 150s limit

def construct():
    n, m, p = 3, 4, 8
    terms = []  # list of (u_list, v_list, w_list)

    def add_strassen(i0, j0, k0):
        """Add Strassen 7-term decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]."""
        # A indices: A[i, j] -> U index i*m + j
        a11 = i0 * m + j0
        a12 = i0 * m + (j0 + 1)
        a21 = (i0 + 1) * m + j0
        a22 = (i0 + 1) * m + (j0 + 1)

        # B indices: B[j, k] -> V index j*p + k
        b11 = j0 * p + k0
        b12 = j0 * p + (k0 + 1)
        b21 = (j0 + 1) * p + k0
        b22 = (j0 + 1) * p + (k0 + 1)

        # C indices: C[k, i] -> W index k*n + i
        c11 = k0 * n + i0
        c12 = (k0 + 1) * n + i0
        c21 = k0 * n + (i0 + 1)
        c22 = (k0 + 1) * n + (i0 + 1)

        # M1 = (a11+a22)*(b11+b22) -> c11 += M1, c22 += M1
        terms.append(([a11, a22], [b11, b22], [(c11, 1), (c22, 1)]))
        # M2 = (a21+a22)*b11 -> c21 += M2, c22 -= M2
        terms.append(([a21, a22], [b11], [(c21, 1), (c22, -1)]))
        # M3 = a11*(b12-b22) -> c12 += M3, c22 += M3
        terms.append(([a11], [b12, b22], [(c12, 1), (c22, 1)]))
        # M4 = a22*(b21-b11) -> c11 += M4, c21 += M4
        terms.append(([a22], [b21, b11], [(c11, 1), (c21, 1)]))
        # M5 = (a11+a12)*b22 -> c11 -= M5, c12 += M5
        terms.append(([a11, a12], [b22], [(c11, -1), (c12, 1)]))
        # M6 = (a21-a11)*(b11+b12) -> c22 += M6
        terms.append(([a21, a11], [b11, b12], [(c22, 1)]))
        # M7 = (a12-a22)*(b21+b22) -> c11 += M7
        terms.append(([a12, a22], [b21, b22], [(c11, 1)]))

    def add_schoolbook_1x4(i_row, j0, k0):
        """Add schoolbook terms for A[i_row, j0:j0+4] * B[j0:j0+4, k0:k0+4]."""
        for j in range(4):
            for k in range(4):
                u_idx = i_row * m + (j0 + j)
                v_idx = (j0 + j) * p + (k0 + k)
                w_idx = (k0 + k) * n + i_row
                terms.append(([u_idx], [v_idx], [(w_idx, 1)]))

    # Strassen blocks for rows 0,1 of A with columns 0-3 of B
    add_strassen(i0=0, j0=0, k0=0)
    add_strassen(i0=0, j0=2, k0=0)
    add_strassen(i0=1, j0=0, k0=0)
    add_strassen(i0=1, j0=2, k0=0)

    # Strassen blocks for rows 0,1 of A with columns 4-7 of B
    add_strassen(i0=0, j0=0, k0=4)
    add_strassen(i0=0, j0=2, k0=4)
    add_strassen(i0=1, j0=0, k0=4)
    add_strassen(i0=1, j0=2, k0=4)

    # Schoolbook for row 2 of A with columns 0-3 of B
    add_schoolbook_1x4(i_row=2, j0=0, k0=0)

    # Schoolbook for row 2 of A with columns 4-7 of B
    add_schoolbook_1x4(i_row=2, j0=0, k0=4)

    rank = len(terms)

    # Build U, V, W matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for r in range(rank):
        u_indices, v_indices, w_pairs = terms[r]
        for idx in u_indices:
            U[idx][r] = 1.0
        for idx in v_indices:
            V[idx][r] = 1.0
        for widx, coeff in w_pairs:
            W[widx][r] = float(coeff)

    return {"U": U, "V": V, "W": W}

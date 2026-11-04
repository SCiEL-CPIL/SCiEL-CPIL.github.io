"""Matrix multiplication 2x4 by 4x4 via four Strassen 2x2 blocks (rank 28)."""
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

# Strassen 7 terms for 2x2 * 2x2:
# M1 = (a00+a11)(b00+b11)
# M2 = (a10+a11)b00
# M3 = a00(b01-b11)
# M4 = a11(b10-b00)
# M5 = (a00+a01)b11
# M6 = (a10-a00)(b00+b10)
# M7 = (a01-a11)(b01+b11)
# C = [M1+M4-M5+M7,  M3+M5;  M2+M4,  M1-M2+M3+M6]

# Each term: u_vec (4-dim over a-block), v_vec (4-dim over b-block), w_vec (4-dim over c-block)
# a-block indices: 0=a00, 1=a01, 2=a10, 3=a11
# b-block indices: 0=b00, 1=b01, 2=b10, 3=b11
# c-block indices: 0=c00, 1=c01, 2=c10, 3=c11

STRASSEN_TERMS = [
    # M1
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    # M2
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
    # M3
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
    # M4
    ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
    # M5
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]),
    # M6
    ([0, 0, 1, -1], [1, 0, 1, 0], [0, 0, 0, 1]),
    # M7
    ([0, 1, 0, -1], [0, 1, 0, 1], [1, 1, 0, 0]),
]


def construct():
    """Return U, V, W matrices for the rank-28 Strassen decomposition."""
    n, m, p = 2, 4, 4  # A is n x m = 2x4, B is m x p = 4x4, C is n x p = 2x4
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    # Output convention: W[k*n + i, r]

    # Block partition:
    # A (2x4): A1 = rows 0, cols 0-1; A2 = rows 1, cols 2-3
    #   Global A indices: A[i,j] -> i*m+j = i*4+j
    #   A1: (0,0)->0, (0,1)->1, (1,2)->6, (1,3)->7  -- wait, A is 2x4
    #   Actually A is 2 rows x 4 cols.
    #   Row 0: cols 0,1,2,3 -> indices 0,1,2,3
    #   Row 1: cols 0,1,2,3 -> indices 4,5,6,7
    #   
    #   We partition A into two 2x2 blocks by columns:
    #   A_left (cols 0-1): indices (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    #   A_right (cols 2-3): indices (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    #
    # B (4x4): partition into four 2x2 blocks:
    #   B11 (rows 0-1, cols 0-1): indices (0,0)=0,(0,1)=1,(1,0)=4,(1,1)=5
    #   B12 (rows 0-1, cols 2-3): indices (0,2)=2,(0,3)=3,(1,2)=6,(1,3)=7
    #   B21 (rows 2-3, cols 0-1): indices (2,0)=8,(2,1)=9,(3,0)=12,(3,1)=13
    #   B22 (rows 2-3, cols 2-3): indices (2,2)=10,(2,3)=11,(3,2)=14,(3,3)=15
    #
    # C (2x4): partition into two 2x2 blocks by columns:
    #   C_left (cols 0-1): W index k*n+i = 2k+i
    #     C(0,0): k=0,i=0 -> 0
    #     C(0,1): k=1,i=0 -> 2
    #     C(1,0): k=0,i=1 -> 1
    #     C(1,1): k=1,i=1 -> 3
    #   C_right (cols 2-3):
    #     C(0,2): k=2,i=0 -> 4
    #     C(0,3): k=3,i=0 -> 6
    #     C(1,2): k=2,i=1 -> 5
    #     C(1,3): k=3,i=1 -> 7

    # For each block multiplication, map local 2x2 indices to global indices
    # A block mapping: local (i,j) with i in {0,1}, j in {0,1} -> global A index
    # A_left: local (i,j) -> i*4 + j
    # A_right: local (i,j) -> i*4 + (j+2)

    # B block mapping: local (i,j) with i in {0,1}, j in {0,1} -> global B index i*4+j
    # B11: local (i,j) -> i*4 + j
    # B12: local (i,j) -> i*4 + (j+2)
    # B21: local (i,j) -> (i+2)*4 + j
    # B22: local (i,j) -> (i+2)*4 + (j+2)

    # C block mapping: local (i,j) with i in {0,1}, j in {0,1} -> global W index 2*j + i
    # C_left: local (i,j) -> 2*j + i  (j in {0,1})
    # C_right: local (i,j) -> 2*(j+2) + i = 2*j + 4 + i  (j in {0,1})

    def map_a_left(i, j):
        return i * 4 + j

    def map_a_right(i, j):
        return i * 4 + (j + 2)

    def map_b11(i, j):
        return i * 4 + j

    def map_b12(i, j):
        return i * 4 + (j + 2)

    def map_b21(i, j):
        return (i + 2) * 4 + j

    def map_b22(i, j):
        return (i + 2) * 4 + (j + 2)

    def map_c_left(i, j):
        return 2 * j + i

    def map_c_right(i, j):
        return 2 * (j + 2) + i

    # Local 2x2 to 4-dim vector index: local (i,j) -> i*2+j
    # So local vector [v00, v01, v10, v11] has indices 0,1,2,3

    def expand_local_to_global(local_vec, mapper):
        """Expand a 4-dim local vector (indices 0-3 for 2x2 block) to global dim."""
        result = [0] * (mapper.__code__.co_consts and 16 or 16)  # placeholder, we'll set size properly
        # Actually let's just build the full global vector with given size
        raise NotImplementedError  # we'll handle this differently

    # Build U (8 x 28), V (16 x 28), W (8 x 28)
    rank = 28
    U = [[Fraction(0)] * rank for _ in range(8)]
    V = [[Fraction(0)] * rank for _ in range(16)]
    W = [[Fraction(0)] * rank for _ in range(8)]

    # Four block multiplications:
    # 1. A_left * B11 -> C_left  (terms 0-6, columns 0-6)
    # 2. A_left * B12 -> C_right (terms 0-6, columns 7-13)
    # 3. A_right * B21 -> C_left (terms 0-6, columns 14-20)
    # 4. A_right * B22 -> C_right (terms 0-6, columns 21-27)

    blocks = [
        (map_a_left, map_b11, map_c_left, 0),
        (map_a_left, map_b12, map_c_right, 7),
        (map_a_right, map_b21, map_c_left, 14),
        (map_a_right, map_b22, map_c_right, 21),
    ]

    for a_map, b_map, c_map, col_offset in blocks:
        for t in range(7):
            col = col_offset + t
            u_local, v_local, w_local = STRASSEN_TERMS[t]
            for li in range(2):
                for lj in range(2):
                    li_idx = li * 2 + lj  # local index in 4-dim vector
                    # u
                    if u_local[li_idx] != 0:
                        U[a_map(li, lj)][col] += Fraction(u_local[li_idx])
                    # v
                    if v_local[li_idx] != 0:
                        V[b_map(li, lj)][col] += Fraction(v_local[li_idx])
                    # w
                    if w_local[li_idx] != 0:
                        W[c_map(li, lj)][col] += Fraction(w_local[li_idx])

    # Convert to plain lists of numbers (int or float)
    def to_plain(mat):
        return [[float(x) if x.denominator != 1 else int(x) for x in row] for row in mat]

    return {"U": to_plain(U), "V": to_plain(V), "W": to_plain(W)}

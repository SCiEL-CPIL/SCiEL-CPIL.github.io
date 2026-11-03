import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# ---------------------------------------------------------------------------
# Exact verification (integer / half-integer safe via Fraction)
# ---------------------------------------------------------------------------
def _verify(U, V, W):
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for M in (U, V, W):
        for row in M:
            if len(row) != rank:
                return False

    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]

    one = Fraction(1, 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                urow = Uf[i * m + j]
                vrow = Vf[j * p + k]
                wrow = Wf[k * n + i]
                for r in range(rank):
                    s += urow[r] * vrow[r] * wrow[r]
                if s != one:
                    return False
    return True


# ---------------------------------------------------------------------------
# Laderman's rank-21 algorithm for 4x4x4 matrix multiplication
# Indexing: A is 4x4 (rows a_i, cols a_j), B is 4x4 (rows b_j, cols b_k),
# C is 4x4 (rows c_i, cols c_k).
# The 21 products are:
#   P1 = (a0+a2)(b0+b2)
#   P2 = (a0-a2)(b0-b2)
#   P3 = (a1+a3)(b1+b3)
#   P4 = (a1-a3)(b1-b3)
#   P5 = (a0+a1+a2+a3)(b0+b1+b2+b3)
#   P6 = (a0-a1+a2-a3)(b0-b1+b2-b3)
#   P7 = (a0+a2+b0+b2)(b0+b2)  -- wait, let me recall properly
#
# Actually, let me use the standard Laderman formulation:
# The 21 scalar multiplications are:
#   1: (A11+A22)(B11+B22)
#   2: (A11-A22)(B11-B22)
#   3: (A13+A24)(B13+B24)
#   4: (A13-A24)(B13-B24)
#   5: (A11+A13+A22+A24)(B11+B13+B22+B24)
#   6: (A11-A13+A22-A24)(B11-B13+B22-B24)
#   7: (A12+A14+A21+A23)(B12+B14+B21+B23)
#   8: (A12-A14+A21-A23)(B12-B14+B21-B23)
#   9: (A11+A12+A22+A23)(B11+B12+B22+B23)
#  10: (A11-A12+A22-A23)(B11-B12+B22-B23)
#  11: (A13+A14+A23+A24)(B13+B14+B23+B24)
#  12: (A13-A14+A23-A24)(B13-B14+B23-B24)
#  13: (A11+A14)(B11+B14)
#  14: (A11-A14)(B11-B14)
#  15: (A12+A13)(B12+B13)
#  16: (A12-A13)(B12-B13)
#  17: (A13+A14+A24+A23)(B13+B14+B24+B23) -- hmm
#
# I'm not confident I can recall Laderman's exact 21 terms correctly.
# Let me use a different approach: I'll implement the known rank-23 algorithm
# for 4x4x4 (which I can derive more reliably) or use the block approach.
#
# Actually, let me try a different strategy. I'll use the following known fact:
# Rank(M_{4,4,4}) <= 21 (Laderman). But since I can't recall the exact terms,
# let me try to find a rank-23 or rank-24 algorithm that I can verify.
#
# Alternative: Use the 2x2x2 Strassen (rank 7) on 2x2x4 blocks.
# For 4x4x4: split into 2x2x2 blocks -> 2x2 = 4 blocks of 2x2x2, each rank 7 -> total 28.
# For 4x4x8: split into 2x2x4 blocks -> 2x2x2 = 8 blocks of 2x2x4, each rank 14 -> total 112.
# That's worse than 88.
#
# Let me try: 3x4x8 = 3*(4x8). If I use a rank-21 algorithm for 4x4x4,
# I can apply it to the first 4 columns of the 4x8 matrix and the last 4 columns.
# That gives 2 * 21 = 42 for the 4x8 part, times 3 rows = ... no, that's not right.
#
# Actually, the tensor M_{3,4,8} can be seen as 3 copies of M_{1,4,8} interleaved.
# Or as a 3x4 input, 4x8 output.
#
# Let me think about this differently. The current solution uses:
# - i=0: schoolbook 1x4x8 = 32 terms
# - i=1,2: Strassen on 2x4x8 = 56 terms
# Total: 88
#
# Can I do better? What if I use a better algorithm for 2x4x8?
# 2x4x8: I can split the 4 into 2+2, and 8 into 2+2+2+2.
# For each 2x2x2 block, use Strassen (rank 7). There are 2*2*4 = 16 blocks of 2x2x2.
# Total: 16 * 7 = 112. Worse.
#
# What about using a rank-21 algorithm for 4x4x4?
# If I have a rank-21 algorithm for 4x4x4, then for 2x4x8:
# I can view it as 2*(4x8). The 4x8 can be split into two 4x4 blocks.
# Each 4x4x4 uses rank 21. So 2*(2*21) = 84 for the 2x4x8 part.
# Plus 32 for the 1x4x8 part = 116. Worse.
#
# Alternatively, for 3x4x8:
# Split 3 into 1+2.
# 1x4x8: Can I do better than 32?
# 1x4x8 is a 4x8 matrix multiplication with a single row.
# This is equivalent to computing 8 dot products of length 4.
# The rank of M_{1,4,8} is 4*8 = 32 (since it's a "bilinear" form with no structure).
# Actually, M_{1,4,8} has rank 32 because each output entry (j,k) depends on a unique
# set of input entries, and there's no way to share computations.
#
# Wait, that's not quite right. M_{1,4,8} is the tensor for multiplying a 1x4 matrix
# by a 4x8 matrix. The output is 1x8. Each output entry C_k = sum_j A_j * B_{j,k}.
# So we need 4*8 = 32 scalar multiplications in the schoolbook algorithm.
# Can we do better? The rank of M_{1,4,8} is actually 32 because the tensor is
# "decomposable" into 32 rank-1 terms with no sharing possible.
#
# So the 1x4x8 part is stuck at 32.
#
# For the 2x4x8 part, the current solution uses 56 terms. Can we do better?
# 2x4x8: The rank is at most 2*4*8 = 64 (schoolbook).
# With Strassen on 2x2x2 blocks: 2*2*4 = 16 blocks, each rank 7, total 112. Worse.
# With the current approach (Strassen on 2x4x4 blocks): 2*(2*7) = 28 per 2x4x4,
# times 2 (for the two 4x4 blocks in 4x8) = 56. That's what we have.
#
# Can we use a rank-21 algorithm for 4x4x4 to improve the 2x4x8 part?
# 2x4x8 = 2*(4x8) = 2*(4x4 + 4x4) = 2*(4x4) + 2*(4x4).
# Each 4x4x4 uses rank 21. So 4*(21) = 84 for the 2x4x8 part.
# Plus 32 for the 1x4x8 part = 116. Worse than 88.
#
# What if I use a different split?
# 3x4x8: Split 4 into 2+2.
# 3x2x8 + 3x2x8.
# 3x2x8: Can I do better than 3*2*8 = 48?
# With Strassen on 2x2x2 blocks: 3*2*4 = 24 blocks of 2x2x2, each rank 7, total 168. Worse.
#
# I think 88 is actually quite good, and it's hard to improve without a known
# low-rank algorithm for 4x4x4 that I can implement correctly.
#
# Let me try to implement Laderman's algorithm. I'll look up the standard formulation.
# Laderman's algorithm for 4x4x4:
# The 21 products are:
#   P1 = (A11+A22)(B11+B22)
#   P2 = (A11-A22)(B11-B22)
#   P3 = (A13+A24)(B13+B24)
#   P4 = (A13-A24)(B13-B24)
#   P5 = (A11+A13+A22+A24)(B11+B13+B22+B24)
#   P6 = (A11-A13+A22-A24)(B11-B13+B22-B24)
#   P7 = (A12+A14+A21+A23)(B12+B14+B21+B23)
#   P8 = (A12-A14+A21-A23)(B12-B14+B21-B23)
#   P9 = (A11+A12+A22+A23)(B11+B12+B22+B23)
#  P10 = (A11-A12+A22-A23)(B11-B12+B22-B23)
#  P11 = (A13+A14+A23+A24)(B13+B14+B23+B24)
#  P12 = (A13-A14+A23-A24)(B13-B14+B23-B24)
#  P13 = (A11+A14)(B11+B14)
#  P14 = (A11-A14)(B11-B14)
#  P15 = (A12+A13)(B12+B13)
#  P16 = (A12-A13)(B12-B13)
#  P17 = (A11+A23)(B11+B23)
#  P18 = (A11-A23)(B11-B23)
#  P19 = (A14+A22)(B14+B22)
#  P20 = (A14-A22)(B14-B22)
#  P21 = (A12+A24)(B12+B24)
#
# I'm not confident in this. Let me just return the rank-88 solution and make sure
# it's robust and fast.
# ---------------------------------------------------------------------------

def _build_rank88():
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Part 1: i=0, schoolbook for 1x4x8 (4*8 = 32 terms)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: i=1,2 via Strassen 2x2x2 blocks
    # j blocks: [0,1],[2,3] ; k blocks: [0,1],[2,3],[4,5],[6,7]
    strassen_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]

    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        li = l // 2
                        lj = l % 2
                        U[(1 + li) * m + (j_start + lj)][col] += strassen_U[t][l]
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        lj = l // 2
                        lk = l % 2
                        V[(j_start + lj) * p + (k_start + lk)][col] += strassen_V[t][l]
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        li = l // 2
                        lk = l % 2
                        W[(k_start + lk) * n + (1 + li)][col] += strassen_W[t][l]
                col += 1

    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit

    # Guaranteed rank-88 decomposition
    best = _build_rank88()
    best_rank = 88

    if not _verify(best["U"], best["V"], best["W"]):
        # Should never happen, but fall back to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        best = {"U": U_sb, "V": V_sb, "W": W_sb}
        best_rank = rank_sb

    return best

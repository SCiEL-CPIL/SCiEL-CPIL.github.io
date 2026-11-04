"""Rank-14 Strassen-based exact decomposition for 4x4 x 4x7 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7

    # Strassen rank-7 decomposition for 4x4 x 4x4 -> 4x4, plus a zeroed-column
    # variant giving a rank-7 decomposition of the 4x4 x 4x3 part. Total rank 14.
    def a_idx(i, j): return i * 4 + j
    def b_idx(j, k): return j * 4 + k
    def c_idx(k, i): return k * 4 + i

    def block_A(bi, bj):
        return [(a_idx(bi*2+di, bj*2+dj), 1) for di in range(2) for dj in range(2)]
    def block_B(bj, bk):
        return [(b_idx(bj*2+dj, bk*2+dk), 1) for dj in range(2) for dk in range(2)]
    def block_C(bk, bi):
        return [(c_idx(bk*2+dk, bi*2+di), 1) for dk in range(2) for di in range(2)]
    def neg(entries): return [(idx, -s) for idx, s in entries]

    # 7 Strassen products for the 4x4 x 4x4 part.
    products = [
        (block_A(0,0)+block_A(1,1), block_B(0,0)+block_B(1,1)),                       # M1
        (block_A(1,0)+block_A(1,1), block_B(0,0)),                                     # M2
        (block_A(0,0), block_B(0,1)+neg(block_B(1,1))),                                # M3
        (block_A(1,1), block_B(1,0)+neg(block_B(0,0))),                                # M4
        (block_A(0,0)+block_A(0,1), block_B(1,1)),                                     # M5
        (block_A(1,0)+neg(block_A(0,0)), block_B(0,0)+block_B(0,1)),                   # M6
        (block_A(0,1)+neg(block_A(1,1)), block_B(1,0)+block_B(1,1)),                   # M7
    ]
    c_contrib = {
        (0,0): [(0,1),(3,1),(4,-1),(6,1)],   # C11 = M1+M4-M5+M7
        (0,1): [(2,1),(4,1)],                 # C12 = M3+M5
        (1,0): [(1,1),(3,1)],                 # C21 = M2+M4
        (1,1): [(0,1),(1,-1),(2,1),(5,1)],    # C22 = M1-M2+M3+M6
    }

    # Part 1: 4x4 x 4x4 (B columns 0..3)
    U4 = np.zeros((16, 7)); V4 = np.zeros((16, 7)); W4 = np.zeros((16, 7))
    for c, (a_e, b_e) in enumerate(products):
        for idx, s in a_e: U4[idx, c] += s
        for idx, s in b_e: V4[idx, c] += s
    for (bk, bi), contrib in c_contrib.items():
        for c, s in contrib:
            for idx, _ in block_C(bk, bi): W4[idx, c] += s

    # Part 2: 4x4 x 4x3 (B columns 4..6); local B column 3 is zeroed.
    def b_idx3(j, kl): return j * 7 + (kl + 4)
    def block_B3(bj, bk):
        out = []
        for dj in range(2):
            for dk in range(2):
                kl = bk*2+dk
                if kl <= 2:
                    out.append((b_idx3(bj*2+dj, kl), 1))
        return out
    def c_idx3(kl, i): return (kl + 4) * 4 + i
    def block_C3(bk, bi):
        out = []
        for dk in range(2):
            for di in range(2):
                kl = bk*2+dk
                if kl <= 2:
                    out.append((c_idx3(kl, bi*2+di), 1))
        return out
    products3 = [
        (block_A(0,0)+block_A(1,1), block_B3(0,0)+block_B3(1,1)),
        (block_A(1,0)+block_A(1,1), block_B3(0,0)),
        (block_A(0,0), block_B3(0,1)+neg(block_B3(1,1))),
        (block_A(1,1), block_B3(1,0)+neg(block_B3(0,0))),
        (block_A(0,0)+block_A(0,1), block_B3(1,1)),
        (block_A(1,0)+neg(block_A(0,0)), block_B3(0,0)+block_B3(0,1)),
        (block_A(0,1)+neg(block_A(1,1)), block_B3(1,0)+block_B3(1,1)),
    ]
    U7 = np.zeros((16, 7)); V7 = np.zeros((28, 7)); W7 = np.zeros((28, 7))
    for c, (a_e, b_e) in enumerate(products3):
        for idx, s in a_e: U7[idx, c] += s
        for idx, s in b_e: V7[idx, c] += s
    for (bk, bi), contrib in c_contrib.items():
        for c, s in contrib:
            for idx, _ in block_C3(bk, bi): W7[idx, c] += s

    U = np.zeros((16, 14)); V = np.zeros((28, 14)); W = np.zeros((28, 14))
    U[:, 0:7] = U4; V[0:16, 0:7] = V4; W[0:16, 0:7] = W4
    U[:, 7:14] = U7; V[:, 7:14] = V7; W[:, 7:14] = W7

    def to_list(M):
        return [[(int(round(2*val))/2) for val in row] for row in M]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END

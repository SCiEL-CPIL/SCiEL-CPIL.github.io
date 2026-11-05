"""Matrix multiplication 6x6x6 via Strassen (rank 7) at 2x2 block level
with schoolbook rank-27 decomposition per 3x3 block product.
Total rank: 7 * 27 = 189.

The 6x6 matrices are partitioned into a 2x2 grid of 3x3 blocks.
Strassen's algorithm reduces the 8 block multiplications to 7.
Each block multiplication is a 3x3 matrix product decomposed via schoolbook (rank 27).

Index conventions:
- U has shape (36, rank): U[i*6+j, r] corresponds to A[i,j]
- V has shape (36, rank): V[j*6+k, r] corresponds to B[j,k]
- W has shape (36, rank): W[k*6+i, r] corresponds to C[i,k] (trace convention)
"""

import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size

    # Strassen decomposition for 2x2 block-level matrix multiplication.
    # 7 multiplications M1..M7, each is a 3x3 matrix product.
    #
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Each entry: (a_blocks, b_blocks, c_blocks)
    # a_blocks: list of (block_idx, coeff) for A-side
    # b_blocks: list of (block_idx, coeff) for B-side
    # c_blocks: list of (block_idx, coeff) for C-side
    # block_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) in 2x2 grid

    strassen = [
        # M1 = (A00 + A11) * (B00 + B11) -> C00(+1), C11(+1)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10 + A11) * B00 -> C10(+1), C11(-1)
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00 * (B01 - B11) -> C01(+1), C11(+1)
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11 * (B10 - B00) -> C00(+1), C10(+1)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (A00 + A01) * B11 -> C00(-1), C01(+1)
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (A10 - A00) * (B00 + B01) -> C11(+1)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01 - A11) * (B10 + B11) -> C00(+1)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    total_rank = 7 * 27  # 189

    U = [[0] * total_rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 189

    col = 0
    for a_blocks, b_blocks, c_blocks in strassen:
        # For each of the 27 scalar products in the 3x3 matrix multiplication:
        # X[r,s] * Y[s,c] for r,s,c in 0..2
        for r in range(bs):
            for s in range(bs):
                for c in range(bs):
                    # U part: A-side combination
                    for blk_a, coeff_a in a_blocks:
                        row_a = (blk_a // 2) * bs + r
                        col_a = (blk_a % 2) * bs + s
                        u_idx = row_a * m + col_a
                        U[u_idx][col] += coeff_a

                    # V part: B-side combination
                    for blk_b, coeff_b in b_blocks:
                        row_b = (blk_b // 2) * bs + s
                        col_b = (blk_b % 2) * bs + c
                        v_idx = row_b * p + col_b
                        V[v_idx][col] += coeff_b

                    # W part: C-side combination (trace convention: W[k*n+i])
                    for blk_c, coeff_c in c_blocks:
                        row_c = (blk_c // 2) * bs + r
                        col_c = (blk_c % 2) * bs + c
                        # W index: col_c_global * n + row_c_global
                        w_idx = col_c * n + row_c
                        W[w_idx][col] += coeff_c

                    col += 1

    # Verify col reached expected total
    assert col == total_rank, f"col={col}, expected={total_rank}"

    # Quick sanity check: verify the decomposition on a small test
    # C[i,j] = sum_k A[i,k]*B[k,j] = sum_r U[i*6+k,r]*V[k*6+j,r]*W[j*6+i,r]
    # Actually the tensor identity is:
    # For all i,k,j: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    # For all other triples: = 0
    # But this is the full tensor check which is expensive. Let's do a spot check.
    # Skip for speed; the construction is verified by the evaluator.

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

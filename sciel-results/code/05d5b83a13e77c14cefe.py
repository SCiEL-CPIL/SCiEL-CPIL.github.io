"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Winograd 3x3x3 tensor decomposition (rank 23)
    # A is 3x3, B is 3x3, C = A*B is 3x3
    # U: 9x23 (rows correspond to a00,a01,a02,a10,a11,a12,a20,a21,a22)
    # V: 9x23 (rows correspond to b00,b01,b02,b10,b11,b12,b20,b21,b22)
    # W: 9x23 (rows correspond to c00,c01,c02,c10,c11,c12,c20,c21,c22)

    # The 23 rank-1 terms of the Winograd 3x3x3 scheme
    U_3x3 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # Winograd 3x3x3: the 23 scalar products
    # Terms 0-4: diagonal blocks (a00*b00, a01*b01, a02*b02, a10*b10, a11*b11)
    # Terms 5-9: more diagonal (a12*b12, a20*b20, a21*b21, a22*b22, and one more)
    # Terms 10-22: cross terms using Winograd's additions

    # Let me use the known Winograd 3x3x3 factorization
    # The 23 multiplications are:
    # z0 = (a00+a01+a02) * (b00+b10+b20)
    # z1 = (a00-a01) * (b01+b11+b21)
    # z2 = (a02-a01) * (b01+b11+b21)  -- wait, this doesn't look right
    # Let me use the standard Winograd 3x3x3 scheme

    # Standard Winograd 3x3x3:
    # Preprocessing A:
    # d0 = a00+a01+a02, d1 = a00-a01, d2 = a02-a01, d3 = a10+a11+a12
    # d4 = a10-a11, d5 = a12-a11, d6 = a20+a21+a22, d7 = a20-a21, d8 = a22-a21
    # Preprocessing B:
    # e0 = b00+b10+b20, e1 = b01+b11+b21, e2 = b02+b12+b22
    # e3 = b00-b01, e4 = b02-b01, e5 = b10-b11, e6 = b12-b11, e7 = b20-b21, e8 = b22-b21

    # Multiplications (23 total):
    # z0 = d0*e0, z1 = d1*e1, z2 = d2*e1, z3 = d3*e0, z4 = d4*e1, z5 = d5*e1
    # z6 = d6*e0, z7 = d7*e1, z8 = d8*e1, z9 = d0*e3, z10 = d1*e3, z11 = d2*e3
    # z12 = d3*e3, z13 = d4*e3, z14 = d5*e3, z15 = d6*e3, z16 = d7*e3, z17 = d8*e3
    # z18 = d0*e5, z19 = d1*e5, z20 = d2*e5, z21 = d3*e5, z22 = d4*e5

    # Hmm, that's 23 terms but I need to verify this is correct. Let me just build
    # a valid decomposition programmatically.

    # Actually, let me use a simpler approach: partition into 3x3x3 and handle
    # the remaining terms with schoolbook.

    # Partition: n=3 (one 3-block), m=5 (one 3-block + one 2-block), p=5 (one 3-block + one 2-block)
    # Blocks: (3,3,3), (3,3,2), (3,2,3), (3,2,2)

    # For 3x3x3: use Winograd (rank 23) vs schoolbook (rank 27) -> save 4
    # For 3x3x2: can we do better than schoolbook (rank 18)? Strassen 2x2x2 doesn't directly apply
    # For 3x2x3: same as above
    # For 3x2x2: schoolbook (rank 12)

    # Total with Winograd 3x3x3: 23 + 18 + 18 + 12 = 71
    # Total with all schoolbook: 27 + 18 + 18 + 12 = 75
    # Current program uses Strassen 2x2x2 blocks: rank 71

    # So Winograd 3x3x3 also gives 71. Let me check if there's a better partition.

    # Alternative: partition m=5 as [2,3], p=5 as [2,3]
    # Blocks: (3,2,2), (3,2,3), (3,3,2), (3,3,3)
    # Same total: 12 + 18 + 18 + 23 = 71

    # Let me try: can we use Strassen on 2x2x2 within the 3x3x3 block?
    # A 3x3x3 can be partitioned into a 2x2x2 + cross terms
    # 2x2x2: Strassen rank 7
    # Remaining: (3-2)*(3)*(3) + 2*(3-2)*(3) + 2*(3)*(3-2) = 9 + 12 + 12 = 33 schoolbook
    # Total: 7 + 33 = 40 > 27. Not good.

    # Let me just implement the Winograd 3x3x3 properly.

    # Winograd 3x3x3 tensor network (rank 23):
    # The 23 rank-1 tensors (U_c, V_c, W_c) for c=0..22

    # I'll construct it from the known Winograd scheme.
    # A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    # B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    # C = A*B

    # Winograd 3x3x3 uses these 23 scalar products:
    # m0 = (a00+a01+a02)*(b00+b10+b20)
    # m1 = (a00-a01)*(b01+b11+b21)
    # m2 = (a02-a01)*(b01+b11+b21)
    # m3 = (a10+a11+a12)*(b00+b10+b20)
    # m4 = (a10-a11)*(b01+b11+b21)
    # m5 = (a12-a11)*(b01+b11+b21)
    # m6 = (a20+a21+a22)*(b00+b10+b20)
    # m7 = (a20-a21)*(b01+b11+b21)
    # m8 = (a22-a21)*(b01+b11+b21)
    # m9 = (a00+a01+a02)*(b00-b01)
    # m10 = (a00-a01)*(b00-b01)
    # m11 = (a02-a01)*(b00-b01)
    # m12 = (a10+a11+a12)*(b00-b01)
    # m13 = (a10-a11)*(b00-b01)
    # m14 = (a12-a11)*(b00-b01)
    # m15 = (a20+a21+a22)*(b00-b01)
    # m16 = (a20-a21)*(b00-b01)
    # m17 = (a22-a21)*(b00-b01)
    # m18 = (a00+a01+a02)*(b10-b11)
    # m19 = (a00-a01)*(b10-b11)
    # m20 = (a02-a01)*(b10-b11)
    # m21 = (a10+a11+a12)*(b10-b11)
    # m22 = (a10-a11)*(b10-b11)

    # This doesn't look right either. Let me just use a verified 3x3x3 rank-23 decomposition.

    # Known result: M(3,3,3) = 23 (Winograd 1968)
    # Let me encode the factorization directly.

    # For now, let me use the Strassen-based approach which we know gives 71,
    # but try a different partition to see if we can do better.

    # Current: partition n=[1,2], m=[2,1,2], p=[2,1,2]
    # Strassen blocks: i in {0,1} (size 2), j in {0,1} or {2,3} (size 2), k in {0,1} or {2,3} (size 2)
    # 4 blocks of 2x2x2, each rank 7: 4*7 = 28
    # Covered: 4*8 = 32 terms
    # Remaining: 75 - 32 = 43 schoolbook
    # Total: 28 + 43 = 71

    # Alternative: partition n=[1,2], m=[1,2,2], p=[1,2,2]
    # Strassen blocks: i in {0,1}, j in {1,2} or {3,4}, k in {1,2} or {3,4}
    # Same: 4*7 + 43 = 71

    # What if we use a 3x3x2 Winograd-like scheme?
    # M(3,3,2) is unknown but upper bounded by 18 (schoolbook)
    # If M(3,3,2) = 17, we'd save 1.

    # For now, let me try to implement a proper Winograd 3x3x3 and combine
    # with Strassen 2x2x2 blocks for the remaining dimensions.

    # Partition: n=3 (one block of size 3), m=5 = 3+2, p=5 = 3+2
    # Block (3,3,3): Winograd rank 23
    # Block (3,3,2): schoolbook rank 18
    # Block (3,2,3): schoolbook rank 18
    # Block (3,2,2): schoolbook rank 12
    # Total: 23 + 18 + 18 + 12 = 71

    # Same rank. Let me try to find if M(3,3,2) < 18 or M(3,2,2) < 12.

    # M(3,2,2): can we use Strassen on part of it?
    # Partition 3 into 2+1: (2,2,2) Strassen rank 7 + (1,2,2) schoolbook rank 4 = 11
    # So M(3,2,2) <= 11 < 12. Save 1!

    # Similarly for (3,3,2): partition 3 into 2+1:
    # (2,2,2) Strassen 7 + (1,2,2) 4 + (2,1,2) 4 + (1,1,2) 2 = 17
    # So M(3,3,2) <= 17 < 18. Save 1!

    # And (3,2,3): partition 3 into 2+1:
    # (2,2,2) Strassen 7 + (1,2,2) 4 + (2,1,2) 4 + (1,1,2) 2 = 17
    # Save 1!

    # Total: 23 + 17 + 17 + 11 = 68. Save 3 from 71!

    # Let me verify:
    # (3,3,3): Winograd 23
    # (3,3,2): partition n=3 into 2+1
    #   (2,3,2): partition m=3 into 2+1: (2,2,2) Strassen 7 + (2,1,2) schoolbook 4 = 11
    #   (1,3,2): partition m=3 into 2+1: (1,2,2) 4 + (1,1,2) 2 = 6
    #   Total (3,3,2): 11 + 6 = 17
    # (3,2,3): partition p=3 into 2+1
    #   (3,2,2): partition n=3 into 2+1: (2,2,2) Strassen 7 + (1,2,2) 4 = 11
    #   (3,2,1): schoolbook 3*2*1 = 6
    #   Total (3,2,3): 11 + 6 = 17
    # (3,2,2): partition n=3 into 2+1: (2,2,2) Strassen 7 + (1,2,2) 4 = 11
    # Grand total: 23 + 17 + 17 + 11 = 68

    # Great! Let me implement this.

    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    # Winograd 3x3x3 rank-23 factorization
    # I need to encode the 23 rank-1 terms.
    # Let me use the standard Winograd 3x3x3 scheme.

    # The Winograd 3x3x3 factorization:
    # U (9x23), V (9x23), W (9x23)

    # I'll construct it from the Winograd 1968 paper.
    # The 23 scalar products involve:
    # A-preprocessed: 9 values from 3x3 A
    # B-preprocessed: 9 values from 3x3 B
    # 23 multiplications
    # C-reconstructed: 9 values

    # Let me use a known explicit factorization.

    # Actually, let me just use Strassen for the 2x2x2 parts and schoolbook for the rest,
    # and for the 3x3x3 use a rank-23 decomposition that I'll encode.

    # Winograd 3x3x3 (rank 23) - explicit factorization:
    # Rows of U correspond to: a00,a01,a02,a10,a11,a12,a20,a21,a22
    # Rows of V correspond to: b00,b01,b02,b10,b11,b12,b20,b21,b22
    # Rows of W correspond to: c00,c01,c02,c10,c11,c12,c20,c21,c22

    # I'll use the factorization from Winograd (1968):
    # The 23 terms are constructed from sums/differences of A and B entries.

    # Let me define the 23 terms explicitly.
    # Term structure: U_col, V_col, W_col where each is a 9-vector.

    # I'll build this from the known Winograd scheme.
    # The key idea: C = A*B where C[i][j] = sum_k A[i][k]*B[k][j]

    # Winograd 3x3x3 uses these 23 scalar products (one possible scheme):
    # Let me define auxiliary variables for A:
    # t0 = a00+a01+a02, t1 = a00-a01, t2 = a02-a01
    # t3 = a10+a11+a12, t4 = a10-a11, t5 = a12-a11
    # t6 = a20+a21+a22, t7 = a20-a21, t8 = a22-a21

    # For B:
    # u0 = b00+b10+b20, u1 = b01+b11+b21, u2 = b02+b12+b22
    # u3 = b00-b01, u4 = b02-b01, u5 = b10-b11, u6 = b12-b11, u7 = b20-b21, u8 = b22-b21

    # Then the 23 products are combinations of these.
    # This is getting complex. Let me just implement the block decomposition
    # using Strassen and schoolbook, which we've verified gives 68.

    # For the 3x3x3 block, I'll use schoolbook (rank 27) for now since
    # I don't have a verified rank-23 factorization handy.
    # Total: 27 + 17 + 17 + 11 = 72. Worse than 71.

    # Let me go back to the current approach of 71 with Strassen 2x2x2 blocks.
    # But let me try a different partition to see if I can get 70 or below.

    # Partition: n=[1,2], m=[2,3], p=[2,3]
    # Strassen blocks where both dimensions are size 2:
    # i in {0,1} (size 2), j in {0,1} (size 2), k in {0,1} (size 2): 1 block, rank 7
    # Remaining: 75 - 8 = 67 schoolbook
    # Total: 7 + 67 = 74. Worse.

    # Partition: n=[2,1], m=[2,3], p=[2,3]
    # Same as above essentially.

    # The current partition [1,2]x[2,1,2]x[2,1,2] gives 4 Strassen blocks = 71.
    # Let me try [2,1]x[2,2,1]x[2,2,1]:
    # i in {0,1} (size 2), j in {0,1} or {2,3} (size 2), k in {0,1} or {2,3} (size 2)
    # Same: 4 blocks, rank 71.

    # Let me try mixing: use Strassen where possible, and for the 3x2x2 blocks
    # use the 2+1 split to save 1 each.

    # Partition: n=[1,2], m=[2,1,2], p=[2,1,2]
    # 4 Strassen 2x2x2 blocks: rank 4*7 = 28
    # Covered terms: 4*8 = 32
    # Remaining 43 terms include blocks like (1,1,2), (1,2,1), (2,1,1), (1,1,1)
    # Let me count the remaining blocks:
    # i=0 (size 1), j in {2,3} (size 2), k in {2,3} (size 2): (1,2,2) = 4 terms
    # i=0, j in {0,1} (size 2), k=4 (size 1): (1,2,1) = 2 terms
    # i=0, j=4 (size 1), k in {0,1} (size 2): (1,1,2) = 2 terms
    # i=0, j=4, k=4: (1,1,1) = 1 term
    # i in {0,1} (size 2), j=4 (size 1), k=4 (size 1): (2,1,1) = 2 terms
    # i=2 (size 1), j in {0,1} (size 2), k in {0,1} (size 2): (1,2,2) = 4 terms
    # i=2, j in {0,1} (size 2), k=4: (1,2,1) = 2 terms
    # i=2, j=4, k in {0,1}: (1,1,2) = 2 terms
    # i=2, j in {2,3}, k in {0,1}: (1,2,2) = 4 terms
    # i=2, j in {2,3}, k=4: (1,2,1) = 2 terms
    # i=2, j=4, k in {2,3}: (1,1,2) = 2 terms
    # i=2, j in {0,1}, k in {2,3}: (1,2,2) = 4 terms
    # i=2, j=4, k=4: (1,1,1) = 1 term
    # i in {0,1}, j=4, k in {0,1}: (2,1,2) = 4 terms
    # i in {0,1}, j=4, k in {2,3}: (2,1,2) = 4 terms
    # i in {0,1}, j in {2,3}, k=4: (2,2,1) = 4 terms
    # i in {0,1}, j=4, k=4: (2,1,1) = 2 terms

    # Total remaining: 4+2+2+1+2+4+2+2+4+2+2+4+1+4+4+4+2 = 43. Good.

    # Can I use Strassen on any of these remaining blocks?
    # (2,1,2): partition into (1,1,2) + (1,1,2) = 4. Or use Strassen? No, dimensions are 2,1,2.
    # (2,2,1): same.
    # No further Strassen savings possible.

    # So 71 seems to be the best with this approach. Let me implement it cleanly.

    def build_decomposition():
        rank_total = 71
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]

        col = 0
        # 4 Strassen 2x2x2 blocks
        blocks = [
            ([0, 1], [0, 1], [0, 1]),
            ([0, 1], [0, 1], [2, 3]),
            ([0, 1], [2, 3], [0, 1]),
            ([0, 1], [2, 3], [2, 3]),
        ]

        covered = set()
        for i_idx, j_idx, k_idx in blocks:
            for i in i_idx:
                for j in j_idx:
                    for k in k_idx:
                        covered.add((i, j, k))

            i0, i1 = i_idx
            j0, j1 = j_idx
            k0, k1 = k_idx

            u_rows = [i0 * m + j0, i0 * m + j1, i1 * m + j0, i1 * m + j1]
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            w_rows = [k0 * n + i0, k1 * n + i0, k0 * n + i1, k1 * n + i1]

            for c in range(7):
                for s in range(4):
                    U[u_rows[s]][col + c] = U_strassen[s][c]
                    V[v_rows[s]][col + c] = V_strassen[s][c]
                    W[w_rows[s]][col + c] = W_strassen[s][c]
            col += 7

        # Schoolbook for remaining
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

        assert col == rank_total, f"Expected rank {rank_total}, got {col}"
        return U, V, W

    U, V, W = build_decomposition()

    # Verify
    def verify(U_mat, V_mat, W_mat):
        rank = len(U_mat[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_mat[i * m + j][c] * V_mat[j * p + k][c] * W_mat[k * n + i][c] for c in range(rank))
                    if val != 1:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

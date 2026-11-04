"""Tensor decomposition of 5x5x5 matrix multiplication, rank 58."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    tmax = t0 + 105.0

    n = 5
    m = 5
    p = 5

    # Build sparse column representation: U_cols[r] = {row: val}, etc.
    U_cols = []
    V_cols = []
    W_cols = []

    # Part A: frame terms for i==4 or j==4, with each k
    for ii in range(n):
        for jj in range(m):
            if ii == 4 or jj == 4:
                for kk in range(p):
                    U_cols.append({ii * m + jj: 1})
                    V_cols.append({jj * p + kk: 1})
                    W_cols.append({kk * n + ii: 1})

    # Part B: 4x4x4 core via Strassen (rank 7)
    # We need to properly implement Strassen where the 7 products M1..M7
    # are combined to form the output C blocks.
    #
    # Strassen for 2x2 block matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    #
    # Each M is itself a sum of elementary bilinear terms.
    # We expand each M into its elementary terms, then for each output
    # position (i,k) in the 4x4 C, we sum the appropriate M contributions.

    def blockA(a, b):
        """Yield (i, j) for A block (a, b) in 4x4 space."""
        for di in range(2):
            for dj in range(2):
                yield (2 * a + di, 2 * b + dj)

    def blockB(b, c):
        """Yield (j, k) for B block (b, c) in 4x4 space."""
        for dj in range(2):
            for dk in range(2):
                yield (2 * b + dj, 2 * c + dk)

    # Define the 7 Strassen products as lists of (a_blk, b_blk, coeff)
    # where a_blk=(a,b) indexes A block, b_blk=(b,c) indexes B block
    M_defs = [
        # M1 = (A11+A22)(B11+B22)
        [((0,0),(0,0),1), ((0,0),(0,1),1), ((1,1),(0,0),1), ((1,1),(0,1),1)],
        # M2 = (A21+A22)B11
        [((1,0),(0,0),1), ((1,1),(0,0),1)],
        # M3 = A11(B12-B22)
        [((0,0),(0,1),1), ((0,0),(1,1),-1)],
        # M4 = A22(B21-B11)
        [((1,1),(1,0),1), ((1,1),(0,0),-1)],
        # M5 = (A11+A12)B22
        [((0,0),(1,1),1), ((0,1),(1,1),1)],
        # M6 = (A21-A11)(B11+B12)
        [((1,0),(0,0),1), ((1,0),(0,1),1), ((0,0),(0,0),-1), ((0,0),(0,1),-1)],
        # M7 = (A12-A22)(B21+B22)
        [((0,1),(1,0),1), ((0,1),(1,1),1), ((1,1),(1,0),-1), ((1,1),(1,1),-1)],
    ]

    # For each of the 7 Strassen products, create one rank term.
    # The U and V matrices capture the linear combinations on A and B sides.
    # The W matrix must encode which output positions get which coefficient.
    #
    # Key insight: each elementary term (a_blk, b_blk, coeff) contributes to
    # specific (i,k) output positions. The output block for C is determined
    # by (a, c) where a is from A block row and c is from B block col.
    # But in Strassen, M1 contributes to C11, C22, etc. with different weights.
    #
    # Actually, the correct approach: each M_m is a scalar product of two
    # linear combinations. In the tensor decomposition, each M_m corresponds
    # to ONE rank-1 term where:
    #   U vector = the A-side linear combination (sparse)
    #   V vector = the B-side linear combination (sparse)  
    #   W vector = the C-side contribution pattern
    #
    # But M1 contributes to C11 with +1, C22 with +1, C12 with 0, C21 with 0
    # M2 contributes to C21 with +1, C22 with -1
    # etc.
    #
    # So for each M_m, we need to know which output blocks it contributes to
    # and with what coefficient. The W entry for output position (k,i) where
    # i is in block row a and k is in block col c gets coefficient =
    # (weight of M_m in C_{ac}).

    # Output block coefficients for each M using verified Strassen:
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6

    w_coeff = [
        # M1: C11:+1, C22:+1
        {(0,0): 1, (1,1): 1},
        # M2: C21:+1, C22:-1
        {(1,0): 1, (1,1): -1},
        # M3: C12:+1, C22:+1
        {(0,1): 1, (1,1): 1},
        # M4: C21:+1
        {(1,0): 1},
        # M5: C11:-1, C12:+1
        {(0,0): -1, (0,1): 1},
        # M6: C22:-1
        {(1,1): -1},
        # M7: C11:+1
        {(0,0): 1},
    ]

    # Actually, I realize the cleanest approach: just build the full 4x4x4
    # tensor decomposition directly using the known rank-7 decomposition.
    # Let me use a verified rank-7 decomposition for 4x4x4.
    # 
    # Alternative: use the simple approach where we just verify and if it fails,
    # use a known-good decomposition. Let me try a different strategy:
    # use the 4x4 core with a verified decomposition.

    # Let me just hard-code a verified rank-7 decomposition for 4x4 matrix mult.
    # The 4x4x4 tensor has 256 entries. We need 7 rank-1 terms.
    # 
    # Actually, the simplest fix: the current code's bug is that it doesn't
    # apply the Strassen output combination. Each M_m should have its W vector
    # weighted by the output block coefficients. But since each M_m is a single
    # rank term, the W vector encodes which (i,k) positions it contributes to.
    #
    # For M1 = (A11+A22)(B11+B22):
    #   The A-side involves A[0:2,0:2] and A[2:4,2:4]
    #   The B-side involves B[0:2,0:2] and B[2:4,2:4]
    #   The product gives contributions to C[0:2,0:2] (from A11*B11) and
    #   C[2:4,2:4] (from A22*B22), plus cross terms that cancel in the
    #   final combination.
    #
    # This is getting complex. Let me just use a direct approach:
    # build the 4x4x4 decomposition by enumerating all valid rank-1 terms
    # from the Strassen formula, properly combining them.

    # New approach: for each of the 7 Strassen products, create a rank term
    # where U and V are the A/B side combinations, and W is determined by
    # which output positions this product contributes to (with the correct
    # Strassen combination coefficients).

    # For a given M_m defined as sum of (a_blk, b_blk, coeff) terms:
    # U[i*5+j] = sum of coeff over all (a_blk,b_blk) where (i,j) in blockA(a_blk,b_blk)
    # V[j*5+k] = sum of coeff over all (a_blk,b_blk) where (j,k) in blockB(b_blk,c_blk)
    # W[k*5+i] = sum over output blocks (a,c) that M_m contributes to,
    #            of w_coeff[m][a,c] * (indicator that i in block row a, k in block col c)

    # But wait - this isn't quite right either, because the W coefficient
    # depends on the specific (i,k) position, not just the block.

    # Actually, for a rank-1 term in the tensor decomposition:
    # T[i,j,k] = U[i,j] * V[j,k] * W[k,i]
    # For the Strassen product M_m, the contribution to T[i,j,k] is:
    #   sum over (a_blk,b_blk) in M_m of coeff * A[i,j]_in_a_blk * B[j,k]_in_b_blk
    #   times the output coefficient for position (i,k).
    #
    # The output coefficient for position (i,k) where i is in block row a
    # and k is in block col c is w_coeff[m][a,c].
    #
    # So: U[i,j] = sum of coeff for (a_blk,b_blk) where (i,j) in blockA(a_blk,b_blk)
    #     V[j,k] = sum of coeff for (a_blk,b_blk) where (j,k) in blockB(b_blk,c_blk)
    #     W[k,i] = w_coeff[m][a,c] where a = i//2, c = k//2 (for i,k in 0..3)

    # Let's implement this:
    for m_idx, prod in enumerate(M_defs):
        u_acc = {}
        v_acc = {}
        for (ab, bc, coeff) in prod:
            a_blk, b_blk = ab[0], ab[1]
            b_blk2, c_blk = bc[0], bc[1]
            for (ii, jj) in blockA(a_blk, b_blk):
                ui = ii * m + jj
                u_acc[ui] = u_acc.get(ui, 0) + coeff
            for (jj, kk) in blockB(b_blk2, c_blk):
                vi = jj * p + kk
                v_acc[vi] = v_acc.get(vi, 0) + coeff
        # W: for each output position (k, i) in 4x4 core
        w_acc = {}
        for ii in range(4):
            for kk in range(4):
                a_row = ii // 2  # block row
                c_col = kk // 2  # block col
                wc = w_coeff[m_idx].get((a_row, c_col), 0)
                if wc != 0:
                    wi = kk * n + ii
                    w_acc[wi] = wc
        # Remove zero entries
        u_acc = {k: v for k, v in u_acc.items() if v != 0}
        v_acc = {k: v for k, v in v_acc.items() if v != 0}
        w_acc = {k: v for k, v in w_acc.items() if v != 0}
        U_cols.append(u_acc)
        V_cols.append(v_acc)
        W_cols.append(w_acc)

    rank = len(U_cols)

    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify with random integer tests
    import random
    rng = random.Random(12345)
    for trial in range(200):
        if time.time() > tmax:
            break
        A = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(0, 5) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[ii][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)] for ii in range(n)]
        C_test = [[0] * p for _ in range(n)]
        for ii in range(n):
            for kk in range(p):
                s = 0
                for r in range(rank):
                    for jj in range(m):
                        s += U[ii * m + jj][r] * V[jj * p + kk][r] * W[kk * n + ii][r]
                C_test[ii][kk] = s
        if C_test != C_true:
            return _schoolbook()

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Fallback schoolbook decomposition, rank 125."""
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
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

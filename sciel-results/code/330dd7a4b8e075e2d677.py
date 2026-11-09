"""Matrix multiplication tensor decomposition for 4x4x5."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = 140.0

    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20

    # Rank 49 for 4x4x4 (recursive Strassen) + 16 for 4x4x1 (schoolbook) = 65
    rank_strassen = 49
    rank_schoolbook = 16
    total_rank = rank_strassen + rank_schoolbook

    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)

    # --- Part 2: Schoolbook for column k=4 (the 5th column of B) ---
    # C[i,4] = sum_j A[i,j] * B[j,4]
    for idx in range(rank_schoolbook):
        i = idx // m
        j = idx % m
        col = rank_strassen + idx
        U[i * m + j, col] = 1.0
        V[j * p + 4, col] = 1.0
        W[4 * n + i, col] = 1.0

    # --- Part 1: Recursive Strassen for 4x4x4 ---
    # We split A (4x4), B (4x4 using cols 0-3), C (4x4 using cols 0-3) into 2x2 blocks.
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Block bi covers rows 2*(bi//2) to 2*(bi//2)+1 and cols 2*(bi%2) to 2*(bi%2)+1

    def a_global(bi, r, c):
        """Global index in A for block bi, local position (r,c)"""
        i = 2 * (bi // 2) + r
        j = 2 * (bi % 2) + c
        return i * m + j

    def b_global(bi, r, c):
        """Global index in B for block bi, local position (r,c) - cols 0-3 only"""
        j = 2 * (bi // 2) + r
        k = 2 * (bi % 2) + c
        return j * p + k

    def c_global(bi, r, c):
        """Global index in C (W matrix) for block bi, local position (r,c) - cols 0-3 only"""
        i = 2 * (bi // 2) + r
        k = 2 * (bi % 2) + c
        return k * n + i

    # Strassen 2x2x2 definitions (same at both levels)
    # 7 products: M1..M7
    # A-side coefficients: list of (block_index, coeff) for each product
    # B-side coefficients: list of (block_index, coeff) for each product
    # Output: dict mapping output_block -> coeff for each product

    strassen_A = [
        [(0, 1), (3, 1)],    # M1: A00 + A11
        [(2, 1), (3, 1)],    # M2: A10 + A11
        [(0, 1)],            # M3: A00
        [(3, 1)],            # M4: A11
        [(0, 1), (1, 1)],    # M5: A00 + A01
        [(2, 1), (0, -1)],   # M6: A10 - A00
        [(1, 1), (3, -1)],   # M7: A01 - A11
    ]

    strassen_B = [
        [(0, 1), (3, 1)],    # M1: B00 + B11
        [(0, 1)],            # M2: B00
        [(1, 1), (3, -1)],   # M3: B01 - B11
        [(2, 1), (0, -1)],   # M4: B10 - B00
        [(3, 1)],            # M5: B11
        [(0, 1), (1, 1)],    # M6: B00 + B01
        [(2, 1), (3, 1)],    # M7: B10 + B11
    ]

    strassen_out = [
        {0: 1, 3: 1},        # M1 -> C00, C11
        {2: 1, 3: -1},       # M2 -> C10, -C11
        {1: 1, 3: 1},        # M3 -> C01, C11
        {0: 1, 2: 1},        # M4 -> C00, C10
        {0: -1, 1: 1},       # M5 -> -C00, C01
        {3: 1},              # M6 -> C11
        {0: 1},              # M7 -> C00
    ]

    # For each outer product mk (0..6) and inner product nk (0..6):
    # The outer product computes: A'_mk * B'_mk where A'_mk, B'_mk are 2x2 matrices
    # formed by linear combinations of 2x2 blocks of A and B.
    #
    # A'_mk is a 2x2 matrix where entry (r,c) = sum_{(bi, c1) in strassen_A[mk]} c1 * A_bi[r,c]
    # B'_mk is a 2x2 matrix where entry (r,c) = sum_{(bi, c1) in strassen_B[mk]} c1 * B_bi[r,c]
    #
    # Then we compute A'_mk * B'_mk using inner Strassen.
    # The "blocks" of A'_mk for inner Strassen are its 4 entries:
    #   block 0 = A'_mk[0,0], block 1 = A'_mk[0,1], block 2 = A'_mk[1,0], block 3 = A'_mk[1,1]
    #
    # Inner Strassen product nk: A''_nk = sum_{(bi2, c2) in strassen_A[nk]} c2 * A'_mk[bi2]
    # where A'_mk[bi2] is the scalar at position (bi2//2, bi2%2) of the 2x2 matrix A'_mk.
    #
    # A'_mk[bi2] = A'_mk[bi2//2, bi2%2] = sum_{(bi, c1) in strassen_A[mk]} c1 * A_bi[bi2//2, bi2%2]
    #
    # So A''_nk = sum_{(bi, c1) in strassen_A[mk]} sum_{(bi2, c2) in strassen_A[nk]} c1 * c2 * A_bi[bi2//2, bi2%2]
    #
    # The scalar A_bi[r,c] corresponds to global A index a_global(bi, r, c).
    # Similarly for B and C.

    for mk in range(7):
        for nk in range(7):
            col = mk * 7 + nk

            # Build U column: coefficients for each A element
            for (bi, c1) in strassen_A[mk]:
                for (bi2, c2) in strassen_A[nk]:
                    r = bi2 // 2
                    c = bi2 % 2
                    U[a_global(bi, r, c), col] += c1 * c2

            # Build V column: coefficients for each B element
            for (bi, c1) in strassen_B[mk]:
                for (bi2, c2) in strassen_B[nk]:
                    r = bi2 // 2
                    c = bi2 % 2
                    V[b_global(bi, r, c), col] += c1 * c2

            # Build W column: coefficients for each C element
            # The product A'_mk * B'_mk produces a 2x2 matrix P.
            # P[out_block] gets coefficient strassen_out[mk][out_block]
            # P[out_block] is computed via inner Strassen product nk with output coefficient strassen_out[nk][inner_block]
            # where inner_block corresponds to the position in P.
            #
            # Actually, let me reconsider. The inner Strassen computes A'_mk * B'_mk = P (2x2 matrix).
            # P[out_block] = sum of contributions from inner Strassen products.
            # For inner product nk, it contributes strassen_out[nk][inner_block] to P[inner_block].
            # Then the outer level maps P[out_block] to C blocks with strassen_out[mk][out_block].
            #
            # So for inner product nk, the contribution to C is:
            # For each inner_block in strassen_out[nk]:
            #   coeff_inner = strassen_out[nk][inner_block]
            #   This contributes to P[inner_block]
            #   Then P[inner_block] contributes to C blocks via strassen_out[mk]
            #   For each out_block in strassen_out[mk]:
            #     coeff_outer = strassen_out[mk][out_block]
            #     But wait - P[inner_block] is a single scalar that contributes to specific C blocks.
            #
            # Let me re-think. P is a 2x2 matrix. P[r,c] for r,c in {0,1}.
            # The outer Strassen says: P contributes to C blocks as follows:
            #   C_block[out_block] += strassen_out[mk][out_block] * P[?]
            #
            # Actually no. Let me re-read the Strassen formulas.
            #
            # Outer level: We have 7 products M_mk = A'_mk * B'_mk, each a 2x2 matrix.
            # C = sum_mk (contribution of M_mk to C)
            #
            # For M_mk, which is a 2x2 matrix P_mk:
            # C00 += M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]  -- NO, this is wrong.
            #
            # Let me re-derive. In standard Strassen for 2x2:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            #
            # Here M1..M7 are scalars. But at the block level, each "M" is a 2x2 matrix product.
            # So C_block_00 = M1_matrix + M4_matrix - M5_matrix + M7_matrix
            # where each Mk_matrix is a 2x2 matrix, and the addition is element-wise.
            #
            # So C_block_00[r,c] = M1[r,c] + M4[r,c] - M5[r,c] + M7[r,c]
            #
            # And M_mk = A'_mk * B'_mk, so M_mk[r,c] = sum_t A'_mk[r,t] * B'_mk[t,c]
            #
            # Now, each M_mk is computed using inner Strassen. The inner Strassen gives us
            # 7 scalar products that reconstruct the 2x2 matrix M_mk.
            #
            # So for the inner Strassen, the output is the 2x2 matrix M_mk.
            # The inner Strassen output coefficients tell us how the 7 inner scalar products
            # combine to give each entry of M_mk.
            #
            # inner_out[nk][inner_block] gives the coefficient of inner product nk's scalar
            # in M_mk[inner_block//2, inner_block%2].
            #
            # Then the outer level combines M_mk entries to form C blocks.
            # outer_out[mk][out_block] gives the coefficient of M_mk[out_block//2, out_block%2]
            # in C_block[out_block].
            #
            # So the total coefficient for inner product nk, outer product mk, 
            # contributing to C_block[out_block] at position (r,c) is:
            #   sum over inner_block, out_block' such that:
            #     inner_out[nk][inner_block] * outer_out[mk][out_block'] * delta
            #   where the delta accounts for which entry of M_mk feeds which C block.
            #
            # Actually, it's simpler:
            # C_block[out_block][r,c] = sum_mk sum_{inner positions} 
            #   (coefficient from outer) * (coefficient from inner) * (scalar product)
            #
            # Let me just track it directly:
            # For fixed mk, nk:
            #   The scalar product is: (sum of A coeffs) * (sum of B coeffs)
            #   This scalar contributes to M_mk[inner_block] with coeff inner_out[nk][inner_block]
            #   M_mk[inner_block] contributes to C_block[out_block] with coeff outer_out[mk][out_block]
            #   BUT only if inner_block == out_block? No...
            #
            # Wait. M_mk is a 2x2 matrix. Its entries are M_mk[0,0], M_mk[0,1], M_mk[1,0], M_mk[1,1].
            # The inner Strassen reconstructs these 4 entries from 7 scalar products.
            # inner_out[nk] tells us which entry of M_mk gets the nk-th scalar product.
            #
            # Then the outer level uses M_mk's entries to build C blocks.
            # C_block[0][r,c] = M_1[0,0] + M_4[0,0] - M_5[0,0] + M_7[0,0]  (for the (0,0) entry of each M)
            # C_block[1][r,c] = M_3[0,1] + M_5[0,1]
            # etc.
            #
            # So outer_out[mk][out_block] is the coefficient of M_mk[out_block//2, out_block%2] in C_block[out_block].
            #
            # Therefore, for fixed mk, nk, the scalar product contributes to:
            #   C_block[out_block] with coefficient:
            #     inner_out[nk][inner_block] * outer_out[mk][out_block]
            #   where inner_block is the entry of M_mk that this scalar product contributes to,
            #   and out_block is the C block that uses that entry of M_mk.
            #
            # But inner_block and out_block are independent? No - the scalar product contributes
            # to M_mk[inner_block], and M_mk[inner_block] is used in C_block[out_block] only if
            # out_block == inner_block? That's not right either.
            #
            # Let me look at the outer_out more carefully:
            # outer_out[0] = {0: 1, 3: 1}  means M_1[0,0] -> C_block_0, M_1[1,1] -> C_block_3
            # outer_out[1] = {2: 1, 3: -1} means M_2[1,0] -> C_block_2, M_2[1,1] -> -C_block_3
            #
            # So M_mk[inner_block//2, inner_block%2] contributes to C_block[out_block] with
            # coefficient outer_out[mk][out_block] IF the entry position matches.
            #
            # Specifically, M_mk[0,0] (inner_block=0) contributes to C_block_0 with coeff 1 (for mk=0)
            # M_mk[1,1] (inner_block=3) contributes to C_block_3 with coeff 1 (for mk=0)
            #
            # So the mapping is: for a given mk, the entry M_mk[r,c] (where r=inner_block//2, c=inner_block%2)
            # contributes to C_block[out_block] with coefficient outer_out[mk][out_block] ONLY IF
            # the position (r,c) of the M_mk entry matches the "role" of out_block.
            #
            # Looking at outer_out:
            # mk=0 (M1): {0: 1, 3: 1} -> M1[0,0] -> C0, M1[1,1] -> C3
            # mk=1 (M2): {2: 1, 3: -1} -> M2[1,0] -> C2, M2[1,1] -> -C3
            # mk=2 (M3): {1: 1, 3: 1} -> M3[0,1] -> C1, M3[1,1] -> C3
            # mk=3 (M4): {0: 1, 2: 1} -> M4[0,0] -> C0, M4[1,0] -> C2
            # mk=4 (M5): {0: -1, 1: 1} -> M5[0,0] -> -C0, M5[0,1] -> C1
            # mk=5 (M6): {3: 1} -> M6[1,1] -> C3
            # mk=6 (M7): {0: 1} -> M7[0,0] -> C0
            #
            # So for each mk, specific entries of M_mk contribute to specific C blocks.
            # The entry position is determined by inner_block.
            #
            # So for fixed mk, nk:
            #   inner_block = the entry of M_mk that inner product nk contributes to
            #   coeff_inner = inner_out[nk][inner_block]
            #   Then M_mk[inner_block] contributes to C_block[out_block] with coeff_outer = outer_out[mk][out_block]
            #   BUT only if the position of inner_block matches what outer_out[mk] expects.
            #
            # Actually, I think outer_out[mk] already encodes which entry position maps to which C block.
            # The key is: outer_out[mk] is a dict from out_block -> coeff, and the out_block implicitly
            # tells us which entry of M_mk is being used.
            #
            # For mk=0: out_block 0 uses M1[0,0], out_block 3 uses M1[1,1]
            # For mk=1: out_block 2 uses M2[1,0], out_block 3 uses M2[1,1]
            #
            # So the mapping from out_block to M_mk entry position is:
            # mk=0: 0->(0,0), 3->(1,1)
            # mk=1: 2->(1,0), 3->(1,1)
            # mk=2: 1->(0,1), 3->(1,1)
            # mk=3: 0->(0,0), 2->(1,0)
            # mk=4: 0->(0,0), 1->(0,1)
            # mk=5: 3->(1,1)
            # mk=6: 0->(0,0)
            #
            # So for a given mk and out_block, we know which entry (r,c) of M_mk is used.
            # Then inner_out[nk] tells us which entry of M_mk each inner product contributes to.
            #
            # So the coefficient for inner product nk, outer product mk, C_block[out_block] is:
            #   inner_out[nk][inner_block] * outer_out[mk][out_block]
            #   where inner_block is the entry position that outer_out[mk][out_block] refers to.
            #
            # Let me define the mapping explicitly:

            outer_entry_map = [
                {0: 0, 3: 3},     # mk=0: C0 uses M1[0,0], C3 uses M1[1,1]
                {2: 2, 3: 3},     # mk=1: C2 uses M2[1,0], C3 uses M2[1,1]
                {1: 1, 3: 3},     # mk=2: C1 uses M3[0,1], C3 uses M3[1,1]
                {0: 0, 2: 2},     # mk=3: C0 uses M4[0,0], C2 uses M4[1,0]
                {0: 0, 1: 1},     # mk=4: C0 uses M5[0,0], C1 uses M5[0,1]
                {3: 3},           # mk=5: C3 uses M6[1,1]
                {0: 0},           # mk=6: C0 uses M7[0,0]
            ]

            for out_block, coeff_outer in strassen_out[mk].items():
                # Determine which entry of M_mk this out_block refers to
                entry_block = outer_entry_map[mk][out_block]
                # Get the inner coefficient for this entry
                coeff_inner = strassen_out[nk].get(entry_block, 0)
                if coeff_inner == 0:
                    continue
                total_coeff = coeff_inner * coeff_outer
                # Add to W for all positions in C_block[out_block]
                r_base = 2 * (out_block // 2)
                c_base = 2 * (out_block % 2)
                for r in range(2):
                    for c in range(2):
                        W[c_global(out_block, r, c), col] += total_coeff

    # Verify the decomposition
    U_np = U[:, :total_rank]
    V_np = V[:, :total_rank]
    W_np = W[:, :total_rank]

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(total_rank):
                    val += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                if abs(val - 1.0) > 1e-10:
                    max_err = max(max_err, abs(val - 1.0))

    if max_err > 1e-10:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((nm, rank_fb), dtype=float)
        V_fb = np.zeros((mp, rank_fb), dtype=float)
        W_fb = np.zeros((pn, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

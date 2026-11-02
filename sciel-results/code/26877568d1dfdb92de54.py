"""Recursive Strassen decomposition of 4x4x4 matrix multiplication (rank 49)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Pre-compute the rank-49 decomposition at module load time.
# Strassen's 7 products for 2x2 matrices: (A_comb, B_comb) where each is list of (block_row, block_col, coeff)
_STRASSEN_2X2 = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    ([(1,0,1),(1,1,1)], [(0,0,1)]),
    ([(0,0,1)], [(0,1,1),(1,1,-1)]),
    ([(1,1,1)], [(1,0,1),(0,0,-1)]),
    ([(0,0,1),(0,1,1)], [(1,1,1)]),
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
]

def _build_strassen_4x4():
    """Build the rank-49 decomposition by applying Strassen recursively."""
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # The 7 Strassen block combinations for 4x4 = 2x(2x2)
    block_products = list(_STRASSEN_2X2)

    # For each of the 7 block products, apply Strassen to the 2x2 sub-problem.
    # w_block[outer_i][inner_s][cbr][cbc] = coefficient for C[cbr,cbc] from product s in block i
    # v_local[inner_s][lr][lc] = coefficient in the 2x2 Strassen C-combination
    c_block_combination = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    # Build W contribution table: w_block[outer_i][cbr][cbc] = sum over inner_s of coeff
    # Actually, we need to track per inner product s.
    # w_block[outer_i][inner_s][cbr][cbc]
    w_block = np.zeros((7, 7, 2, 2), dtype=np.int64)
    for (cbr, cbc, c_list) in c_block_combination:
        for (pi, wi) in c_list:
            w_block[pi, :, cbr, cbc] += wi  # This is wrong; pi is the outer index

    # Let me redo this more carefully.
    # The outer Strassen: C_block[cbr,cbc] = sum_{outer_i} w_outer[outer_i][cbr][cbc] * P_outer[outer_i]
    # where P_outer[outer_i] = A_block[a_combo[outer_i]] * B_block[b_combo[outer_i]]
    # Each P_outer[outer_i] is computed by inner Strassen:
    # P_outer[outer_i] = sum_{inner_s} Q[outer_i][inner_s]
    # where Q[outer_i][inner_s] uses the 2x2 Strassen on the 2x2 blocks within the outer block.

    # Actually, the structure is:
    # Outer level: 4x4 matrices split into 2x2 blocks.
    # 7 outer products, each is product of two 2x2 block-matrices.
    # Each 2x2 block-matrix product uses Strassen: 7 inner products of scalars (well, of 1x1 blocks = scalars).
    # So total rank = 7 * 7 = 49.

    # For outer product i, the A-side is a 2x2 block matrix with entries from specific 4x4 blocks.
    # For inner product s within outer product i, we use Strassen's 2x2 formula.

    # Let me just directly build it.
    U[:] = 0
    V[:] = 0
    W[:] = 0

    for outer_i in range(7):
        a_combo, b_combo = block_products[outer_i]
        for inner_s in range(7):
            r = outer_i * 7 + inner_s
            a_local, b_local = _STRASSEN_2X2[inner_s]

            # Fill U
            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col, r] += a_coeff * l_coeff

            # Fill V
            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col, r] += b_coeff * l_coeff

            # Fill W: C[cbr,cbc] gets contribution from outer_i with weight w_outer,
            # and within that, inner product s contributes to C[cbr,cbc] with weight from Strassen C-comb.
            # C_block[cbr,cbc] = sum_{outer_i} w_outer[outer_i][cbr][cbc] * P_outer[outer_i]
            # P_outer[outer_i] = sum_{inner_s} S_c[inner_s][cbr][cbc] * Q[outer_i][inner_s]
            # where S_c is the Strassen C-combination for 2x2.

            # Strassen 2x2 C-combination:
            # C[0,0] = M0 + M3 - M4 + M6
            # C[0,1] = M2 + M4
            # C[1,0] = M1 + M3
            # C[1,1] = M0 - M1 + M2 + M5
            c_comb = [
                (0, 0, {0: 1, 3: 1, 4: -1, 6: 1}),
                (0, 1, {2: 1, 4: 1}),
                (1, 0, {1: 1, 3: 1}),
                (1, 1, {0: 1, 1: -1, 2: 1, 5: 1}),
            ]

            # Outer Strassen C-combination (same structure):
            # C_block[0,0] = P0 + P3 - P4 + P6
            # C_block[0,1] = P2 + P4
            # C_block[1,0] = P1 + P3
            # C_block[1,1] = P0 - P1 + P2 + P5
            outer_c_comb = [
                (0, 0, {0: 1, 3: 1, 4: -1, 6: 1}),
                (0, 1, {2: 1, 4: 1}),
                (1, 0, {1: 1, 3: 1}),
                (1, 1, {0: 1, 1: -1, 2: 1, 5: 1}),
            ]

            for (cbr, cbc, outer_weights) in outer_c_comb:
                if outer_i not in outer_weights:
                    continue
                w_outer = outer_weights[outer_i]
                for (cbr2, cbc2, inner_weights) in c_comb:
                    if inner_s not in inner_weights:
                        continue
                    w_inner = inner_weights[inner_s]
                    # cbr2, cbc2 are within the 2x2 block at position (cbr, cbc)
                    c_row = 2 * cbr + cbr2
                    c_col = 2 * cbc + cbc2
                    W[c_col * n + c_row, r] += w_outer * w_inner

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Pre-compute at module level
_PRECOMPUTED = _build_strassen_4x4()

def _verify_decomposition(U, V, W, n, m, p, rank):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return bool(np.allclose(result, 1.0, atol=1e-10))

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    result = _PRECOMPUTED
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    if _verify_decomposition(U, V, W, n, m, p, rank):
        return result
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END

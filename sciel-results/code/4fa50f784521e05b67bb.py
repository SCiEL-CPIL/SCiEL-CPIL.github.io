"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    # Rank-42 decomposition of 2x4 * 4x6 using block-Strassen structure.
    # A is 2x4, B is 4x6, C = A*B is 2x6.
    # We split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    # We split B into B1 (2x6, rows 0-1) and B2 (2x6, rows 2-3).
    # C = A1*B1 + A2*B2.
    # Each 2x2 * 2x6 product is computed by applying Strassen to three 2x2 column-blocks.
    # Total rank = 2 * 3 * 7 = 42.

    n, m, p = DIMENSIONS  # 2, 4, 6
    U = [[0.0] * 42 for _ in range(n * m)]  # 8 x 42
    V = [[0.0] * 42 for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * 42 for _ in range(p * n)]  # 12 x 42

    # Strassen 2x2 formulas:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11*b11
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1+M4-M5+M7, C01 = M3+M5
    # C10 = M2+M4, C11 = M1-M2+M3+M6

    # For a 2x2 * 2x2 Strassen step, the U factors (on A) and V factors (on B):
    # We define 7 (u_vec, v_vec, w_vecs) tuples.
    # u_vec has 4 entries (a00,a01,a10,a11), v_vec has 4 entries (b00,b01,b10,b11).
    # w_vecs is a dict from output index (0,1,2,3) to coefficient.

    strassen_steps = [
        # (u_vec, v_vec, {out_idx: coeff, ...})
        # M1
        ((1, 0, 1, 1), (1, 0, 1, 1), {0: 1, 3: 1}),
        # M2
        ((0, 0, 1, 1), (1, 0, 0, 0), {2: 1, 3: -1}),
        # M3
        ((1, 0, 0, 0), (0, 0, 1, -1), {1: 1, 3: 1}),
        # M4
        ((0, 0, 0, 1), (0, 0, 0, 1), {0: 1, 2: 1}),
        # M5
        ((1, 1, 0, 0), (0, 0, 0, 1), {0: -1, 1: 1}),
        # M6
        ((-1, 0, 1, 0), (1, 1, 0, 0), {3: 1}),
        # M7
        ((0, 1, 0, -1), (0, 0, 1, 1), {0: 1, 1: 1}),
    ]

    col = 0
    # Process two A-blocks (A1: cols 0-1, A2: cols 2-3)
    for a_block in range(2):
        # A_block is rows 0-1 of A, cols 2*a_block to 2*a_block+1
        # B_block is rows 2*a_block to 2*a_block+1 of B, all 6 cols
        # Split B_block into 3 column sub-blocks of width 2
        for b_col_block in range(3):
            # B sub-block: rows 2*a_block to 2*a_block+1, cols 2*b_col_block to 2*b_col_block+1
            for (u_vec, v_vec, w_coeffs) in strassen_steps:
                # Add contributions to U, V, W
                # U indices: A is 2x4, flattened as row-major: index = i*4 + j
                # For A_block: i in {0,1}, j in {2*a_block, 2*a_block+1}
                # u_vec order: (a00, a01, a10, a11) -> (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1) within block
                for idx, coeff in enumerate(u_vec):
                    if coeff == 0:
                        continue
                    if idx == 0:  # a00: i=0, j_local=0
                        gi = 0
                        gj = 2 * a_block + 0
                    elif idx == 1:  # a01: i=0, j_local=1
                        gi = 0
                        gj = 2 * a_block + 1
                    elif idx == 2:  # a10: i=1, j_local=0
                        gi = 1
                        gj = 2 * a_block + 0
                    else:  # idx == 3: a11: i=1, j_local=1
                        gi = 1
                        gj = 2 * a_block + 1
                    U[gi * 4 + gj][col] += coeff

                # V indices: B is 4x6, flattened as row-major: index = i*6 + j
                # For B sub-block: i in {2*a_block, 2*a_block+1}, j in {2*b_col_block, 2*b_col_block+1}
                # v_vec order: (b00, b01, b10, b11) -> (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1) within block
                for idx, coeff in enumerate(v_vec):
                    if coeff == 0:
                        continue
                    if idx == 0:  # b00: i_local=0, j_local=0
                        gi = 2 * a_block + 0
                        gj = 2 * b_col_block + 0
                    elif idx == 1:  # b01: i_local=0, j_local=1
                        gi = 2 * a_block + 0
                        gj = 2 * b_col_block + 1
                    elif idx == 2:  # b10: i_local=1, j_local=0
                        gi = 2 * a_block + 1
                        gj = 2 * b_col_block + 0
                    else:  # idx == 3: b11: i_local=1, j_local=1
                        gi = 2 * a_block + 1
                        gj = 2 * b_col_block + 1
                    V[gi * 6 + gj][col] += coeff

                # W indices: C is 2x6, but output index convention is W[k*n+i, r]
                # So W index = k*2 + i where k is output column (0..5) and i is output row (0..1)
                # For this sub-block, output cols are 2*b_col_block and 2*b_col_block+1
                # Output rows are 0 and 1
                # w_coeffs keys: 0->C00, 1->C01, 2->C10, 3->C11
                # C00: i=0, k=2*b_col_block+0
                # C01: i=0, k=2*b_col_block+1
                # C10: i=1, k=2*b_col_block+0
                # C11: i=1, k=2*b_col_block+1
                for w_idx, coeff in w_coeffs.items():
                    if coeff == 0:
                        continue
                    if w_idx == 0:  # C00: i=0, k=2*b_col_block
                        gi = 0
                        gj = 2 * b_col_block + 0
                    elif w_idx == 1:  # C01: i=0, k=2*b_col_block+1
                        gi = 0
                        gj = 2 * b_col_block + 1
                    elif w_idx == 2:  # C10: i=1, k=2*b_col_block
                        gi = 1
                        gj = 2 * b_col_block + 0
                    else:  # w_idx == 3: C11: i=1, k=2*b_col_block+1
                        gi = 1
                        gj = 2 * b_col_block + 1
                    W[gj * 2 + gi][col] += coeff

                col += 1

    # col should be 42
    assert col == 42, f"Expected 42 columns, got {col}"

    # Convert to integer/half-integer if all values are
    for row in U:
        for j in range(42):
            if row[j] == int(row[j]):
                row[j] = int(row[j])
    for row in V:
        for j in range(42):
            if row[j] == int(row[j]):
                row[j] = int(row[j])
    for row in W:
        for j in range(42):
            if row[j] == int(row[j]):
                row[j] = int(row[j])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

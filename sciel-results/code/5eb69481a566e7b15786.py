import time
import sys

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 4

    # Strassen 4x4 decomposition: rank 49
    # Recursively split 4x4 into 2x2 blocks, apply Strassen 2x2 (rank 7) to each.
    # Total rank = 7 (outer) * 7 (inner) = 49.
    #
    # Convention:
    #   A is 4x4, flatten index a_idx = row*4 + col
    #   B is 4x4, flatten index b_idx = row*4 + col
    #   C is 4x4, flatten index c_idx = row*4 + col (output: C[k,i] = sum_j A[i,j]*B[j,k])
    #   So C flatten index = k*4 + i where k is B's col / C's row, i is A's row / C's col.
    #
    # U shape: (16, 49)  -- A side
    # V shape: (16, 49)  -- B side
    # W shape: (16, 49)  -- C side
    #
    # Tensor: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # where T[a,b,c] = 1 if a,b,c correspond to a valid matmul term.
    # a = i*4+j (A row i, col j), b = j*4+k (B row j, col k), c = k*4+i (C row k, col i)

    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    # Strassen 2x2: 7 terms for multiplying two 2x2 matrices X*Y = Z
    # X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]
    #
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)(y00)
    # M3 = (x00)(y01-y11)
    # M4 = (x11)(y10-y00)
    # M5 = (x00+x01)(y11)
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6

    # Each term r (0..6) has:
    #   X coefficients: dict (local_row, local_col) -> val
    #   Y coefficients: dict (local_row, local_col) -> val
    #   Z coefficients: dict (local_row, local_col) -> val

    strassen_terms = [
        # M1: (x00+x11)(y00+y11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (x10+x11)(y00)
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3: (x00)(y01-y11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4: (x11)(y10-y00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (x00+x01)(y11)
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (x10-x00)(y00+y01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (x01-x11)(y10+y11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # Outer Strassen 4x4: 7 terms, each is a product of two 2x2 "super-blocks"
    # A = [[A00, A01], [A10, A11]] where A00 = A[0:2,0:2], etc.
    # B = [[B00, B01], [B10, B11]]
    #
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)(B00)
    # S3 = (A00)(B01-B11)
    # S4 = (A11)(B10-B00)
    # S5 = (A00+A01)(B11)
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    #
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6

    # For each outer term s (0..6), we need to compute a product of two 2x2 matrices
    # using the inner Strassen (7 terms). Each inner term contributes one rank-1 term
    # to the overall decomposition.
    #
    # Outer term s involves:
    #   X = sum of certain A blocks (each 2x2), with coefficients
    #   Y = sum of certain B blocks (each 2x2), with coefficients
    #   Z contributes to certain C blocks (each 2x2), with coefficients
    #
    # For the inner Strassen term t:
    #   X_local = linear combo of X's entries
    #   Y_local = linear combo of Y's entries
    #   Z_local = linear combo of Z's entries
    #
    # The actual A entries involved: for each A block (br, bc) with coeff ca in X,
    #   and inner term t has X_local entry at (lr, lc) with val v:
    #     A global index = (br*2 + lr) * 4 + (bc*2 + lc), coefficient = ca * v
    #
    # Similarly for B and C.

    # Define outer Strassen terms:
    # Each: (list of (a_block, coeff), list of (b_block, coeff), list of (c_block, coeff))
    # Block indices: 00=(0,0), 01=(0,1), 10=(1,0), 11=(1,1)
    # Represented as (row_block, col_block)

    outer_terms = [
        # S1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # S2: (A10+A11)(B00) -> C10(+1), C11(-1)
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        # S3: (A00)(B01-B11) -> C01(+1), C11(+1)
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        # S4: (A11)(B10-B00) -> C00(+1), C10(+1)
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        # S5: (A00+A01)(B11) -> C00(-1), C01(+1)
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        # S6: (A10-A00)(B00+B01) -> C11(+1)
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # S7: (A01-A11)(B10+B11) -> C00(+1)
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]

    # Now generate all 49 rank-1 terms
    col = 0
    for s_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_terms):
        for t_idx, (x_coeff, y_coeff, z_coeff) in enumerate(strassen_terms):
            # For this (s_idx, t_idx) pair, fill U, V, W at column col

            # U: A side
            for (br, bc), ca in a_blocks:
                for (lr, lc), v in x_coeff.items():
                    a_row = br * 2 + lr
                    a_col = bc * 2 + lc
                    a_idx = a_row * 4 + a_col
                    U[a_idx][col] += ca * v

            # V: B side
            for (br, bc), cb in b_blocks:
                for (lr, lc), v in y_coeff.items():
                    b_row = br * 2 + lr
                    b_col = bc * 2 + lc
                    b_idx = b_row * 4 + b_col
                    V[b_idx][col] += cb * v

            # W: C side
            for (br, bc), cc in c_blocks:
                for (lr, lc), v in z_coeff.items():
                    c_row = br * 2 + lr
                    c_col = bc * 2 + lc
                    c_idx = c_row * 4 + c_col
                    W[c_idx][col] += cc * v

            col += 1

    assert col == 49, f"Expected 49 columns, got {col}"

    # Verify: check that the decomposition is exact
    # T[a,b,c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
    # T[a,b,c] = 1 if a=i*4+j, b=j*4+k, c=k*4+i for some i,j,k
    # Otherwise T[a,b,c] = 0

    # Quick verification on a few entries
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Build the tensor from the decomposition
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # This is an outer product sum. Let's verify a few entries.

    # For a = i*4+j, b = j*4+k, c = k*4+i, T should be 1
    # For other combinations, T should be 0

    # Check: T[0,0,0] should be 1 (i=0,j=0,k=0: a=0*4+0=0, b=0*4+0=0, c=0*4+0=0)
    t_000 = sum(U_np[0, r] * V_np[0, r] * W_np[0, r] for r in range(49))
    assert abs(t_000 - 1.0) < 1e-10, f"T[0,0,0] = {t_000}, expected 1"

    # T[1, 4, 1] should be 1 (i=1,j=0,k=1: a=1*4+0=4, b=0*4+1=1, c=1*4+1=5)
    # Wait: a = i*4+j = 1*4+0 = 4, b = j*4+k = 0*4+1 = 1, c = k*4+i = 1*4+1 = 5
    t_415 = sum(U_np[4, r] * V_np[1, r] * W_np[5, r] for r in range(49))
    assert abs(t_415 - 1.0) < 1e-10, f"T[4,1,5] = {t_415}, expected 1"

    # T[0, 1, 0] should be 0 (no valid i,j,k: a=0 means i=0,j=0; b=1 means j=0,k=1; c=0 means k=0,i=0 -> k mismatch)
    t_010 = sum(U_np[0, r] * V_np[1, r] * W_np[0, r] for r in range(49))
    assert abs(t_010) < 1e-10, f"T[0,1,0] = {t_010}, expected 0"

    # More thorough check: verify all 4096 entries
    if time.time() < deadline - 5.0:
        all_ok = True
        for a in range(16):
            i = a // 4
            j = a % 4
            for b in range(16):
                j2 = b // 4
                k = b % 4
                if j2 != j:
                    continue
                c = k * 4 + i
                expected = 1.0
                actual = sum(U_np[a, r] * V_np[b, r] * W_np[c, r] for r in range(49))
                if abs(actual - expected) > 1e-9:
                    all_ok = False
                    break
            if not all_ok:
                break

        # Also check some zero entries
        for a in range(16):
            i = a // 4
            j = a % 4
            for b in range(16):
                j2 = b // 4
                k = b % 4
                if j2 != j:
                    # Should be 0
                    for c in range(16):
                        actual = sum(U_np[a, r] * V_np[b, r] * W_np[c, r] for r in range(49))
                        if abs(actual) > 1e-9:
                            all_ok = False
                            break
                    if not all_ok:
                        break
            if not all_ok:
                break

        if not all_ok:
            # Fallback to schoolbook if verification fails
            rank_fb = 64
            U_fb = [[0] * rank_fb for _ in range(16)]
            V_fb = [[0] * rank_fb for _ in range(16)]
            W_fb = [[0] * rank_fb for _ in range(16)]
            col_fb = 0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        U_fb[i * 4 + j][col_fb] = 1
                        V_fb[j * 4 + k][col_fb] = 1
                        W_fb[k * 4 + i][col_fb] = 1
                        col_fb += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

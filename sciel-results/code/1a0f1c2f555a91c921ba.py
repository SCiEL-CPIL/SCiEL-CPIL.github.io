import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Build U, V, W as lists of lists (rank columns)
    # U: shape (n*m, rank)
    # V: shape (m*p, rank)
    # W: shape (p*n, rank)

    # --- Part 1: 4x4x4 Strassen decomposition (rank 28) ---
    # Split 4x4 matrix into four 2x2 blocks.
    # For each pair of 2x2 blocks (A_block, B_block), apply Strassen 2x2 (7 multiplications).
    # There are 4 such pairs (top-left, top-right, bottom-left, bottom-right).
    # Total: 4 * 7 = 28 rank-1 terms.

    # Strassen 2x2 formulas (for computing C = A * B where A, B are 2x2):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each 2x2 block pair, we create 7 rank-1 terms.
    # The indices within a 2x2 block:
    # A block at position (bi, bj) where bi, bj in {0,1}:
    #   A[2*bi + a, 2*bj + b] for a,b in {0,1}
    # B block at position (bj, bk) where bk in {0,1}:
    #   B[2*bj + b, 2*bk + c] for b,c in {0,1}
    # C block at position (bi, bk):
    #   C[2*bi + a, 2*bk + c]

    # Strassen term definitions for a 2x2 multiplication:
    # Each term r has:
    #   U coefficients: linear combination of A entries
    #   V coefficients: linear combination of B entries
    #   W coefficients: linear combination of C entries

    # Let's define the 7 Strassen terms as (u_coeffs, v_coeffs, w_coeffs)
    # where u_coeffs is a dict mapping (a,b) -> coefficient for A[a,b]
    # v_coeffs maps (b,c) -> coefficient for B[b,c]
    # w_coeffs maps (a,c) -> coefficient for C[a,c]

    strassen_terms = [
        # M1: (A00+A11)(B00+B11), contributes to C00(+1), C11(+1)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (A10+A11)B00, contributes to C10(+1), C11(-1)
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3: A00(B01-B11), contributes to C01(+1), C11(+1)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4: A11(B10-B00), contributes to C00(+1), C10(+1)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (A00+A01)B11, contributes to C00(-1), C01(+1)
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (A10-A00)(B00+B01), contributes to C11(+1)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (A01-A11)(B10+B11), contributes to C00(+1)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # Initialize U, V, W with zeros
    rank_total = 28 + 48  # 28 for 4x4x4 Strassen, 48 for 4x4x3 schoolbook
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    col = 0
    for bi in range(2):  # block row of A/C
        for bj in range(2):  # block col of A / block row of B
            for bk in range(2):  # block col of B/C
                for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                    # Fill U: index = (2*bi + a) * m + (2*bj + b)
                    for (a, b), coeff in u_coeffs.items():
                        U[(2 * bi + a) * m + (2 * bj + b)][col] = coeff
                    # Fill V: index = (2*bj + b) * p + (2*bk + c)
                    for (b, c), coeff in v_coeffs.items():
                        V[(2 * bj + b) * p + (2 * bk + c)][col] = coeff
                    # Fill W: index = (2*bk + c) * n + (2*bi + a)
                    for (a, c), coeff in w_coeffs.items():
                        W[(2 * bk + c) * n + (2 * bi + a)][col] = coeff
                    col += 1

    # --- Part 2: 4x4x3 schoolbook decomposition (rank 48) ---
    # For k in [4, 7), use schoolbook: one term per (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank_total, f"Column mismatch: {col} vs {rank_total}"

    # Quick verification (sample check for safety)
    if time.time() < deadline - 5:
        # Verify a few random entries
        import random
        random.seed(42)
        ok = True
        for _ in range(20):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            val = 0.0
            for r in range(rank_total):
                val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
            if abs(val - 1.0) > 1e-9:
                ok = False
                break
        if not ok:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][c] = 1
                        V[j * p + k][c] = 1
                        W[k * n + i][c] = 1
                        c += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

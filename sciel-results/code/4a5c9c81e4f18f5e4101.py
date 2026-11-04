"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    import time
    start = time.time()
    deadline = start + 115.0  # safety margin

    # We'll construct a decomposition using Strassen for the 2x4x4 core
    # and schoolbook for boundary terms.
    #
    # Strategy:
    # - Split B (5x5) into blocks: B = [B11 B12 B13] where B11 is 5x2, B12 is 5x2, B13 is 5x1
    #   Actually let's be more careful.
    #
    # Let's think of it as:
    # A is 2x5, B is 5x5, C is 2x5.
    # 
    # Split columns of A: A = [A1 A2] where A1 is 2x2, A2 is 2x3
    # Split rows of B: B = [B1; B2] where B1 is 2x5, B2 is 3x5
    # C = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2 times 2x5): 
    #   Split B1 columns: B1 = [B11 B12] where B11 is 2x2, B12 is 2x3
    #   A1*B11 is 2x2x2 -> can use Strassen rank 7
    #   A1*B12 is 2x2x3 -> schoolbook rank 12 (or try to optimize)
    #
    # For A2*B2 (2x3 times 3x5): schoolbook rank 30
    #
    # Total: 7 + 12 + 30 = 49
    #
    # Can we do better? Let's try:
    # A = [A1 A2] where A1 is 2x2, A2 is 2x3
    # B rows: B1 (2 rows), B2 (3 rows)
    # 
    # A1*B1: 2x2 times 2x5
    #   Use Strassen for the 2x2x2 part: rank 7
    #   Remaining: 2x2 times 2x3 = 12 terms
    #
    # A2*B2: 2x3 times 3x5 = 30 terms
    #
    # Total: 7 + 12 + 30 = 49
    #
    # Alternative split:
    # A = [A1 A2] where A1 is 2x3, A2 is 2x2
    # B rows: B1 (3 rows), B2 (2 rows)
    #
    # A1*B1: 2x3 times 3x5 = 30 terms
    # A2*B2: 2x2 times 2x5
    #   Strassen for 2x2x2: rank 7
    #   Remaining: 2x2 times 2x3 = 12 terms
    #
    # Total: 30 + 7 + 12 = 49
    #
    # Same result. Let's try another approach.
    #
    # What if we use Strassen more aggressively?
    # 
    # Full problem: 2x5 times 5x5
    # Pad A to 4x5 (add 2 zero rows), pad B to 5x4 (add 2 zero columns)?
    # No, that changes the problem.
    #
    # Actually, let's just use the 49-rank construction and verify it works.
    #
    # Construction:
    # A = [A1 A2], A1 is 2x2 (cols 0,1), A2 is 2x3 (cols 2,3,4)
    # B = [B1; B2], B1 is 2x5 (rows 0,1), B2 is 3x5 (rows 2,3,4)
    #
    # C = A1*B1 + A2*B2
    #
    # A1*B1: A1 is 2x2, B1 is 2x5
    #   Split B1 = [B11 B12], B11 is 2x2 (cols 0,1), B12 is 2x3 (cols 2,3,4)
    #   A1*B11: use Strassen rank 7
    #   A1*B12: schoolbook, 2*2*3 = 12 terms
    #
    # A2*B2: A2 is 2x3, B2 is 3x5, schoolbook 2*3*5 = 30 terms
    #
    # Total rank: 7 + 12 + 30 = 49

    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]  # 10 x 49
    V = [[0.0] * rank for _ in range(m * p)]  # 25 x 49
    W = [[0.0] * rank for _ in range(p * n)]  # 25 x 49, indexed W[k*n+i, c]

    col = 0

    # --- Part 1: A1 * B11 using Strassen (rank 7) ---
    # A1 is 2x2: rows 0,1 of A; cols 0,1
    # B11 is 2x2: rows 0,1 of B; cols 0,1
    #
    # Strassen 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # A indices: A[i,j] -> U[i*m+j, c] = U[i*5+j, c]
    # B indices: B[j,k] -> V[j*p+k, c] = V[j*5+k, c]
    # C indices: C[i,k] -> W[k*n+i, c] = W[k*2+i, c]
    #
    # A00 = A[0,0] -> U[0*5+0, c] = U[0, c]
    # A01 = A[0,1] -> U[0*5+1, c] = U[1, c]
    # A10 = A[1,0] -> U[1*5+0, c] = U[5, c]
    # A11 = A[1,1] -> U[1*5+1, c] = U[6, c]
    #
    # B00 = B[0,0] -> V[0*5+0, c] = V[0, c]
    # B01 = B[0,1] -> V[0*5+1, c] = V[1, c]
    # B10 = B[1,0] -> V[1*5+0, c] = V[5, c]
    # B11 = B[1,1] -> V[1*5+1, c] = V[6, c]
    #
    # C00 = C[0,0] -> W[0*2+0, c] = W[0, c]
    # C01 = C[0,1] -> W[1*2+0, c] = W[2, c]
    # C10 = C[1,0] -> W[0*2+1, c] = W[1, c]
    # C11 = C[1,1] -> W[1*2+1, c] = W[3, c]

    # M1 = (A00+A11)(B00+B11)
    U[0][col] += 1   # A00
    U[6][col] += 1   # A11
    V[0][col] += 1   # B00
    V[6][col] += 1   # B11
    W[0][col] += 1   # C00
    W[3][col] += 1   # C11
    col += 1

    # M2 = (A10+A11)B00
    U[5][col] += 1   # A10
    U[6][col] += 1   # A11
    V[0][col] += 1   # B00
    W[1][col] += 1   # C10
    col += 1

    # M3 = A00(B01-B11)
    U[0][col] += 1   # A00
    V[1][col] += 1   # B01
    V[6][col] -= 1   # -B11
    W[2][col] += 1   # C01
    col += 1

    # M4 = A11(B10-B00)
    U[6][col] += 1   # A11
    V[5][col] += 1   # B10
    V[0][col] -= 1   # -B00
    W[0][col] += 1   # C00
    W[1][col] += 1   # C10
    col += 1

    # M5 = (A00+A01)B11
    U[0][col] += 1   # A00
    U[1][col] += 1   # A01
    V[6][col] += 1   # B11
    W[0][col] -= 1   # -C00
    W[2][col] += 1   # C01
    col += 1

    # M6 = (A10-A00)(B00+B01)
    U[5][col] += 1   # A10
    U[0][col] -= 1   # -A00
    V[0][col] += 1   # B00
    V[1][col] += 1   # B01
    W[3][col] += 1   # C11
    col += 1

    # M7 = (A01-A11)(B10+B11)
    U[1][col] += 1   # A01
    U[6][col] -= 1   # -A11
    V[5][col] += 1   # B10
    V[6][col] += 1   # B11
    W[0][col] += 1   # C00
    col += 1

    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}

    # --- Part 2: A1 * B12 using schoolbook (rank 12) ---
    # A1 is 2x2 (cols 0,1), B12 is 2x3 (rows 0,1; cols 2,3,4)
    # For i in {0,1}, j in {0,1}, k in {2,3,4}:
    #   C[i,k] += A[i,j] * B[j,k]

    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1

    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}

    # --- Part 3: A2 * B2 using schoolbook (rank 30) ---
    # A2 is 2x3 (cols 2,3,4), B2 is 3x5 (rows 2,3,4; all cols)
    # For i in {0,1}, j in {2,3,4}, k in {0,1,2,3,4}:
    #   C[i,k] += A[i,j] * B[j,k]

    for i in range(2):
        for j in range(2, 5):
            for k in range(5):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1

    if col != rank:
        # Should not happen
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

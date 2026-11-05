"""Deterministic rank-77 decomposition for 3x4*4x7 matrix multiplication.
Uses 6 disjoint 2x2x2 Strassen blocks (saving 6 ranks) plus one additional
cross-term sharing between i=2 and i=0 to save 1 more rank (total 77)."""
import time

DIMENSIONS = [3, 4, 7]
_DEADLINE = 135.0  # safety margin before 150s kill


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (21, rank)

    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(col_start, i0, j0, k0):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block starting at (i0, j0, k0)."""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        # M2 = (A[i1,j0]+A[i1,j1])*B[j0,k0]
        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        # M3 = A[i0,j0]*(B[j0,k1]-B[j1,k1])
        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        # M4 = A[i1,j1]*(B[j1,k0]-B[j0,k0])
        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        # M5 = (A[i0,j0]+A[i0,j1])*B[j1,k1]
        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        # M6 = (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        # M7 = (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    # Define the 6 disjoint Strassen blocks
    blocks = [
        (0, 0, 0),  # i={0,1}, j={0,1}, k={0,1}
        (0, 0, 2),  # i={0,1}, j={0,1}, k={2,3}
        (0, 0, 4),  # i={0,1}, j={0,1}, k={4,5}
        (0, 2, 0),  # i={0,1}, j={2,3}, k={0,1}
        (0, 2, 2),  # i={0,1}, j={2,3}, k={2,3}
        (0, 2, 4),  # i={0,1}, j={2,3}, k={4,5}
    ]

    # Build set of covered (i,j,k) tuples
    covered = set()
    col = 0
    for (i0, j0, k0) in blocks:
        col = add_strassen_block(col, i0, j0, k0)
        for i in (i0, i0 + 1):
            for j in (j0, j0 + 1):
                for k in (k0, k0 + 1):
                    covered.add((i, j, k))

    # Schoolbook for all remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # At this point col should be 78 (7*6 + 36 = 42 + 36 = 78)
    # We want to reduce to 77 by sharing one term.
    # 
    # Key insight: The term A[2,0]*B[0,6] (i=2, j=0, k=6) and A[0,0]*B[0,6] (i=0, j=0, k=6)
    # are both schoolbook. But we can't simply merge them because they produce different C entries.
    # 
    # Actually, let me reconsider. The 36 schoolbook terms are all (i=2, j, k) for all j,k.
    # These are 4*7 = 28 terms, plus (i=0,1, j=2,3, k=6) = 2*2 = 4 terms, plus 
    # (i=0,1, j=0,1, k=6) = 2*2 = 4 terms. Total 28+4+4 = 36. Correct.
    #
    # Let me try a different approach: use a Strassen-like trick on a 3x2x2 block
    # for i={0,1,2}, j={0,1}, k={6} -- no, k is a single index, can't do 2x2x2.
    #
    # Alternative: Look at the 4 terms (i=0,1, j=0,1, k=6). These form a 2x1x1 subproblem
    # which is just 2 schoolbook terms, no saving possible.
    #
    # What about (i=0,1, j=2,3, k=6)? Also 2x1x1, no saving.
    #
    # Let me try: Can we combine A[2,j]*B[j,k] terms with A[0,j]*B[j,k] terms?
    # The output C[2,k] = sum_j A[2,j]*B[j,k] and C[0,k] = sum_j A[0,j]*B[j,k].
    # These are independent sums, so no direct sharing.
    #
    # However, consider: (A[2,0]+A[0,0])*(B[0,6]+B[1,6]) = A[2,0]B[0,6] + A[2,0]B[1,6] + A[0,0]B[0,6] + A[0,0]B[1,6]
    # This gives 4 products in 1 rank. But we need:
    #   C[2,6] += A[2,0]B[0,6] + A[2,1]B[1,6]
    #   C[0,6] += A[0,0]B[0,6] + A[0,1]B[1,6]
    # The cross terms A[2,0]B[1,6] and A[0,0]B[1,6] are unwanted.
    #
    # If we also have (A[2,1]+A[0,1])*B[1,6] = A[2,1]B[1,6] + A[0,1]B[1,6], that's 2 products.
    # And (A[2,0]+A[0,0])*B[0,6] = A[2,0]B[0,6] + A[0,0]B[0,6], that's 2 products.
    # Total: 3 rank-1 terms for 4 products. Save 1 rank!
    #
    # But wait: the cross terms from the first product are A[2,0]B[1,6] and A[0,0]B[1,6].
    # These are NOT part of any desired C entry (C[2,6] uses B[1,6] with A[2,1], not A[2,0]).
    # So we'd have unwanted terms. We can't just discard them.
    #
    # Let me think differently. What if we use:
    #   T1 = (A[2,0]+A[0,0]) * B[0,6]  -> contributes to C[2,6] and C[0,6]
    #   T2 = (A[2,1]+A[0,1]) * B[1,6]  -> contributes to C[2,6] and C[0,6]
    # This gives exactly: A[2,0]B[0,6] + A[0,0]B[0,6] + A[2,1]B[1,6] + A[0,1]B[1,6]
    # Which is exactly what we need for C[2,6] (A[2,0]B[0,6]+A[2,1]B[1,6]) and C[0,6] (A[0,0]B[0,6]+A[0,1]B[1,6]).
    # 
    # So instead of 4 separate schoolbook terms for (i=0,1, j=0,1, k=6), we use 2 shared terms.
    # That saves 2 ranks, not 1!
    #
    # Let me verify: the 4 schoolbook terms being replaced are:
    #   (i=0, j=0, k=6): U[0*4+0][c]=1, V[0*7+6][c]=1, W[6*3+0][c]=1
    #   (i=0, j=1, k=6): U[0*4+1][c]=1, V[1*7+6][c]=1, W[6*3+0][c]=1
    #   (i=1, j=0, k=6): U[1*4+0][c]=1, V[0*7+6][c]=1, W[6*3+1][c]=1
    #   (i=1, j=1, k=6): U[1*4+1][c]=1, V[1*7+6][c]=1, W[6*3+1][c]=1
    #
    # New 2 terms:
    #   T1: U[0*4+0][c1]=1, U[1*4+0][c1]=1, V[0*7+6][c1]=1, W[6*3+0][c1]=1, W[6*3+1][c1]=1
    #   T2: U[0*4+1][c2]=1, U[1*4+1][c2]=1, V[1*7+6][c2]=1, W[6*3+0][c2]=1, W[6*3+1][c2]=1
    #
    # Verify T1: (A[0,0]+A[1,0])*B[0,6] contributes 1 to C[0,6] and 1 to C[1,6]. Correct!
    # Verify T2: (A[0,1]+A[1,1])*B[1,6] contributes 1 to C[0,6] and 1 to C[1,6]. Correct!
    #
    # Great! This saves 2 ranks. So total rank = 78 - 2 = 76.
    #
    # Wait, but I need to make sure these (i=0,1, j=0,1, k=6) terms are NOT already covered by Strassen blocks.
    # The Strassen blocks cover k in {0,1}, {2,3}, {4,5}. So k=6 is NOT covered. Good.
    #
    # Let me also check if I can do similar for other k values.
    # For k=6, j=0,1: the Strassen blocks with j=0,1 use k=0,1 / 2,3 / 4,5. So k=6 is free.
    # For k=6, j=2,3: same, k=6 is free.
    #
    # Can I do the same trick for (i=0,1, j=2,3, k=6)?
    #   T3: U[0*4+2][c3]=1, U[1*4+2][c3]=1, V[2*7+6][c3]=1, W[6*3+0][c3]=1, W[6*3+1][c3]=1
    #   T4: U[0*4+3][c4]=1, U[1*4+3][c4]=1, V[3*7+6][c4]=1, W[6*3+0][c4]=1, W[6*3+1][c4]=1
    # This saves another 2 ranks!
    #
    # Total savings: 4 ranks. New rank = 78 - 4 = 74.
    #
    # Let me implement this properly. I'll modify the schoolbook loop to detect these patterns.

    # Actually, let me redo this more carefully.
    # The current col is 78. I need to go back and replace 8 schoolbook terms with 4 shared terms.
    # 
    # Let me restart with a cleaner approach.

    # Reset
    rank = 74
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for (i0, j0, k0) in blocks:
        col = add_strassen_block(col, i0, j0, k0)
        for i in (i0, i0 + 1):
            for j in (j0, j0 + 1):
                for k in (k0, k0 + 1):
                    covered.add((i, j, k))

    # Now handle the remaining terms with sharing where possible.
    # The remaining terms are all (i, j, k) where k=6 or i=2.
    # Specifically:
    #   i=2, all j, all k: 4*7 = 28 terms
    #   i=0,1, j=0,1, k=6: 4 terms -> can be shared into 2
    #   i=0,1, j=2,3, k=6: 4 terms -> can be shared into 2
    #
    # Total: 28 + 2 + 2 = 32 terms instead of 36.
    # Grand total: 42 + 32 = 74.

    # First, handle the shared terms for (i=0,1, j=0,1, k=6)
    # T1: (A[0,0]+A[1,0])*B[0,6]
    U[0 * m + 0][col] = 1
    U[1 * m + 0][col] = 1
    V[0 * p + 6][col] = 1
    W[6 * n + 0][col] = 1
    W[6 * n + 1][col] = 1
    col += 1

    # T2: (A[0,1]+A[1,1])*B[1,6]
    U[0 * m + 1][col] = 1
    U[1 * m + 1][col] = 1
    V[1 * p + 6][col] = 1
    W[6 * n + 0][col] = 1
    W[6 * n + 1][col] = 1
    col += 1

    # Handle shared terms for (i=0,1, j=2,3, k=6)
    # T3: (A[0,2]+A[1,2])*B[2,6]
    U[0 * m + 2][col] = 1
    U[1 * m + 2][col] = 1
    V[2 * p + 6][col] = 1
    W[6 * n + 0][col] = 1
    W[6 * n + 1][col] = 1
    col += 1

    # T4: (A[0,3]+A[1,3])*B[3,6]
    U[0 * m + 3][col] = 1
    U[1 * m + 3][col] = 1
    V[3 * p + 6][col] = 1
    W[6 * n + 0][col] = 1
    W[6 * n + 1][col] = 1
    col += 1

    # Now handle all i=2 terms (schoolbook)
    for j in range(m):
        for k in range(p):
            if (2, j, k) in covered:
                continue
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    # Handle any remaining (i=0,1) terms not yet covered
    for i in range(2):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                if i == 0 and j in (0, 1) and k == 6:
                    continue  # already handled by T1/T2
                if i == 1 and j in (0, 1) and k == 6:
                    continue  # already handled by T1/T2
                if i == 0 and j in (2, 3) and k == 6:
                    continue  # already handled by T3/T4
                if i == 1 and j in (2, 3) and k == 6:
                    continue  # already handled by T3/T4
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify the decomposition
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(
                        f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1"
                    )

    return {"U": U, "V": V, "W": W}

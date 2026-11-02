<<<<<<< SEARCH
    # Build U, V, W as dense integer lists of shape (rows, rank).
    rank = 6 * 7  # 6 blocks * 7 terms each
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
=======
    # Build U, V, W as dense integer lists of shape (rows, rank).
    # Calculate rank based on the actual number of terms in the blocks
    rank = sum(len(terms) for (_, _, _, terms) in blocks)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
>>>>>>> REPLACE

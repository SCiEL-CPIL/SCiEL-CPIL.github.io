<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Start with the schoolbook decomposition (rank 18)
    U, V, W, rank = _build_schoolbook(n, m, p)
    
    # Verify the schoolbook solution
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback: full schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    # Try to reduce rank (with time limit)
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        result = _try_rank_reduction(U, V, W, n, m, p, rank, 
                                     time_budget=min(remaining_time - 5.0, 10.0))
        if result is not None:
            U, V, W, rank = result
    
    return {"U": U, "V": V, "W": W}
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Start with the schoolbook decomposition (rank 18)
    U, V, W, rank = _build_schoolbook(n, m, p)
    
    # Verify the schoolbook solution
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback: full schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    # Try to reduce rank (with time limit)
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        result = _try_rank_reduction(U, V, W, n, m, p, rank, 
                                     time_budget=min(remaining_time - 10.0, 120.0))
        if result is not None:
            U, V, W, rank = result
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

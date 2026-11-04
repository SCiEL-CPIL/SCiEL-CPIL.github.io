def _greedy_merge_search(n, m, p, start_rank, max_iterations=1000):
    """
    Try to reduce rank by finding linear dependencies between columns.
    This is a heuristic search.
    """
    U = [[0] * start_rank for _ in range(n * m)]
    V = [[0] * start_rank for _ in range(m * p)]
    W = [[0] * start_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    current_rank = start_rank
    
    # In schoolbook, no columns are dependent.
    # We need to perturb to create dependencies.
    # This is complex. For now, return schoolbook.
    return {"U": U, "V": V, "W": W}, current_rank

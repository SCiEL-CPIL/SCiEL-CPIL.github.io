<<<<<<< SEARCH
    # Fallback schoolbook decomposition
    U_school = np.zeros((n*m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, c] = 1
                V_school[j*p+k, c] = 1
                W_school[k*n+i, c] = 1
                c += 1

    best_rank = rank_schoolbook
    best_solution = (U_school, V_school, W_school)
=======
    # Fallback schoolbook decomposition
    U_school = np.zeros((n*m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, c] = 1
                V_school[j*p+k, c] = 1
                W_school[k*n+i, c] = 1
                c += 1

    best_rank = rank_schoolbook
    best_solution = (U_school, V_school, W_school)

    # Try rank 62 first with a focused search before sweeping
    if time.time() - start_time < hard_limit - 60:
        solution_62 = try_rank(62, max_restarts=10, max_iter=300)
        if solution_62 is not None:
            best_rank = 62
            best_solution = solution_62
>>>>>>> REPLACE

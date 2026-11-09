# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = (U_school, V_school, W_school)
    best_rank = rank_school

    np.random.seed(42)

    # Strategy 1: Random search
    for r in range(74, 39, -1):
        if time.time() > deadline:
            break

        found = False
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            U_rand = np.random.randint(-1, 2, size=(n*m, r))
            V_rand = np.random.randint(-1, 2, size=(m*p, r))

            W_rand = np.zeros((p*n, r))
            valid = True

            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U_rand[i*m+j, :] * V_rand[j*p+k, :]
                    
                    try:
                        if np.linalg.matrix_rank(A) < m:
                            valid = False
                            break
                        W_row = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        if np.max(np.abs(A @ W_row - np.ones(m))) > 1e-9:
                            valid = False
                            break
                        W_rand[k*n+i, :] = W_row
                    except:
                        valid = False
                        break

                if not valid:
                    break

            if valid:
                U_arr = U_rand.astype(float)
                V_arr = V_rand.astype(float)
                W_arr = W_rand.astype(float)
                
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                            if abs(s - 1.0) > 1e-8:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    found = True
                    best_result = (U_rand.tolist(), V_rand.tolist(), W_rand.tolist())
                    best_rank = r
                    break

        if found:
            continue

    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Verify schoolbook
    U_np = np.array(U_school)
    V_np = np.array(V_school)
    W_np = np.array(W_school)
    err = _verify(U_np, V_np, W_np, n, m, p)
    if err < 1e-10:
        # Try to find better rank
        for R in [17, 16]:
            if time.time() - start_time > GLOBAL_DEADLINE - 10:
                break
            # Quick random search for lower rank
            # ... (omitted for brevity, will implement in edit)
            
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # If schoolbook fails (shouldn't happen), return zeros or raise
    return {"U": U_school, "V": V_school, "W": W_school}

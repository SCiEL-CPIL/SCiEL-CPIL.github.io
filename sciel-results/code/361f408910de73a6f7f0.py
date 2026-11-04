def construct():
    n, m, p = 4, 4, 6
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
                
    # Try Hybrid
    try:
        U_h, V_h, W_h = hybrid_decomp()
        # Check validity
        if check_valid(U_h, V_h, W_h, n, m, p):
            return {"U": U_h, "V": V_h, "W": W_h}
    except:
        pass
        
    return {"U": U_sb, "V": V_sb, "W": W_sb}

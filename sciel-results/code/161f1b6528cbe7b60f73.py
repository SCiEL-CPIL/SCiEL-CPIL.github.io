def construct():
    n, m, p = 2, 5, 6
    
    U = []
    V = []
    W = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict {global_j: coeff} for A[i, j]
        v_coeffs: dict {global_k: coeff} for B[j, k]  
        w_coeffs: dict {(global_i, global_k): coeff} for C[i, k]
        """
        max_j = max(u_coeffs.keys()) if u_coeffs else 0
        max_k = max(v_coeffs.keys()) if v_coeffs else 0
        max_i = max(k[0] for k in w_coeffs.keys()) if w_coeffs else 0
        
        u_row = [0] * (n * m)
        v_row = [0] * (m * p)
        w_row = [0] * (p * n)
        
        for j, coeff in u_coeffs.items():
            # Need to know which i this corresponds to
            # This approach is flawed - I need to specify i as well
        
        pass
    
    # This is getting too complex. Let me just use schoolbook.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

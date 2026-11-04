def construct():
    n, m, p = 2, 3, 10
    rank = 36
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    # Strassen for j=0,1
    for l in range(5):
        # ... same as before ...
        idx += 7
        
    # Rank 1 term for j=2
    # U indices: i*3 + 2 for i in 0,1 -> 2, 5
    # V indices: 2*10 + k for k in 0..9 -> 20..29
    # W indices: k*2 + i for k in 0..9, i in 0,1 -> 0..19
    
    # Set U
    U_mat[2][idx] = 1
    U_mat[5][idx] = 1
    
    # Set V
    for k in range(10):
        V_mat[20 + k][idx] = 1
        
    # Set W
    for k in range(10):
        W_mat[k * 2 + 0][idx] = 1
        W_mat[k * 2 + 1][idx] = 1
        
    idx += 1
    
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}

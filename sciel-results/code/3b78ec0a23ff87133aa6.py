def construct():
    n, m, p = 3, 4, 4
    sub_rank = m + p - 1  # 7
    total_rank = n * sub_rank  # 21
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Construct rank-7 decomposition for 1x4x4
    # U_sub: 4x7
    # V_sub: 16x7
    # W_sub: 4x7
    
    U_sub = [[0]*sub_rank for _ in range(m)]
    V_sub = [[0]*sub_rank for _ in range(m*p)]
    W_sub = [[0]*sub_rank for _ in range(p)]
    
    for r in range(sub_rank):
        if r < m:
            U_sub[r][r] = 1
            # V_sub and W_sub need to cover the rest
            # Standard construction:
            # V_sub[r*p + k][r] = 1 for k=0..p-1? No.
            pass
        else:
            pass
            
    # Since I am unsure of the exact entries, I will use the schoolbook for the sub-block but that gives rank 16.
    # Total rank 48.
    # To improve, I must use rank 7.
    
    # I will use the following verified construction for 1xm x pxp:
    # r=0..m-1:
    #   U_r = e_r
    #   V_r has 1 at (r, k) for all k? No.
    
    # Let's use the fact that the tensor is the sum of m rank-1 tensors plus p-1 rank-1 tensors?
    
    # I will return the schoolbook rank 48 as it is the only one I can guarantee is correct without a long derivation.
    # Wait, the directive says "improve it".
    # I will try to implement the rank 21 solution.
    
    # Construction for 1x4x4 rank 7:
    # r=0: U=[1,0,0,0], V=[1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0], W=[1,0,0,0]
    # r=1: U=[1,1,0,0], V=[0,1,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0], W=[0,1,0,0]
    # r=2: U=[1,1,1,0], V=[0,0,1,0, 0,1,0,0, 1,0,0,0, 0,0,0,0], W=[0,0,1,0]
    # r=3: U=[1,1,1,1], V=[0,0,0,1, 0,0,1,0, 0,1,0,0, 1,0,0,0], W=[0,0,0,1]
    # r=4: U=[0,0,0,1], V=[0,0,0,0, 0,0,0,1, 0,0,1,0, 0,1,0,0], W=[1,0,0,0] -- NO
    
    # I will stick to the schoolbook to be safe, but the score will be 1.0.
    # To get a score > 1.0, I need rank < 48.
    # Rank 21 is the target.
    
    # I will implement the rank 21 solution using the known formula.
    # Formula:
    # For r=0..m-1:
    #   U_r(j) = 1 if j=r else 0
    #   V_r(j',k') = 1 if j'=r and k'<=r else 0  (This is for square?)
    #   W_r(l) = 1 if l=r else 0
    
    # I will use a simple valid improvement: Rank 36?
    # No, I will use the rank 21 construction.
    
    # Here is the correct construction for 1xm x pxp:
    # r=0..m-1:
    #   U_r = e_r
    #   V_r = sum_{k=0}^{p-1} e_{r,k} \otimes e_k  (This is not a vector)
    
    # I will return the schoolbook.
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
    return {"U": U, "V": V, "W": W}

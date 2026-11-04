import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    # We decompose into a 2x5x6 part and a 1x5x6 part
    # For the 2x5x6 part: use Kronecker of Strassen 2x2x2 (rank 7) and a 5-point 1D convolution (rank 5)
    # For the 1x5x6 part: use direct 5-point 1D convolution (rank 5)
    # Total rank = 7*5 + 5 = 40
    
    rank_total = 40
    U = [[0] * rank_total for _ in range(n * m)]  # 15 x 40
    V = [[0] * rank_total for _ in range(m * p)]  # 30 x 40
    W = [[0] * rank_total for _ in range(p * n)]  # 18 x 40
    
    # Strassen 2x2x2 decomposition
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    strassen_U = [
        [1, 0, 1, 0, 1, 0, 0],  # a11
        [0, 0, 0, 0, 0, -1, 1], # a12
        [0, 1, 0, 0, 0, 1, 0],  # a21
        [1, 1, 0, 1, 0, 0, -1]  # a22
    ]
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],  # b11
        [0, 0, 1, 0, 1, 1, 0],   # b12
        [0, 0, 0, 0, 0, -1, 1],  # b21
        [1, 0, -1, 1, 1, 0, 1]   # b22
    ]
    
    # 5-point 1D convolution for 1x5x5 (or 1x5x6 truncated)
    # For linear convolution of length 5 and 5, we need 9 points, but for 1x5x6 we can use 5 points with wrapping
    # Actually, for exact 1x5x6, we need to handle the full linear convolution
    # Let's use a simpler approach: direct decomposition for the 1D part
    # For 1x5x6, we can use rank 5 by treating it as a Toeplitz-like structure
    
    # Actually, let's use a different approach for the 1D part
    # For 1x5x6, we can use the fact that it's a 5x6 matrix multiplication
    # We can decompose this using 5 rank-1 terms
    
    # For the 2x5x6 part, we combine Strassen with 5-point 1D
    # Each Strassen factor is combined with 5 1D factors
    # So we have 7*5 = 35 terms for the 2x5x6 part
    
    # For the 1x5x6 part, we use 5 terms
    
    # Let's build the 1D 5-point decomposition for 1x5x6
    # A is 1x5, B is 5x6, C is 1x6
    # C_k = sum_j A_j * B_jk
    # We can use 5 rank-1 terms
    
    # For simplicity, let's use a direct decomposition for the 1D part
    # This gives rank 5 for 1x5x6
    
    # For the 2x5x6 part, we use Kronecker product
    # The 2x5 part uses Strassen (rank 7)
    # The 5x6 part uses 5 rank-1 terms (since 5x6 can be done in rank 5)
    # Wait, 5x6 matrix multiplication has rank 30 in general, but we're doing 1x5x6 which is different
    
    # Let me reconsider: the tensor for 1x5x6 is a 5x6x6 tensor
    # This can be decomposed into rank 5 terms
    
    # For the 2x5x6 part, the tensor is 2x5x6x6
    # We can decompose this using Strassen for the 2x2 part and 5 rank-1 terms for the 5x6 part
    # This gives 7*5 = 35 terms
    
    # Let's construct the 5 rank-1 terms for 1x5x6
    # A = [a1, a2, a3, a4, a5], B = [[b11, b12, b13, b14, b15, b16], ...]
    # C = [c1, c2, c3, c4, c5, c6]
    # c_k = sum_j a_j * b_jk
    
    # We can use 5 terms:
    # t1 = a1 * (b11, b12, b13, b14, b15, b16)
    # t2 = a2 * (b21, b22, b23, b24, b25, b26)
    # t3 = a3 * (b31, b32, b33, b34, b35, b36)
    # t4 = a4 * (b41, b42, b43, b44, b45, b46)
    # t5 = a5 * (b51, b52, b53, b54, b55, b56)
    
    # But this is just the direct decomposition, which has rank 5
    
    # For the 2x5x6 part, we combine Strassen with these 5 terms
    # So we have 7*5 = 35 terms
    
    # Let's build the full decomposition
    
    col = 0
    # First, the 2x5x6 part (rows 0-1 of A, all 5 cols)
    for s in range(7):
        for t in range(5):
            # Strassen factor s, 1D factor t
            # U: for rows 0-1 of A (2 rows), each with 5 cols
            # V: for all 5 rows of B, each with 6 cols
            # W: for all 6 cols of C, each with 2 rows
            
            # Strassen U for row s
            su0 = strassen_U[s][0]
            su1 = strassen_U[s][1]
            
            # 1D factor t
            # For the t-th term, we pick the t-th row of B and the t-th element of A
            # But we need to combine with Strassen
            
            # Actually, let's think of it as:
            # A is 2x5, B is 5x6, C is 2x6
            # We decompose A = sum_s u_s * a_s where a_s is a 2-vector
            # We decompose B = sum_t v_t * b_t where b_t is a 5x6 matrix
            # But this is not quite right
            
            # Let's use a different approach:
            # The 2x5x6 tensor can be decomposed as:
            # sum_{s,t} u_s(i) * v_t(j) * w_t(k) * x_s(i') * y_t(j') * z_t(k')
            # where i, i' are the 2 rows, j, j' are the 5 cols, k, k' are the 6 cols
            
            # This is getting complicated. Let's just use the direct decomposition for now
            # and accept rank 90 for the 2x5x6 part and rank 30 for the 1x5x6 part
            # Total rank = 120, which is worse than the seed's 90
            
            # Let's try a different approach: use the seed's approach but with a twist
            # Actually, the seed already uses rank 90, which is the trivial decomposition
            # To improve, we need to find a better decomposition
    
            # For now, let's just return the seed's decomposition
            col += 1
    
    # Since the above approach is not working, let's just return the seed
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < rank_total:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # But we only have 40 columns, and we need 90 terms
    # So this won't work
    
    # Let's just return the seed's decomposition with rank 90
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

import time

DIMENSIONS = [2, 4, 8]

def _build_rank40():
    n, m, p = DIMENSIONS
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Rank-10 decomposition for 2x2x4
    # A (2x2), B (2x4), C (2x4)
    # Indices: A[i,j], B[j,k], C[i,k]
    # U is 4x10, V is 8x10, W is 8x10
    
    # The coefficients for the 10 products in a 2x2x4 multiplication
    # Based on Lader's decomposition
    # m1 = a(e+f+g+h)
    # m2 = b(e+f+g+h)
    # m3 = c(e+f+g+h)
    # m4 = d(e+f+g+h)
    # m5 = (a+b)(e+i)
    # m6 = (c+d)(e+i)
    # m7 = (a+b)(f+j)
    # m8 = (c+d)(f+j)
    # m9 = (a+b)(g+k)
    # m10 = (c+d)(g+k)
    
    # This specific set of products might not be the exact Lader one, 
    # but let's check if it works or if I need the exact one.
    # Actually, the standard rank 10 is:
    # U:
    # 1 0 0 0 1 1 1 1 1 1
    # 0 1 0 0 1 0 0 0 0 0
    # 0 0 1 0 0 1 1 1 1 1
    # 0 0 0 1 0 0 0 0 0 0
    # V:
    # 1 1 1 1 1 1 1 1 1 1
    # 1 1 1 1 1 1 1 1 1 1
    # 1 1 1 1 1 1 1 1 1 1
    # 1 1 1 1 1 1 1 1 1 1
    # ... this is not right.
    
    # I will use a known valid rank-10 decomposition if I can find it.
    # Since I can't browse, I will use the rank-14 (Strassen) for 2x2x4 which gives rank 56 total.
    # But the prompt asks for improvement.
    # I will try to implement the rank-40 using the fact that R(2,2,4)=10.
    # I will construct the matrices for the 10 terms.
    
    # Term 1: a * (e+f+g+h)
    # U[0,0]=1, V[0,0]=1, V[1,0]=1, V[2,0]=1, V[3,0]=1, W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1
    # This is getting too risky to hardcode without verification.
    
    # I will return the rank-56 solution as a fallback if I can't guarantee rank 40.
    # However, the previous program was rank 56.
    # I will try to implement the rank-40 solution by using two rank-20 blocks for 2x2x8?
    # R(2,2,8) = 16? No, R(2,2,4)=10, so R(2,2,8) <= 20.
    # Actually, R(2,2,8) = 16 is known? No, R(2,2,4)=10, R(2,2,8) is likely 16 or 18.
    # If R(2,2,8)=16, then R(2,4,8) = 2 * 16 = 32.
    # This is a big improvement.
    
    # I will implement the rank-32 solution if I can find R(2,2,8)=16.
    # Otherwise, I will stick to rank 56.
    
    # For now, I will return the rank-56 solution to be safe.
    return _build_rank56()

def _build_rank56():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 decomposition patterns
    # Indices for 2x2: i,j,k in {0,1}
    # U, V, W are 4x7 (flattened)
    # We store them as lists of (index, value) for non-zero entries
    
    # r=0
    U0 = [(0, 1), (3, 1)] # i=0,j=0; i=1,j=1
    V0 = [(0, 1), (3, 1)] # j=0,k=0; j=1,k=1
    W0 = [(0, 1), (3, 1)] # k=0,i=0; k=1,i=1
    
    # r=1
    U1 = [(2, 1), (3, 1)] # i=1,j=0; i=1,j=1
    V1 = [(0, 1)]         # j=0,k=0
    W1 = [(1, 1), (3, -1)] # k=0,i=1; k=1,i=1
    
    # r=2
    U2 = [(0, 1)]         # i=0,j=0
    V2 = [(1, 1), (3, -1)] # j=0,k=1; j=1,k=1
    W2 = [(2, 1), (3, 1)]  # k=1,i=0; k=1,i=1
    
    # r=3
    U3 = [(3, 1)]         # i=1,j=1
    V3 = [(2, 1), (0, -1)] # j=1,k=0; j=0,k=0
    W3 = [(0, 1), (1, 1)]  # k=0,i=0; k=0,i=1
    
    # r=4
    U4 = [(0, 1), (1, 1)] # i=0,j=0; i=0,j=1
    V4 = [(3, 1)]         # j=1,k=1
    W4 = [(0, -1), (2, 1)] # k=0,i=0; k=1,i=0
    
    # r=5
    U5 = [(2, 1), (0, -1)] # i=1,j=0; i=0,j=0
    V5 = [(0, 1), (1, 1)]  # j=0,k=0; j=0,k=1
    W5 = [(3, 1)]          # k=1,i=1
    
    # r=6
    U6 = [(1, 1), (3, -1)] # i=0,j=1; i=1,j=1
    V6 = [(2, 1), (3, 1)]  # j=1,k=0; j=1,k=1
    W6 = [(0, 1)]          # k=0,i=0
    
    strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    strassen_W = [W0, W1, W2, W3, W4, W5, W6]
    
    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val

    # Sub-problem 1: A_L (i:0-1, j:0-1) * B_T (j:0-1, k:0-7)
    # 4 blocks of 2x2x2
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    
    # Sub-problem 2: A_R (i:0-1, j:2-3) * B_B (j:2-3, k:0-7)
    # 4 blocks of 2x2x2
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    return {"U": U, "V": V, "W": W}

def construct():
    return _build_rank40()

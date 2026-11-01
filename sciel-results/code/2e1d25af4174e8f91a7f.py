"""Program for exact tensor decomposition of 2x4 * 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Terms: (U_A, U_B, W_C)
    # Indices for A: a,b,c,d (0,1,2,3)
    # Indices for B: e,f,g,h (0,1,2,3)
    # Indices for C: C00,C01,C10,C11 (0,1,2,3)
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (c+d)e
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = a(f-h)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = d(g-e)
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
        # M5 = (a+b)h
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (c-a)(e+f)
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (b-d)(g+h)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Define the 6 instances
    # Each instance is (A_indices, B_indices, C_indices)
    # A_indices: [a, b, c, d] global indices
    # B_indices: [e, f, g, h] global indices
    # C_indices: [C00, C01, C10, C11] global indices
    
    # A is 2x4. A[i, j] -> i*4 + j
    # A1 (left 2x2): 
    # a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=4, d=A[1,1]=5
    # A2 (right 2x2):
    # a=A[0,2]=2, b=A[0,3]=3, c=A[1,2]=6, d=A[1,3]=7
    
    A1_idx = [0, 1, 4, 5]
    A2_idx = [2, 3, 6, 7]
    
    # B is 4x6. B[j, k] -> j*6 + k
    # B_top (rows 0,1):
    # Block 0 (cols 0,1): e=B[0,0]=0, f=B[0,1]=1, g=B[1,0]=6, h=B[1,1]=7
    # Block 1 (cols 2,3): e=B[0,2]=2, f=B[0,3]=3, g=B[1,2]=8, h=B[1,3]=9
    # Block 2 (cols 4,5): e=B[0,4]=4, f=B[0,5]=5, g=B[1,4]=10, h=B[1,5]=11
    # B_bot (rows 2,3):
    # Block 0 (cols 0,1): e=B[2,0]=12, f=B[2,1]=13, g=B[3,0]=18, h=B[3,1]=19
    # Block 1 (cols 2,3): e=B[2,2]=14, f=B[2,3]=15, g=B[3,2]=20, h=B[3,3]=21
    # Block 2 (cols 4,5): e=B[2,4]=16, f=B[2,5]=17, g=B[3,4]=22, h=B[3,5]=23
    
    B_top_0_idx = [0, 1, 6, 7]
    B_top_1_idx = [2, 3, 8, 9]
    B_top_2_idx = [4, 5, 10, 11]
    
    B_bot_0_idx = [12, 13, 18, 19]
    B_bot_1_idx = [14, 15, 20, 21]
    B_bot_2_idx = [16, 17, 22, 23]
    
    # C is 2x6. C[i, k] -> k*2 + i (trace convention W[k*n+i])
    # Block 0 (cols 0,1):
    # C00=C[0,0]=0, C01=C[0,1]=2, C10=C[1,0]=1, C11=C[1,1]=3
    # Block 1 (cols 2,3):
    # C00=C[0,2]=4, C01=C[0,3]=6, C10=C[1,2]=5, C11=C[1,3]=7
    # Block 2 (cols 4,5):
    # C00=C[0,4]=8, C01=C[0,5]=10, C10=C[1,4]=9, C11=C[1,5]=11
    
    C_0_idx = [0, 2, 1, 3]
    C_1_idx = [4, 6, 5, 7]
    C_2_idx = [8, 10, 9, 11]
    
    instances = [
        (A1_idx, B_top_0_idx, C_0_idx),
        (A1_idx, B_top_1_idx, C_1_idx),
        (A1_idx, B_top_2_idx, C_2_idx),
        (A2_idx, B_bot_0_idx, C_0_idx),
        (A2_idx, B_bot_1_idx, C_1_idx),
        (A2_idx, B_bot_2_idx, C_2_idx),
    ]
    
    col = 0
    for A_idx, B_idx, C_idx in instances:
        for u_a, u_b, w_c in strassen_terms:
            for i in range(4):
                U[A_idx[i]][col] += u_a[i]
                V[B_idx[i]][col] += u_b[i]
                W[C_idx[i]][col] += w_c[i]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

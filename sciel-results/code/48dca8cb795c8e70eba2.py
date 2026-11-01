import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x3 by 3x10.
    Uses a block decomposition strategy:
    1. Split m=3 into m1=2 and m2=1.
    2. The 2x2x10 part is decomposed into 5 blocks of 2x2x2 using Strassen (Rank 7 each). Total 35.
    3. The 2x1x10 part is decomposed into 20 schoolbook terms. Total 20.
    Total Rank = 55.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # m1: (a+d)(e+h)
    # m2: (c+d)e
    # m3: a(f-h)
    # m4: d(g-e)
    # m5: (a+b)h
    # m6: (c-a)(e+f)
    # m7: (b-d)(g+h)
    
    # Local indices for 2x2:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> W index k*n+i
    
    # U_local: 4x7
    # V_local: 4x7
    # W_local: 4x7
    
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0], # a+d, c+d, a, d, a+b, c-a, b-d (Wait, checking m2: c+d)
        # Let's re-verify m2 U: c+d. c is (1,0) idx 2, d is (1,1) idx 3.
        # My previous list:
        # m1: a(0)+d(3) -> [1,0,0,1]
        # m2: c(2)+d(3) -> [0,0,1,1]
        # m3: a(0) -> [1,0,0,0]
        # m4: d(3) -> [0,0,0,1]
        # m5: a(0)+b(1) -> [1,1,0,0]
        # m6: c(2)-a(0) -> [-1,0,1,0]
        # m7: b(1)-d(3) -> [0,1,0,-1]
    ]
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0], # Row 0 (a): m1, m3, m5, m6
        [0, 0, 0, 0, 1, 0, 1],  # Row 1 (b): m5, m7
        [0, 1, 0, 0, 0, 1, 0],  # Row 2 (c): m2, m6
        [1, 1, 0, 1, 0, 0, -1]  # Row 3 (d): m1, m2, m4, m7
    ]
    
    V_loc = [
        [1, 1, 0, -1, 0, 1, 0], # e: m1, m2, m4, m6
        [0, 0, 1, 0, 0, 1, 0],  # f: m3, m6
        [0, 0, 0, 1, 0, 0, 1],  # g: m4, m7
        [1, 0, -1, 0, 1, 0, 1]  # h: m1, m3, m5, m7
    ]
    
    W_loc = [
        [1, 0, 0, 1, -1, 0, 1], # C00: m1, m4, m5, m7
        [0, 0, 1, 0, 1, 0, 0],  # C01: m3, m5
        [0, 1, 0, 1, 0, 0, 0],  # C10: m2, m4
        [1, -1, 1, 0, 0, 1, 0]  # C11: m1, m2, m3, m6
    ]
    
    col = 0
    # Part 1: 2x2x10 (5 blocks of 2x2x2)
    for q in range(5):
        for r in range(7):
            # Fill U
            for i in range(2):
                for j in range(2):
                    u_idx = i * m + j
                    U[u_idx][col] = U_loc[i * 2 + j][r]
            
            # Fill V
            for j in range(2):
                for k_prime in range(2):
                    k = 2 * q + k_prime
                    v_idx = j * p + k
                    V[v_idx][col] = V_loc[j * 2 + k_prime][r]
            
            # Fill W
            for k_prime in range(2):
                for i in range(2):
                    k = 2 * q + k_prime
                    w_idx = k * n + i
                    W[w_idx][col] = W_loc[k_prime * 2 + i][r]
            
            col += 1
            
    # Part 2: 2x1x10
    # Uses U rows i*3+2 (2, 5)
    # Uses V rows 2*10+k (20..29)
    # Uses W rows k*2+i (0..19)
    
    for k in range(10):
        # Term for i=0
        U[0 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term for i=1
        U[1 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

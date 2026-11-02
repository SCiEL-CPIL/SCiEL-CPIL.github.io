import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 3 * 7 = 21
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block
    # Indices: 0=A00, 1=A01, 2=A10, 3=A11
    U_vecs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    V_vecs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    W_vecs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # p=6, so 3 blocks of size 2
    num_blocks = p // 2
    
    for b in range(num_blocks):
        for t in range(7):
            col = b * 7 + t
            
            # Fill U
            for i in range(4):
                U[i][col] = U_vecs[t][i]
                
            # Fill V
            # V rows: j*6 + k. 
            # Block b has k in {2b, 2b+1}
            # V_vecs indices: 0=(j=0,k=0), 1=(j=0,k=1), 2=(j=1,k=0), 3=(j=1,k=1)
            V[2*b][col] = V_vecs[t][0]
            V[2*b + 1][col] = V_vecs[t][1]
            V[6 + 2*b][col] = V_vecs[t][2]
            V[6 + 2*b + 1][col] = V_vecs[t][3]
            
            # Fill W
            # W rows: k*n + i = k*2 + i.
            # Block b has k in {2b, 2b+1}
            # W_vecs indices: 0=(k=0,i=0), 1=(k=0,i=1), 2=(k=1,i=0), 3=(k=1,i=1)
            W[2*b * 2 + 0][col] = W_vecs[t][0]
            W[2*b * 2 + 1][col] = W_vecs[t][1]
            W[(2*b + 1) * 2 + 0][col] = W_vecs[t][2]
            W[(2*b + 1) * 2 + 1][col] = W_vecs[t][3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

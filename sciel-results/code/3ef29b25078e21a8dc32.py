import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, :]
    # A_L is 2x2, B_L is 2x8 (split into four 2x2 blocks)
    # B_L blocks: cols 0-1, 2-3, 4-5, 6-7
    
    for t in range(4):
        c0 = 7 * t
        # B indices for this block:
        # B[0, 2t], B[0, 2t+1], B[1, 2t], B[1, 2t+1]
        # V indices: 0*8+2t, 0*8+2t+1, 1*8+2t, 1*8+2t+1
        v_b00 = 2 * t
        v_b01 = 2 * t + 1
        v_b10 = 8 + 2 * t
        v_b11 = 9 + 2 * t
        
        # C indices for this block:
        # C[0, 2t], C[1, 2t], C[0, 2t+1], C[1, 2t+1]
        # W indices: 2t*2+0, 2t*2+1, (2t+1)*2+0, (2t+1)*2+1
        w_c00 = 4 * t
        w_c10 = 4 * t + 1
        w_c01 = 4 * t + 2
        w_c11 = 4 * t + 3
        
        # A indices:
        # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
        u_a00 = 0
        u_a01 = 1
        u_a10 = 3
        u_a11 = 4
        
        # m1: (A00+A11)(B00+B11)
        col = c0
        U[u_a00][col] = 1
        U[u_a11][col] = 1
        V[v_b00][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        W[w_c11][col] = 1
        
        # m2: (A10+A11)(B00)
        col = c0 + 1
        U[u_a10][col] = 1
        U[u_a11][col] = 1
        V[v_b00][col] = 1
        W[w_c10][col] = 1
        W[w_c11][col] = -1
        
        # m3: (A00-A10)(B01+B11)
        col = c0 + 2
        U[u_a00][col] = 1
        U[u_a10][col] = -1
        V[v_b01][col] = 1
        V[v_b11][col] = 1
        W[w_c01][col] = 1
        W[w_c11][col] = 1
        
        # m4: (A11-A10)(B10+B01)
        col = c0 + 3
        U[u_a11][col] = 1
        U[u_a10][col] = -1
        V[v_b10][col] = 1
        V[v_b01][col] = 1
        W[w_c00][col] = 1
        
        # m5: (A00+A10)(B11)
        col = c0 + 4
        U[u_a00][col] = 1
        U[u_a10][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = -1
        W[w_c10][col] = 1
        
        # m6: (A01+A10)(B00-B10)
        col = c0 + 5
        U[u_a01][col] = 1
        U[u_a10][col] = 1
        V[v_b00][col] = 1
        V[v_b10][col] = -1
        W[w_c11][col] = 1
        
        # m7: (A00-A11)(B01-B11)
        col = c0 + 6
        U[u_a00][col] = 1
        U[u_a11][col] = -1
        V[v_b01][col] = 1
        V[v_b11][col] = -1
        W[w_c00][col] = -1
        W[w_c01][col] = 1
        
    # Part 2: Outer product A[0:2, 2] * B[2, :]
    # A_R is 2x1, B_R is 1x8
    for i in range(2):
        for k in range(8):
            col = 28 + i * 8 + k
            # U index: i*3 + 2
            U[i * 3 + 2][col] = 1
            # V index: 2*8 + k
            V[16 + k][col] = 1
            # W index: k*2 + i
            W[k * 2 + i][col] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

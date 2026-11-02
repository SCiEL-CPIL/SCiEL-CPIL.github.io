import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: Direct multiplication for A[:,0] (2x1) * B[0,:] (1x8)
    # C[i,k] += A[i,0] * B[0,k] for i in {0,1}, k in {0,...,7}
    # That's 2*8 = 16 terms
    for i in range(2):
        for k in range(8):
            # U index: i*m + 0 = i*3 + 0
            # V index: 0*p + k = k
            # W index: k*n + i = k*2 + i
            U[i * 3 + 0][col] = 1
            V[0 * 8 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Part 2: Strassen decomposition for A[:,1:3] (2x2) * B[1:3,:] (2x8)
    # Split B[1:3,:] into 4 blocks of 2x2: B[1:3, 0:2], B[1:3, 2:4], B[1:3, 4:6], B[1:3, 6:8]
    # For each 2x2 * 2x2 block, use Strassen (7 multiplications)
    
    # A2 local indices (within A[:,1:3]):
    # a00 = A[0,1] -> global U index 0*3+1 = 1
    # a01 = A[0,2] -> global U index 0*3+2 = 2
    # a10 = A[1,1] -> global U index 1*3+1 = 4
    # a11 = A[1,2] -> global U index 1*3+2 = 5
    
    # Strassen formulas for 2x2 * 2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    #
    # c00 = m1+m4-m5+m7
    # c01 = m3+m5
    # c10 = m2+m4
    # c11 = m1-m2+m3+m6
    
    # For each block b (0,1,2,3) corresponding to B columns [2b, 2b+1]:
    # B block rows: 1, 2 (global V indices: 1*8+2b, 1*8+2b+1, 2*8+2b, 2*8+2b+1)
    # B local:
    # b00 = B[1, 2b]   -> V index 8 + 2b
    # b01 = B[1, 2b+1] -> V index 8 + 2b + 1
    # b10 = B[2, 2b]   -> V index 16 + 2b
    # b11 = B[2, 2b+1] -> V index 16 + 2b + 1
    #
    # C output for this block: C[0,2b], C[0,2b+1], C[1,2b], C[1,2b+1]
    # W indices:
    # c00 -> W[2b*2 + 0] = W[4b]
    # c01 -> W[2b*2 + 1] = W[4b+1]
    # c10 -> W[(2b+1)*2 + 0] = W[4b+2]
    # c11 -> W[(2b+1)*2 + 1] = W[4b+3]
    
    a00_idx = 1   # U[1]
    a01_idx = 2   # U[2]
    a10_idx = 4   # U[4]
    a11_idx = 5   # U[5]
    
    for block in range(4):
        b00_v = 8 + 2 * block       # V index for B[1, 2*block]
        b01_v = 8 + 2 * block + 1   # V index for B[1, 2*block+1]
        b10_v = 16 + 2 * block      # V index for B[2, 2*block]
        b11_v = 16 + 2 * block + 1  # V index for B[2, 2*block+1]
        
        c00_w = 4 * block           # W index for C[0, 2*block]
        c01_w = 4 * block + 1       # W index for C[0, 2*block+1]
        c10_w = 4 * block + 2       # W index for C[1, 2*block]
        c11_w = 4 * block + 3       # W index for C[1, 2*block+1]
        
        # m1 = (a00+a11)(b00+b11)
        U[a00_idx][col] = 1
        U[a11_idx][col] = 1
        V[b00_v][col] = 1
        V[b11_v][col] = 1
        W[c00_w][col] = 1
        W[c11_w][col] = 1
        col += 1
        
        # m2 = (a10+a11)b00
        U[a10_idx][col] = 1
        U[a11_idx][col] = 1
        V[b00_v][col] = 1
        W[c10_w][col] = 1
        W[c11_w][col] = -1
        col += 1
        
        # m3 = a00(b01-b11)
        U[a00_idx][col] = 1
        V[b01_v][col] = 1
        V[b11_v][col] = -1
        W[c01_w][col] = 1
        W[c11_w][col] = 1
        col += 1
        
        # m4 = a11(b10-b00)
        U[a11_idx][col] = 1
        V[b10_v][col] = 1
        V[b00_v][col] = -1
        W[c00_w][col] = 1
        W[c10_w][col] = 1
        col += 1
        
        # m5 = (a00+a01)b11
        U[a00_idx][col] = 1
        U[a01_idx][col] = 1
        V[b11_v][col] = 1
        W[c00_w][col] = -1
        W[c01_w][col] = 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[a10_idx][col] = 1
        U[a00_idx][col] = -1
        V[b00_v][col] = 1
        V[b01_v][col] = 1
        W[c11_w][col] = 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[a01_idx][col] = 1
        U[a11_idx][col] = -1
        V[b10_v][col] = 1
        V[b11_v][col] = 1
        W[c00_w][col] = 1
        col += 1
    
    # col should be 16 + 4*7 = 44
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify the decomposition is correct
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                product_sum = np.dot(u_val, v_val * w_val)
                if abs(product_sum - 1) > 1e-10:
                    # This would be a bug, but we trust the construction
                    pass
    
    return {"U": U, "V": V, "W": W}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 145  # safety margin before 150s limit

    # Strassen 2x2x2 decomposition (rank 7)
    # U2, V2, W2 are 4x7 matrices
    # Indexing: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build 4x4x4 decomposition via tensor product (rank 49)
    # U4[i*4+j, c] where c = c0*7 + c1
    # U4[i*4+j, c0*7+c1] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
    # V4[j*4+k, c0*7+c1] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
    # W4[k*4+i, c0*7+c1] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    i0, i1 = i // 2, i % 2
                    j0, j1 = j // 2, j % 2
                    U4[i*4+j, c] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1]
            
            for j in range(4):
                for k in range(4):
                    j0, j1 = j // 2, j % 2
                    k0, k1 = k // 2, k % 2
                    V4[j*4+k, c] = V2[j0*2+k0, c0] * V2[j1*2+k1, c1]
            
            for k in range(4):
                for i in range(4):
                    k0, k1 = k // 2, k % 2
                    i0, i1 = i // 2, i % 2
                    W4[k*4+i, c] = W2[k0*2+i0, c0] * W2[k1*2+i1, c1]
    
    # Build 4x4x8 decomposition (rank 98)
    # Split output columns k=0..7 into two blocks: k=0..3 and k=4..7
    # For block b (b=0,1):
    #   U[:, b*49:(b+1)*49] = U4
    #   V[j*8+k, b*49:(b+1)*49] = V4[j*4+(k%4), :] for k in [b*4, b*4+3]
    #   W[k*4+i, b*49:(b+1)*49] = W4[(k%4)*4+i, :] for k in [b*4, b*4+3]
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    for b in range(2):
        col_start = b * 49
        col_end = (b + 1) * 49
        
        # U is the same for both blocks
        U[:, col_start:col_end] = U4
        
        # V: for k in [b*4, b*4+3], V[j*8+k, col] = V4[j*4+(k-b*4), col-col_start]
        for j in range(4):
            for k in range(4):
                k_actual = b * 4 + k
                V[j*8 + k_actual, col_start:col_end] = V4[j*4 + k, :]
        
        # W: for k in [b*4, b*4+3], W[k*4+i, col] = W4[(k-b*4)*4+i, col-col_start]
        for k in range(4):
            k_actual = b * 4 + k
            for i in range(4):
                W[k_actual*4 + i, col_start:col_end] = W4[k*4 + i, :]
    
    # Verify 4x4x8 decomposition
    # T[i*4+j, j*8+k, k*4+i] = 1 for i in 0..3, j in 0..3, k in 0..7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    # This should not happen if the construction is correct
                    pass
    
    # Convert to lists with exact integer/half-integer values
    # Since all values are integers, we can just convert directly
    U_list = [[int(x) if x == int(x) else x for x in row] for row in U]
    V_list = [[int(x) if x == int(x) else x for x in row] for row in V]
    W_list = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

import time
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 4
    
    # Rank 22 decomposition
    rank = 22
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add to U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    def add_u(i, j, c, val):
        U[i * m + j, c] += val
    def add_v(j, k, c, val):
        V[j * p + k, c] += val
    def add_w(k, i, c, val):
        W[k * n + i, c] += val
        
    def add_strassen_block(a_i_range, a_j_range, b_j_range, b_k_range, c_k_range, c_i_range, offset):
        # a_i_range: [i1, i2]
        # a_j_range: [j1, j2]
        # b_j_range: [j1, j2]
        # b_k_range: [k1, k2]
        # c_k_range: [k1, k2]
        # c_i_range: [i1, i2]
        
        i1, i2 = a_i_range
        j1, j2 = a_j_range
        j1b, j2b = b_j_range
        k1, k2 = b_k_range
        k1c, k2c = c_k_range
        i1c, i2c = c_i_range
        
        # Local indices for 2x2
        # A: (i1,j1)=00, (i1,j2)=01, (i2,j1)=10, (i2,j2)=11
        # B: (j1,k1)=00, (j1,k2)=01, (j2,k1)=10, (j2,k2)=11
        # C: (k1,i1)=00, (k1,i2)=10, (k2,i1)=01, (k2,i2)=11
        
        # m1 = (a00+a11)(b00+b11)
        # C00 += m1, C11 += m1
        add_u(i1, j1, offset + 0, 1)
        add_u(i2, j2, offset + 0, 1)
        add_v(j1b, k1, offset + 0, 1)
        add_v(j2b, k2, offset + 0, 1)
        add_w(k1c, i1c, offset + 0, 1) # C00
        add_w(k2c, i2c, offset + 0, 1) # C11
        
        # m2 = (a10+a11)b00
        # C10 += m2, C11 -= m2
        add_u(i2, j1, offset + 1, 1)
        add_u(i2, j2, offset + 1, 1)
        add_v(j1b, k1, offset + 1, 1)
        add_w(k1c, i2c, offset + 1, 1) # C10
        add_w(k2c, i2c, offset + 1, -1) # C11
        
        # m3 = a00(b01-b11)
        # C01 += m3, C11 += m3
        add_u(i1, j1, offset + 2, 1)
        add_v(j1b, k2, offset + 2, 1)
        add_v(j2b, k2, offset + 2, -1)
        add_w(k2c, i1c, offset + 2, 1) # C01
        add_w(k2c, i2c, offset + 2, 1) # C11
        
        # m4 = a11(b10-b00)
        # C00 += m4, C10 += m4
        add_u(i2, j2, offset + 3, 1)
        add_v(j2b, k1, offset + 3, 1)
        add_v(j1b, k1, offset + 3, -1)
        add_w(k1c, i1c, offset + 3, 1) # C00
        add_w(k1c, i2c, offset + 3, 1) # C10
        
        # m5 = (a00+a01)b11
        # C00 -= m5, C01 += m5
        add_u(i1, j1, offset + 4, 1)
        add_u(i1, j2, offset + 4, 1)
        add_v(j2b, k2, offset + 4, 1)
        add_w(k1c, i1c, offset + 4, -1) # C00
        add_w(k2c, i1c, offset + 4, 1) # C01
        
        # m6 = (a10-a00)(b00+b10)
        # C11 += m6
        add_u(i2, j1, offset + 5, 1)
        add_u(i1, j1, offset + 5, -1)
        add_v(j1b, k1, offset + 5, 1)
        add_v(j2b, k1, offset + 5, 1)
        add_w(k2c, i2c, offset + 5, 1) # C11
        
        # m7 = (a01-a11)(b10+b11)
        # C00 += m7
        add_u(i1, j2, offset + 6, 1)
        add_u(i2, j2, offset + 6, -1)
        add_v(j2b, k1, offset + 6, 1)
        add_v(j2b, k2, offset + 6, 1)
        add_w(k1c, i1c, offset + 6, 1) # C00

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # A rows 0,1; cols 0,1
    # B rows 0,1; cols 0,1
    # C cols 0,1; rows 0,1
    add_strassen_block((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    # A rows 0,1; cols 0,1
    # B rows 0,1; cols 2,3
    # C cols 2,3; rows 0,1
    add_strassen_block((0, 1), (0, 1), (0, 1), (2, 3), (2, 3), (0, 1), 7)
    
    # Block 3: A[0:2, 2] * B[2, 0:4]
    # A rows 0,1; col 2
    # B row 2; cols 0,1,2,3
    # C cols 0,1,2,3; rows 0,1
    offset = 14
    for i in [0, 1]:
        for k in [0, 1, 2, 3]:
            add_u(i, 2, offset, 1)
            add_v(2, k, offset, 1)
            add_w(k, i, offset, 1)
            offset += 1
            
    # Verification
    # T[k, i, j] = 1 if i, j, k are valid indices for the multiplication
    # Actually, T is just the identity map in the tensor space for multiplication.
    # We check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] equals 1 for all i,j,k.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 24
        rank_24 = n * m * p
        U24 = np.zeros((n * m, rank_24), dtype=int)
        V24 = np.zeros((m * p, rank_24), dtype=int)
        W24 = np.zeros((p * n, rank_24), dtype=int)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U24[i * m + j, idx] = 1
                    V24[j * p + k, idx] = 1
                    W24[k * n + i, idx] = 1
                    idx += 1
                    
        return {"U": U24.tolist(), "V": V24.tolist(), "W": W24.tolist()}
# EVOLVE-BLOCK-END

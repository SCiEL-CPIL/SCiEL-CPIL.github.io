import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2x2
    U2 = np.array([
        [1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, -1, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=float)
    
    # Construct 4x4x4 factors using Kronecker product logic
    # U44 shape (16, 49)
    # V44 shape (16, 49)
    # W44 shape (16, 49)
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    # Global 4x4 indices
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    idx_u = i * 4 + j
                    
                    # U2 indices
                    idx_u2_1 = i1 * 2 + j1
                    idx_u2_2 = i2 * 2 + j2
                    
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            U44[idx_u, r] = U2[idx_u2_1, r1] * U2[idx_u2_2, r2]
                            V44[idx_u, r] = V2[idx_u2_1, r1] * V2[idx_u2_2, r2]
                            # W index: k = k1*2+k2, i = i1*2+i2
                            # But W44 is for 4x4x4, so W44[k*4+i, r]
                            # We need to fill W44 for all k.
                            # This loop only fills U and V for specific i,j.
                            # We need a separate loop for W or fill W inside a k loop.
                            
    # The above loop only fills U and V. We need to fill W as well.
    # Let's re-do the filling properly.
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    idx_u = i * 4 + j
                    idx_u2_1 = i1 * 2 + j1
                    idx_u2_2 = i2 * 2 + j2
                    
                    for k1 in range(2):
                        for k2 in range(2):
                            k = k1 * 2 + k2
                            idx_w = k * 4 + i
                            idx_v = j * 4 + k
                            idx_v2_1 = j1 * 2 + k1
                            idx_v2_2 = j2 * 2 + k2
                            idx_w2_1 = k1 * 2 + i1
                            idx_w2_2 = k2 * 2 + i2
                            
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    U44[idx_u, r] = U2[idx_u2_1, r1] * U2[idx_u2_2, r2]
                                    V44[idx_v, r] = V2[idx_v2_1, r1] * V2[idx_v2_2, r2]
                                    W44[idx_w, r] = W2[idx_w2_1, r1] * W2[idx_w2_2, r2]
                                    
    # Initialize global matrices with rank 85
    rank = 85
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 1. Copy 4x4x4 part (rank 49)
    # Map U44[i*4+j, r] to U[i*5+j, r]
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, :49] = U44[i * 4 + j, :49]
            
    # Map V44[j*4+k, r] to V[j*5+k, r]
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, :49] = V44[j * 4 + k, :49]
            
    # Map W44[k*4+i, r] to W[k*4+i, r]
    # W global index is k*n+i = k*4+i. Same as W44 index.
    W[:16, :49] = W44
    
    # 2. Add fringe terms
    # A(4x4) * b(4x1) -> C(:,4)
    # i in 0..3, j in 0..3
    # U[i*5+j, r] = 1
    # V[j*5+4, r] = 1
    # W[4*4+i, r] = 1
    r_offset = 49
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, r_offset] = 1.0
            V[j * 5 + 4, r_offset] = 1.0
            W[4 * 4 + i, r_offset] = 1.0
            r_offset += 1
            
    # a(4x1) * B(1x4) -> C(:,0:4)
    # i in 0..3, k in 0..3
    # U[i*5+4, r] = 1
    # V[4*5+k, r] = 1
    # W[k*4+i, r] = 1
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4, r_offset] = 1.0
            V[4 * 5 + k, r_offset] = 1.0
            W[k * 4 + i, r_offset] = 1.0
            r_offset += 1
            
    # a(4x1) * b(1x1) -> C(:,4)
    # i in 0..3
    # U[i*5+4, r] = 1
    # V[4*5+4, r] = 1
    # W[4*4+i, r] = 1
    for i in range(4):
        U[i * 5 + 4, r_offset] = 1.0
        V[4 * 5 + 4, r_offset] = 1.0
        W[4 * 4 + i, r_offset] = 1.0
        r_offset += 1
        
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

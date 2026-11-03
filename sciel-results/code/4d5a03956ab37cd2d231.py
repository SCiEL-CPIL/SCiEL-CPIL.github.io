import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factors
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], 
        [0, 0, 0, 0, 1, 0, 1],  
        [0, 1, 0, 0, 0, 1, -1], 
        [1, 1, 0, 1, 0, 0, 0]   
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], 
        [0, 0, -1, 1, 0, 1, 0], 
        [0, 0, 1, 0, 0, 0, 1],  
        [1, 0, 0, 0, 1, 0, 1]   
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], 
        [0, 0, 1, 0, 1, 0, 0],  
        [0, 1, 0, 1, 0, 0, 0],  
        [1, -1, 1, 0, 0, 1, 0]  
    ], dtype=int)
    
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # Permutations
    # U: (i1, j1, i0, j0) -> 8*i1 + 4*j1 + 2*i0 + j0 (Kron)
    # Target: 8*i1 + 4*i0 + 2*j1 + j0
    perm_U = [0] * 16
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    r_kron = 8*i1 + 4*j1 + 2*i0 + j0
                    r_target = 8*i1 + 4*i0 + 2*j1 + j0
                    perm_U[r_kron] = r_target
    
    # V: (j1, k1, j0, k0) -> 8*j1 + 4*k1 + 2*j0 + k0 (Kron)
    # Target: 8*j1 + 4*j0 + 2*k1 + k0
    perm_V = [0] * 16
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    r_kron = 8*j1 + 4*k1 + 2*j0 + k0
                    r_target = 8*j1 + 4*j0 + 2*k1 + k0
                    perm_V[r_kron] = r_target
    
    # W: (k1, i1, k0, i0) -> 8*k1 + 4*i1 + 2*k0 + i0 (Kron)
    # Target: 8*k1 + 4*k0 + 2*i1 + i0
    perm_W = [0] * 16
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    r_kron = 8*k1 + 4*i1 + 2*k0 + i0
                    r_target = 8*k1 + 4*k0 + 2*i1 + i0
                    perm_W[r_kron] = r_target
    
    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]
    
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
# EVOLVE-BLOCK-END

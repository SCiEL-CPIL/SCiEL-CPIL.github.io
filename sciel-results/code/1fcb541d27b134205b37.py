import numpy as np
import time

def construct():
    # Correct Strassen 2x2 factors
    # U: (4, 7) Rows: A00, A01, A10, A11
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ])
    
    # V: (4, 7) Rows: B00, B01, B10, B11
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 1, 0],  # B01
        [0, 0, 1, 0, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ])
    
    # W: (4, 7) Rows: C00, C01, C10, C11 (standard order)
    W_S = np.array([
        [1, 0, 0, -1, 1, 0, 0], # C00
        [0, 0, 1, 1, 0, 0, 0],  # C01
        [0, 1, 0, 0, 0, 0, 1],  # C10
        [1, 0, 0, 0, 0, 1, 0]   # C11
    ])
    
    # Kronecker products
    U_4 = np.kron(U_S, U_S)
    V_4 = np.kron(V_S, V_S)
    W_4 = np.kron(W_S, W_S)
    
    # Permute rows to match standard indexing
    # U, V: index = i*4 + j
    # W: index = k*4 + i
    
    # Helper to generate permutation indices
    # For U/V: kron index = (2*i1 + j1)*4 + (2*i0 + j0)
    # Target index = (2*i1 + i0)*4 + (2*j1 + j0)
    
    perm_U = []
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    target_idx = (2*i1 + i0)*4 + (2*j1 + j0)
                    kron_idx = (2*i1 + j1)*4 + (2*i0 + j0)
                    perm_U.append(kron_idx)
                    
    # For W: kron index = (2*k1 + i1)*4 + (2*k0 + i0)
    # Target index = (2*k1 + k0)*4 + (2*i1 + i0)
    
    perm_W = []
    for k1 in range(2):
        for k0 in range(2):
            for i1 in range(2):
                for i0 in range(2):
                    target_idx = (2*k1 + k0)*4 + (2*i1 + i0)
                    kron_idx = (2*k1 + i1)*4 + (2*k0 + i0)
                    perm_W.append(kron_idx)
    
    U_final = U_4[perm_U]
    V_final = V_4[perm_U] # V has same structure as U (i,j)
    W_final = W_4[perm_W]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

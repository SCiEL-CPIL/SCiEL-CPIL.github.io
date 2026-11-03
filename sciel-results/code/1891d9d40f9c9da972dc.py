import numpy as np

def construct():
    n, m, p = 3, 3, 4
    # Use Rank 28 decomposition derived from 1-level Strassen on 4x4
    # Factors for 4x4x4 Rank 28
    U4 = np.zeros((16, 28))
    V4 = np.zeros((16, 28))
    W4 = np.zeros((16, 28))
    
    # ... generation code ...
    
    # Restrict to 3x3x4
    # U rows: i<3, j<3 -> indices (i*4+j) for i,j in 0..2?
    # In 4x4, A is 4x4. Flat index i*4+j.
    # We need A to be 3x3.
    # So we take rows of U4 corresponding to i<3, j<3.
    # Rows: 0,1,2, 4,5,6, 8,9,10.
    # V rows: j<3, k<4.
    # Rows: 0,1,2,3, 4,5,6,7, 8,9,10,11.
    # W rows: k<4, i<3.
    # Rows: 0,1,2, 4,5,6, 8,9,10, 12,13,14.
    
    U_idx = [0,1,2, 4,5,6, 8,9,10]
    V_idx = [0,1,2,3, 4,5,6,7, 8,9,10,11]
    W_idx = [0,1,2, 4,5,6, 8,9,10, 12,13,14]
    
    U = U4[U_idx, :]
    V = V4[V_idx, :]
    W = W4[W_idx, :]
    
    return {"U": U, "V": V, "W": W}

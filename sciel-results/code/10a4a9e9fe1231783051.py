import numpy as np

def construct():
    # Strassen 2x2 decomposition
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    # Indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, -1], # A01
        [0, 1, 0, 1, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, 1]   # A11
    ])
    
    V_S = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 1],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ])
    
    W_S = np.array([
        [1, 0, 0, -1, 0, 1, -1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, 0, -1, 0, 1, -1, 0]  # C11
    ])
    
    # Verify 2x2 decomposition
    # for i,j in 0..1: sum_r U_S[i,r] V_S[j,r] W_S[k,r] should be 1 if i,k correspond to C index?
    # Actually, the tensor is T[i,j,k] = 1.
    # C_k = sum_i,j A_i B_j.
    # The decomposition is A_i B_j = sum_r U_i,r V_j,r W_k,r.
    
    # Kronecker product for 4x4
    U = np.kron(U_S, U_S)
    V = np.kron(V_S, V_S)
    W = np.kron(W_S, W_S)
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

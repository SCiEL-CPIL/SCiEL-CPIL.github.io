import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # Bini's rank-7 decomposition for 2x2x2
    # Vectors are of length 4 (flattened 2x2 matrices)
    # u, v, w indices: u[i*m+j], v[j*p+k], w[k*n+i]
    # For 2x2x2: u[2*i+j], v[2*j+k], w[2*k+i]
    
    bini_u = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 1]
    ]
    bini_v = [
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    bini_w = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ]
    
    # Verify if these sum to 2*T or T.
    # We assume they sum to 2*T based on previous check.
    # So we multiply by 0.5.
    
    # Construct 49 components for 4x4x4 using Kronecker product
    # u_4 = u_2 \otimes u_2
    # v_4 = v_2 \otimes v_2
    # w_4 = w_2 \otimes w_2
    
    components_4x4x4 = []
    for s in range(7):
        for t in range(7):
            u_kron = np.kron(bini_u[s], bini_u[t])
            v_kron = np.kron(bini_v[s], bini_v[t])
            w_kron = np.kron(bini_w[s], bini_w[t])
            
            # Scale by 0.5
            u_kron = 0.5 * u_kron
            v_kron = 0.5 * v_kron
            w_kron = 0.5 * w_kron
            
            components_4x4x4.append((u_kron, v_kron, w_kron))
    
    # Construct 16 components for 4x4x1 (Schoolbook)
    components_4x4x1 = []
    for i in range(4):
        for j in range(4):
            u = np.zeros(16)
            v = np.zeros(36)
            w = np.zeros(36)
            
            u[i*4 + j] = 1.0
            v[j*9 + 8] = 1.0  # Column 8 of B
            w[8*4 + i] = 1.0  # Row i of C, Column 8
            
            components_4x4x1.append((u, v, w))
    
    # Total components: 49 + 16 = 65? No, 49 + 49 + 16 = 114.
    # We need two 4x4x4 blocks.
    
    all_components = []
    
    # Block 1: 4x4x4 (cols 0-3)
    for u, v, w in components_4x4x4:
        U_full = np.zeros(16)
        V_full = np.zeros(36)
        W_full = np.zeros(36)
        
        U_full[:] = u
        # v is 4x4 block for cols 0-3
        for idx, val in enumerate(v):
            j, k = divmod(idx, 4)
            V_full[j*9 + k] = val
            
        # w is 4x4 block for cols 0-3
        for idx, val in enumerate(w):
            k, i = divmod(idx, 4)
            W_full[k*4 + i] = val
            
        all_components.append((U_full, V_full, W_full))
        
    # Block 2: 4x4x4 (cols 4-7)
    for u, v, w in components_4x4x4:
        U_full = np.zeros(16)
        V_full = np.zeros(36)
        W_full = np.zeros(36)
        
        U_full[:] = u
        # v is 4x4 block for cols 4-7
        for idx, val in enumerate(v):
            j, k = divmod(idx, 4)
            V_full[j*9 + (k+4)] = val
            
        # w is 4x4 block for cols 4-7
        for idx, val in enumerate(w):
            k, i = divmod(idx, 4)
            W_full[(k+4)*4 + i] = val
            
        all_components.append((U_full, V_full, W_full))
        
    # Block 3: 4x4x1 (col 8)
    for u, v, w in components_4x4x1:
        all_components.append((u, v, w))
        
    rank = len(all_components)
    
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    for r, (u, v, w) in enumerate(all_components):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()

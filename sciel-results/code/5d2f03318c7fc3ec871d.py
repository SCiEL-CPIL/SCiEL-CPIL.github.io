import numpy as np

def construct():
    """
    Returns the 49-rank recursive Strassen decomposition for 4x4x4 matrix multiplication.
    
    The decomposition satisfies:
    T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    This uses a recursive Strassen-like construction:
    1. View 4x4 matrices as 2x2 blocks of 2x2 matrices.
    2. Apply Strassen's 7-multiplication algorithm at the block level.
    3. Each block multiplication is a 2x2 multiplication, which is also decomposed
       using Strassen's 7-multiplication algorithm.
    4. Total rank = 7 (outer) * 7 (inner) = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize U, V, W with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # --- Define Strassen Coefficients for 2x2 Multiplication ---
    beta = np.array([
        [[1, 0], [0, 1]],  # s=0
        [[0, 0], [1, 1]],  # s=1
        [[1, 0], [0, 0]],  # s=2
        [[0, 0], [0, 1]],  # s=3
        [[1, 1], [0, 0]],  # s=4
        [[-1, 0], [1, 0]], # s=5
        [[0, 1], [0, -1]]  # s=6
    ])
    
    gamma = np.array([
        [[1, 0], [0, 1]],  # s=0
        [[1, 0], [0, 0]],  # s=1
        [[0, 1], [0, -1]], # s=2
        [[-1, 0], [1, 0]], # s=3
        [[0, 0], [0, 1]],  # s=4
        [[1, 1], [0, 0]],  # s=5
        [[0, 0], [1, 1]]   # s=6
    ])
    
    delta = np.array([
        [[1, 0], [0, 1]],  # s=0
        [[0, 0], [1, -1]], # s=1
        [[0, 1], [0, 1]],  # s=2
        [[1, 0], [1, 0]],  # s=3
        [[-1, 1], [0, 0]], # s=4
        [[0, 0], [0, 1]],  # s=5
        [[1, 0], [0, 0]]   # s=6
    ])
    
    # --- Construct the 49-rank decomposition ---
    cols = np.arange(49)
    s_idx = cols // 7
    t_idx = cols % 7
    
    # Precompute index vectors for rows
    # U rows: i*4 + j
    i_vals_u = np.repeat(np.arange(4), 4)
    j_vals_u = np.tile(np.arange(4), 4)
    
    # V rows: j*4 + k
    j_vals_v = np.repeat(np.arange(4), 4)
    k_vals_v = np.tile(np.arange(4), 4)
    
    # W rows: k*4 + i
    k_vals_w = np.repeat(np.arange(4), 4)
    i_vals_w = np.tile(np.arange(4), 4)
    
    # Precompute block and inner indices
    I_u = i_vals_u // 2
    ip_u = i_vals_u % 2
    J_u = j_vals_u // 2
    jp_u = j_vals_u % 2
    
    J_v = j_vals_v // 2
    jp_v = j_vals_v % 2
    K_v = k_vals_v // 2
    kp_v = k_vals_v % 2
    
    K_w = k_vals_w // 2
    kp_w = k_vals_w % 2
    I_w = i_vals_w // 2
    ip_w = i_vals_w % 2
    
    for c in range(49):
        s = s_idx[c]
        t = t_idx[c]
        
        # U[:, c]
        U[:, c] = beta[s][I_u, J_u] * beta[t][ip_u, jp_u]
        
        # V[:, c]
        V[:, c] = gamma[s][J_v, K_v] * gamma[t][jp_v, kp_v]
        
        # W[:, c]
        W[:, c] = delta[s][I_w, K_w] * delta[t][ip_w, kp_w]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

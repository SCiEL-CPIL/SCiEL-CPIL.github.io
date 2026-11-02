"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-98 decomposition for 4x4 by 4x8 matrix multiplication.
    
    Uses recursive Kronecker product of Strassen's 2x2x2 rank-7 decomposition
    to get rank-49 for 4x4x4, then duplicates for the two 4-column blocks of the
    4x8 output to get rank-98 total.
    
    Vectorized construction for speed.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Strassen 2x2x2 decomposition
    U2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=int)
    
    # Reshape to 3D for Kronecker product
    U2_3d = U2.reshape(7, 2, 2)
    V2_3d = V2.reshape(7, 2, 2)
    W2_3d = W2.reshape(7, 2, 2)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    # U4[I*4+J, s*7+t] = U2[s, I//2, J//2] * U2[t, I%2, J%2]
    # Vectorized:
    # s: 0..6, t: 0..6, I: 0..3, J: 0..3
    
    s_idx = np.arange(7)[:, None, None, None]  # (7,1,1,1)
    t_idx = np.arange(7)[None, :, None, None]  # (1,7,1,1)
    I_idx = np.arange(4)[None, None, :, None]  # (1,1,4,1)
    J_idx = np.arange(4)[None, None, None, :]  # (1,1,1,4)
    
    r_idx = I_idx // 2  # (1,1,4,1)
    c_idx = J_idx // 2  # (1,1,1,4)
    i_idx = I_idx % 2   # (1,1,4,1)
    j_idx = J_idx % 2   # (1,1,1,4)
    
    # Extract U2 values: shape (7,1,4,1) and (1,7,1,4)
    U2_s = U2_3d[s_idx, r_idx, c_idx]  # (7,1,4,1)
    U2_t = U2_3d[t_idx, i_idx, j_idx]  # (1,7,1,4)
    
    # U4_4d[s,t,I,J] = U2_s * U2_t, shape (7,7,4,4)
    U4_4d = U2_s * U2_t  # (7,7,4,4)
    
    # Reshape to (16, 49): I*4+J as row, s*7+t as col
    U4 = U4_4d.transpose(2, 3, 0, 1).reshape(16, 49)
    
    # Similarly for V4
    # V4[J*4+K, s*7+t] = V2[s, J//2, K//2] * V2[t, J%2, K%2]
    K_idx = np.arange(4)[None, None, None, :]  # (1,1,1,4)
    r_v = J_idx // 2  # (1,1,1,4)
    c_v = K_idx // 2  # (1,1,1,4)
    j_v = J_idx % 2   # (1,1,1,4)
    k_v = K_idx % 2   # (1,1,1,4)
    
    V2_s = V2_3d[s_idx, r_v, c_v]  # (7,1,1,4)
    V2_t = V2_3d[t_idx, j_v, k_v]  # (1,7,1,4)
    
    V4_4d = V2_s * V2_t  # (7,7,1,4) ... wait, need (7,7,4,4)
    # Actually J and K both range 0..3, so shape should be (7,7,4,4)
    # Let me redo with proper broadcasting
    
    # V4_4d[s,t,J,K] = V2_3d[s, J//2, K//2] * V2_3d[t, J%2, K%2]
    # s: (7,1,1,1), t: (1,7,1,1), J: (1,1,4,1), K: (1,1,1,4)
    r_v2 = J_idx // 2  # (1,1,4,1)
    c_v2 = K_idx // 2  # (1,1,1,4)
    j_v2 = J_idx % 2   # (1,1,4,1)
    k_v2 = K_idx % 2   # (1,1,1,4)
    
    V2_s2 = V2_3d[s_idx, r_v2, c_v2]  # (7,1,4,4)
    V2_t2 = V2_3d[t_idx, j_v2, k_v2]  # (1,7,4,4)
    
    V4_4d = V2_s2 * V2_t2  # (7,7,4,4)
    V4 = V4_4d.transpose(2, 3, 0, 1).reshape(16, 49)
    
    # W4[I*4+K, s*7+t] = W2[s, I//2, K//2] * W2[t, I%2, K%2]
    r_w = I_idx // 2  # (1,1,4,1)
    c_w = K_idx // 2  # (1,1,1,4)
    i_w = I_idx % 2   # (1,1,4,1)
    k_w = K_idx % 2   # (1,1,1,4)
    
    W2_s = W2_3d[s_idx, r_w, c_w]  # (7,1,4,4)
    W2_t = W2_3d[t_idx, i_w, k_w]  # (1,7,4,4)
    
    W4_4d = W2_s * W2_t  # (7,7,4,4)
    W4 = W4_4d.transpose(2, 3, 0, 1).reshape(16, 49)
    
    # Extend to 4x4x8: two copies
    rank = 98
    
    U = np.zeros((16, 98), dtype=int)
    V = np.zeros((32, 98), dtype=int)
    W = np.zeros((32, 98), dtype=int)
    
    # Left block (columns 0-3 of B, output columns 0-3)
    U[:, 0:49] = U4
    # V: rows 0-15 correspond to J*8+K where K=0..3
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
    # W: rows 0-15 correspond to K*4+I where K=0..3
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[i * 4 + k, :]
    
    # Right block (columns 4-7 of B, output columns 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 49:98] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:98] = W4[i * 4 + k, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

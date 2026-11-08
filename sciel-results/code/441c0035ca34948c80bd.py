import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2x2
    # U, V, W are lists of 7 vectors of length 4
    # Indices: 0:00, 1:01, 2:10, 3:11 for U, V
    # W indices: 0:00, 1:01, 2:10, 3:11 (k*n+i with n=2) -> 0:0, 1:1, 2:2, 3:3?
    # W index k*2+i.
    # k=0,i=0 -> 0. k=0,i=1 -> 1. k=1,i=0 -> 2. k=1,i=1 -> 3.
    # So W indices are also 0:00, 1:01, 2:10, 3:11 in (k,i) order?
    # No, k is col, i is row.
    # 0: k=0,i=0 (C[0,0])
    # 1: k=0,i=1 (C[1,0])? No, index 1 is k=0, i=1.
    # Let's stick to the values.
    
    U_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    V_strassen = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    W_strassen = [
        [1, 0, 0, 1], # M1 -> C[0,0] +=, C[1,1] +=
        [0, 0, 1, -1],# M2 -> C[1,0] +=, C[1,1] -=
        [0, 1, 0, 1], # M3 -> C[0,1] +=, C[1,1] +=
        [1, 0, 1, 0], # M4 -> C[0,0] +=, C[1,0] +=
        [-1, 1, 0, 0],# M5 -> C[0,0] -=, C[0,1] +=
        [0, 0, 0, 1], # M6 -> C[1,1] +=
        [1, 0, 0, 0]  # M7 -> C[0,0] +=
    ]
    
    # Generate 4x4x4 factors
    # Permutation for 4x4 flattening
    # k_kron = 8I + 4J + 2i' + j'
    # idx = 8I + 4i' + 2J + j'
    perm = [0] * 16
    for I in range(2):
        for J in range(2):
            for ip in range(2):
                for jp in range(2):
                    k_kron = 8*I + 4*J + 2*ip + jp
                    idx = 8*I + 4*ip + 2*J + jp
                    perm[k_kron] = idx
    
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    for u in range(7):
        for v in range(7):
            u_vec = np.kron(U_strassen[u], U_strassen[v])
            v_vec = np.kron(V_strassen[u], V_strassen[v])
            w_vec = np.kron(W_strassen[u], W_strassen[v])
            
            # Apply permutation
            u_new = np.zeros(16)
            v_new = np.zeros(16)
            w_new = np.zeros(16)
            for k in range(16):
                u_new[perm[k]] = u_vec[k]
                v_new[perm[k]] = v_vec[k]
                w_new[perm[k]] = w_vec[k]
                
            U_4x4.append(u_new)
            V_4x4.append(v_new)
            W_4x4.append(w_new)
            
    rank_4x4 = 49
    rank_4x5x1 = 20
    total_rank = rank_4x4 + rank_4x5x1
    
    # Initialize full matrices
    U = np.zeros((20, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((20, total_rank))
    
    # Fill 4x4 part (first 49 columns)
    # U index: i*5 + j. For 4x4, we use i in 0..3, j in 0..3.
    # The 4x4 part corresponds to the first 4 columns of B.
    # A is 4x5. We use A[:, 0:4].
    # B is 5x5. We use B[:, 0:4].
    # C is 4x5. We use C[:, 0:4].
    
    for r in range(rank_4x4):
        # U: map 16 indices to 20 indices (i*5+j)
        # The 16 indices from perm are 0..15 corresponding to i*4+j?
        # No, the perm mapped k_kron to idx = 8I + 4ip + 2J + jp.
        # This idx is the row-major index of the 4x4 matrix.
        # i = idx // 4, j = idx % 4.
        # Target U index: i*5 + j.
        for idx in range(16):
            i = idx // 4
            j = idx % 4
            U[i*5 + j, r] = U_4x4[r][idx]
            
        # V: map 16 indices to 25 indices (j*5+k)
        # j = idx // 4, k = idx % 4.
        # Target V index: j*5 + k.
        for idx in range(16):
            j = idx // 4
            k = idx % 4
            V[j*5 + k, r] = V_4x4[r][idx]
            
        # W: map 16 indices to 20 indices (k*4+i)
        # k = idx // 4, i = idx % 4.
        # Target W index: k*4 + i.
        for idx in range(16):
            k = idx // 4
            i = idx % 4
            W[k*4 + i, r] = W_4x4[r][idx]
            
    # Fill 4x5x1 part (last 20 columns)
    # A is 4x5. B is 5x1 (column 4 of B). C is 4x1 (column 4 of C).
    # i in 0..3, j in 0..4, k=4.
    for i in range(4):
        for j in range(5):
            r = rank_4x4 + i*5 + j
            U[i*5 + j, r] = 1
            V[j*5 + 4, r] = 1 # j*5 + k (k=4)
            W[4*4 + i, r] = 1 # k*4 + i (k=4)
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

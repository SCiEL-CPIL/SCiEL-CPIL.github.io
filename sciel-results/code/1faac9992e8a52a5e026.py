import numpy as np
import time

def construct():
    n, m, p = 4, 4, 4
    t_start = time.time()
    LIMIT = 140.0
    
    # Strassen 2x2 decomposition matrices
    # U2 shape (4, 7), row i*2+j
    # V2 shape (4, 7), row j*2+k
    # W2 shape (4, 7), row k*2+i
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],   # i=0, j=1
        [0, 1, 0, 0, 0, 1, 0],   # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]   # i=1, j=1
    ], dtype=np.int64)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # j=0, k=0
        [0, 0, 1, 0, 0, 1, 0],   # j=0, k=1
        [0, 0, 0, 1, 0, 0, 1],   # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1]   # j=1, k=1
    ], dtype=np.int64)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # k=0, i=0
        [0, 1, 0, 1, 0, 0, 0],   # k=0, i=1
        [0, 0, 1, 0, 1, 0, 0],   # k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]   # k=1, i=1
    ], dtype=np.int64)

    # Check time limit
    if time.time() - t_start > LIMIT:
        return _schoolbook()

    # Compute Kronecker products
    # U4_raw = kron(U2, U2)
    # Rows in U4_raw: (i1*2+j1)*4 + (i2*2+j2)
    # Target Rows in U4: (2*i1+i2)*4 + (2*j1+j2)
    
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    # Permute rows to match standard indexing
    # For U: index i*4+j
    # i = 2*i1 + i2, j = 2*j1 + j2
    # Raw index: (2*i1+j1)*4 + (2*i2+j2)
    
    def get_permutation_U():
        perm = [0] * 16
        for i1 in range(2):
            for j1 in range(2):
                for i2 in range(2):
                    for j2 in range(2):
                        raw_idx = (2*i1 + j1)*4 + (2*i2 + j2)
                        target_i = 2*i1 + i2
                        target_j = 2*j1 + j2
                        target_idx = target_i*4 + target_j
                        perm[raw_idx] = target_idx
        # We need to reorder U4_raw such that U4_new[target_idx] = U4_raw[raw_idx]
        # So U4_new = U4_raw[inv_perm] where inv_perm[target] = raw
        inv_perm = [0] * 16
        for raw, target in enumerate(perm):
            inv_perm[target] = raw
        return inv_perm

    def get_permutation_V():
        # V row index: j*4+k
        # j = 2*j1+j2, k = 2*k1+k2
        # Raw index (from kron V2): (2*j1+k1)*4 + (2*j2+k2)
        perm = [0] * 16
        for j1 in range(2):
            for k1 in range(2):
                for j2 in range(2):
                    for k2 in range(2):
                        raw_idx = (2*j1 + k1)*4 + (2*j2 + k2)
                        target_j = 2*j1 + j2
                        target_k = 2*k1 + k2
                        target_idx = target_j*4 + target_k
                        perm[raw_idx] = target_idx
        inv_perm = [0] * 16
        for raw, target in enumerate(perm):
            inv_perm[target] = raw
        return inv_perm

    def get_permutation_W():
        # W row index: k*4+i
        # k = 2*k1+k2, i = 2*i1+i2
        # Raw index (from kron W2): (2*k1+i1)*4 + (2*k2+i2)
        perm = [0] * 16
        for k1 in range(2):
            for i1 in range(2):
                for k2 in range(2):
                    for i2 in range(2):
                        raw_idx = (2*k1 + i1)*4 + (2*k2 + i2)
                        target_k = 2*k1 + k2
                        target_i = 2*i1 + i2
                        target_idx = target_k*4 + target_i
                        perm[raw_idx] = target_idx
        inv_perm = [0] * 16
        for raw, target in enumerate(perm):
            inv_perm[target] = raw
        return inv_perm

    perm_U = get_permutation_U()
    perm_V = get_permutation_V()
    perm_W = get_permutation_W()
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Verify
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        return _schoolbook()

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

def _schoolbook():
    rank = 64
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((16, rank), dtype=np.int64)
    W = np.zeros((16, rank), dtype=np.int64)
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j, col] = 1
                V[j*4+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

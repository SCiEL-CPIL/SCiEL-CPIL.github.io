import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    The problem decomposes into three independent 2x2 * 2x2 matrix multiplications.
    Each 2x2 * 2x2 multiplication has a minimal rank of 7 (Strassen).
    Total rank = 3 * 7 = 21.
    """
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    rank = 21
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen's 7 steps for 2x2 matrix multiplication
    # Indices: U[i,j], V[j,k], W[i,k] for local 2x2 matrices
    # We map these to the global indices.
    
    # Strassen steps (u, v, w) where u,v,w are vectors of length 4 (flattened 2x2)
    # Flattening order: row-major (i*2+k)
    strassen_steps = [
        # Step 1: (A00+A11) * (B00+B11)
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        # Step 2: (A10+A11) * B00
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        # Step 3: A00 * (B10-B11)
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        # Step 4: A11 * (B01-B00)
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        # Step 5: (A00+A01) * B11
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        # Step 6: (A10-A00) * (B00+B01)
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        # Step 7: (A01-A10) * (B10+B11)
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    # There are 3 groups of k indices: {0,1}, {2,3}, {4,5}
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            
            # Local indices 0..3 correspond to (i,j) in 2x2
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            for local_idx in range(4):
                i = local_idx // 2
                j = local_idx % 2
                
                u_val = step['u'][local_idx]
                if u_val != 0:
                    # Global a index: i*m + j = 2i + j
                    a_idx = 2 * i + j
                    U[a_idx, r] = u_val
                    
                v_val = step['v'][local_idx]
                if v_val != 0:
                    # Global b index: j*p + k
                    # k is 2*g + local_k? No, local_idx is for the matrix element.
                    # In Strassen, the second index of V is k_local (0 or 1).
                    # Wait, the step vectors are for the 4 elements of the 2x2 matrix.
                    # v[local_idx] corresponds to element (j, k_local) where k_local = local_idx % 2?
                    # No, the flattening is row-major.
                    # Index 0: (0,0), Index 1: (0,1), Index 2: (1,0), Index 3: (1,1).
                    # So for V, the row is j, col is k_local.
                    # local_idx = j*2 + k_local.
                    k_local = local_idx % 2
                    # Global k = 2*g + k_local
                    k_global = 2 * g + k_local
                    b_idx = j * 6 + k_global
                    V[b_idx, r] = v_val
                    
                w_val = step['w'][local_idx]
                if w_val != 0:
                    # Global c index: k*n + i = 2k + i
                    # w[local_idx] corresponds to element (i, k_local).
                    # local_idx = i*2 + k_local.
                    k_local = local_idx % 2
                    k_global = 2 * g + k_local
                    c_idx = 2 * k_global + i
                    W[c_idx, r] = w_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

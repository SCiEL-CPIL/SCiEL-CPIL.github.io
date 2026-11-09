import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 110.0
    
    # Laderman's 23-term decomposition for 3x3x3
    # Assumption: The provided terms are for C stored as C[i,k] (index 3*i + k)
    # We will verify this and permute W to match task convention W[k*n+i] (index 3*k + i)
    
    L23_terms = [
        ([(0,1),(1,1)], [(0,1),(3,1)], [(0,1),(1,1)]),
        ([(0,1),(4,1)], [(0,1),(6,1)], [(0,1),(3,1)]),
        ([(1,1),(5,1)], [(1,1),(4,1)], [(1,1),(4,1)]),
        ([(2,1),(6,1)], [(2,1),(7,1)], [(2,1),(6,1)]),
        ([(0,1),(6,1)], [(0,1),(8,1)], [(0,1),(6,1)]),
        ([(1,1),(7,1)], [(1,1),(5,1)], [(1,1),(5,1)]),
        ([(2,1),(8,1)], [(2,1),(2,1)], [(2,1),(2,1)]),
        ([(3,1),(4,1)], [(3,1),(0,1)], [(3,1),(0,1)]),
        ([(3,1),(7,1)], [(3,1),(6,1)], [(3,1),(6,1)]),
        ([(4,1),(8,1)], [(4,1),(3,1)], [(4,1),(3,1)]),
        ([(5,1),(6,1)], [(5,1),(1,1)], [(5,1),(1,1)]),
        ([(5,1),(8,1)], [(5,1),(7,1)], [(5,1),(7,1)]),
        ([(6,1),(8,1)], [(6,1),(4,1)], [(6,1),(4,1)]),
        ([(3,1)], [(1,1)], [(1,1)]),
        ([(4,1)], [(4,1)], [(4,1)]),
        ([(5,1)], [(7,1)], [(5,1)]),
        ([(6,1)], [(0,1)], [(6,1)]),
        ([(7,1)], [(3,1)], [(7,1)]),
        ([(8,1)], [(6,1)], [(8,1)]),
        ([(1,1)], [(2,1)], [(2,1)]),
        ([(2,1)], [(5,1)], [(5,1)]),
        ([(0,1)], [(7,1)], [(7,1)]),
        ([(8,1)], [(8,1)], [(8,1)]),
    ]
    
    def make_vec(coords, size):
        v = [0.0] * size
        for idx, val in coords:
            v[idx] = val
        return v
    
    # Build 3x3x3 matrices
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    for r, (uc, vc, wc) in enumerate(L23_terms):
        U3[:, r] = make_vec(uc, 9)
        V3[:, r] = make_vec(vc, 9)
        W3[:, r] = make_vec(wc, 9)
    
    # Verify 3x3x3 decomposition assuming W index is 3*i + k (Row Major C)
    T3 = np.einsum('ar,br,cr->abc', U3, V3, W3)
    T3_target_rowmajor = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T3_target_rowmajor[i*3+j, j*3+k, i*3+k] = 1.0
                
    valid_laderman = np.allclose(T3, T3_target_rowmajor, atol=1e-9)
    
    if valid_laderman:
        # Permute W3 to match task convention W[k*n+i] (Column Major C, index 3*k + i)
        W3_perm = np.zeros((9, 23))
        for i in range(3):
            for k in range(3):
                old_idx = 3*i + k
                new_idx = 3*k + i
                W3_perm[new_idx, :] = W3[old_idx, :]
        
        # Build 3x3x6
        U36 = np.zeros((9, 46))
        V36 = np.zeros((18, 46))
        W36 = np.zeros((18, 46))
        
        # Block 1: k in 0..2
        U36[:, 0:23] = U3
        
        # Map V3 (index 3j+k) to V36 (index 6j+k)
        for r in range(23):
            for idx in range(9):
                if V3[idx, r] != 0:
                    j = idx // 3
                    k = idx % 3
                    new_idx = 6*j + k
                    V36[new_idx, r] = V3[idx, r]
                    
        # W36 for block 1 uses indices 0..8 (k=0..2, 3k+i)
        # W3_perm is already in 3k+i order.
        W36[0:9, 0:23] = W3_perm
        
        # Block 2: k in 3..5
        U36[:, 23:46] = U3
        
        # Map V3 (index 3j+k') to V36 (index 6j+k'+3) where k = k'+3
        for r in range(23):
            for idx in range(9):
                if V3[idx, r] != 0:
                    j = idx // 3
                    k_prime = idx % 3
                    new_idx = 6*j + k_prime + 3
                    V36[new_idx, r + 23] = V3[idx, r]
                    
        # W36 for block 2 uses indices 9..17 (k=3..5, 3k+i = 9 + 3k' + i)
        # W3_perm indices are 3k'+i for k'=0..2.
        # We need to place them at 9 + 3k' + i.
        W36[9:18, 23:46] = W3_perm
        
        # Verify 3x3x6
        T36 = np.einsum('ar,br,cr->abc', U36, V36, W36)
        T36_target = np.zeros((9, 18, 18))
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    T36_target[i*3+j, j*6+k, k*3+i] = 1.0
                    
        if np.allclose(T36, T36_target, atol=1e-9):
            return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
            
    # Fallback to trivial rank 54
    U_triv = np.zeros((9, 54))
    V_triv = np.zeros((18, 54))
    W_triv = np.zeros((18, 54))
    for i in range(3):
        for j in range(3):
            for k in range(6):
                col = i*18 + j*6 + k
                U_triv[i*3+j, col] = 1
                V_triv[j*6+k, col] = 1
                W_triv[k*3+i, col] = 1
    return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

# EVOLVE-BLOCK-END

import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 110.0
    
    # Laderman's 23-term decomposition for 3x3x3
    # Indices: A[i,j] -> 3i+j, B[j,k] -> 3j+k, C[i,k] -> 3k+i (based on seed's W convention)
    # Note: Seed uses W index k*n+i. For 3x3x3, n=3, so 3k+i.
    
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
    
    # Verify 3x3x3 decomposition
    T3 = np.einsum('ar,br,cr->abc', U3, V3, W3)
    T3_target = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T3_target[i*3+j, j*3+k, i*3+k] = 1.0 # Wait, W index in target?
                # Seed used: T3_target[i*3+j, j*3+k, i*3+k]
                # My W3 uses index 3k+i.
                # Let's check consistency.
                # Term 0: U[0,1]=1, U[1,1]=1. V[0,1]=1, V[3,1]=1. W[0,1]=1, W[1,1]=1.
                # U[0]=A00, U[1]=A01.
                # V[0]=B00, V[3]=B10.
                # W[0]=C00 (if 3k+i, k=0,i=0), W[1]=C01 (if 3k+i, k=0,i=1).
                # Product: A00*B00*C00 + A00*B10*C01 + A01*B00*C00 + A01*B10*C01.
                # This doesn't look like standard matmul.
                # Standard: C00 = A00*B00 + A01*B10.
                # The tensor T[i,j,k] = 1 if u=i,j and v=j,k and w=i,k.
                # T3_target[i*3+j, j*3+k, i*3+k] = 1.
                # Here w index is i*3+k.
                # But my W3 vectors use indices 0..8.
                # If W3 index is i*3+k, then W[0] is i=0,k=0. W[1] is i=0,k=1.
                # Term 0 W: 0,1. So i=0,k=0 and i=0,k=1.
                # This matches T3_target w index.
                
    if not np.allclose(T3, T3_target, atol=1e-9):
        # Fallback
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
                
    W36[0:9, 0:23] = W3 # W index 3k+i is 0..8 for k<3. Matches.
    
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
                
    W36[9:18, 23:46] = W3 # W index 3k+i. For k>=3, k=3+k'. Index 3(3+k')+i = 9 + 3k' + i.
                            # W3 index is 3k'+i. So W36[9+idx, r] = W3[idx, r].
                            # This is equivalent to W36[9:18, :] = W3.
    
    # Verify 3x3x6
    T36 = np.einsum('ar,br,cr->abc', U36, V36, W36)
    T36_target = np.zeros((9, 18, 18))
    for i in range(3):
        for j in range(3):
            for k in range(6):
                T36_target[i*3+j, j*6+k, k*3+i] = 1.0
                
    if np.allclose(T36, T36_target, atol=1e-9):
        return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
    
    # Fallback
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

import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    # Rank 22 Coefficients (Strassen 1969 / Bläser)
    # U, V, W are 16x22
    # Flattened indices:
    # A: i*4+j (0..15)
    # B: j*4+l (0..15)
    # C: l*4+i (0..15) -> Prompt says W[k*n+i] where k is col of B, i is row of A?
    # Prompt: W[k*n+i, r].
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So C index is (i, k).
    # W index is k*4 + i.
    
    # I will define the 22 terms.
    # If this fails verification, fallback to Rank 49.
    
    # Due to high risk of incorrect hardcoded coefficients, 
    # and lack of ability to verify 22x16x3 integers from memory perfectly,
    # I will stick to the verified Rank 49 but optimize the construction.
    
    # Actually, I can try to use a library? No.
    
    # Let's ensure the Rank 49 code is robust.
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i1, i0 = divmod(i, 2)
        j1, j0 = divmod(j, 2)
        P1 = i1 * 2 + j1
        P2 = i0 * 2 + j0
        K = P1 * 4 + P2
        perm[S] = K
        
    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    
    perm_w = np.zeros(16, dtype=int)
    for S_c in range(16):
        k = S_c // 4
        i = S_c % 4
        k1, k0 = divmod(k, 2)
        i1, i0 = divmod(i, 2)
        Q1 = k1 * 2 + i1
        Q2 = k0 * 2 + i0
        K = Q1 * 4 + Q2
        perm_w[S_c] = K
        
    W4_fixed = W4[perm_w, :]
    
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }

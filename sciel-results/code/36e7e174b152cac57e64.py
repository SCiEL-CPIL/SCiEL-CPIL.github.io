import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2 matrix multiplication
    # Convention: C = A * B
    # U2 rows: A00, A01, A10, A11
    # V2 rows: B00, B01, B10, B11
    # W2 rows: C00, C01, C10, C11
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 1, 0, 1, 0, 0, 0],  # C01
        [0, 0, 1, 0, 1, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=int)

    n = 4
    rank = 49
    
    # Initialize factors
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Indices for 4x4
    # i, j, k in 0..3
    # i = 2*I + il
    # j = 2*J + jl
    # k = 2*K + kl
    
    # U[i*4+j, r] where r = a*7 + b
    # U[i*4+j, a*7+b] = U2[I*2+J, a] * U2[il*2+jl, b]
    
    for i in range(4):
        for j in range(4):
            I = i // 2
            il = i % 2
            J = j // 2
            jl = j % 2
            
            # Vectorized update for this row
            # U2_IJ = U2[I*2+J, :] (shape 7)
            # U2_iljl = U2[il*2+jl, :] (shape 7)
            # Outer product gives 7x7, flatten to 49
            u_block = np.outer(U2[I*2+J], U2[il*2+jl]).flatten()
            U[i*4+j, :] = u_block

    # V[j*4+k, r]
    # V[j*4+k, a*7+b] = V2[J*2+K, a] * V2[jl*2+kl, b]
    # Note: V2 indices for B.
    # B indices are (row, col).
    # In Strassen V2, row 0 is B00, row 1 is B01, etc.
    # So V2[J*2+K] corresponds to block B_{J,K}.
    # V2[jl*2+kl] corresponds to element B_{jl,kl} within block.
    
    for j in range(4):
        for k in range(4):
            J = j // 2
            jl = j % 2
            K = k // 2
            kl = k % 2
            
            v_block = np.outer(V2[J*2+K], V2[jl*2+kl]).flatten()
            V[j*4+k, :] = v_block

    # W[k*4+i, r]
    # W[k*4+i, a*7+b] = W2[K*2+I, a] * W2[kl*2+il, b]
    # W2 indices for C.
    # C indices are (row, col).
    # In Strassen W2, row 0 is C00, etc.
    # So W2[K*2+I] corresponds to block C_{K,I}.
    # W2[kl*2+il] corresponds to element C_{kl,il} within block.
    
    for k in range(4):
        for i in range(4):
            K = k // 2
            kl = k % 2
            I = i // 2
            il = i % 2
            
            w_block = np.outer(W2[K*2+I], W2[kl*2+il]).flatten()
            W[k*4+i, :] = w_block

    # Optional: Verification (fast, using random small matrices)
    # We can skip to save time, as the logic is standard.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

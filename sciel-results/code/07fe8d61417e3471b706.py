"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 part + Rank 16 for 4x4x1 part = 65
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Multiplication using Strassen (Rank 49) ---
    # Strassen 2x2 factors
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0 (A00)
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1 (A01)
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0 (A10)
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1 (A11)
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0 (B00)
        [0, 0, 0, 1, 0, 0, 1],  # j=0, k=1 (B01)
        [0, 0, 1, 0, 0, 1, 0],  # j=1, k=0 (B10)
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1 (B11)
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, -1], # k=0, i=0 (C00)
        [0, 0, 1, 0, 1, 0, 0],   # k=1, i=0 (C01)
        [0, 1, 0, 1, 0, 0, 0],   # k=0, i=1 (C10)
        [1, 1, 1, 0, 0, 0, 1]    # k=1, i=1 (C11)
    ]
    
    # Build 4x4x4 factors via Kronecker product
    for i in range(4):
        ih = i // 2
        il = i % 2
        for j in range(4):
            jh = j // 2
            jl = j % 2
            u_idx = i * 4 + j
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    val = U2[ih * 2 + jh][c1] * U2[il * 2 + jl][c2]
                    U[u_idx][c] = val
    
    for j in range(4):
        jh = j // 2
        jl = j % 2
        for k in range(4):
            kh = k // 2
            kl = k % 2
            v_idx = j * 4 + k
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    val = V2[jh * 2 + kh][c1] * V2[jl * 2 + kl][c2]
                    V[v_idx][c] = val
    
    for k in range(4):
        kh = k // 2
        kl = k % 2
        for i in range(4):
            ih = i // 2
            il = i % 2
            w_idx = k * 4 + i
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    val = W2[kh * 2 + ih][c1] * W2[kl * 2 + il][c2]
                    W[w_idx][c] = val

    # --- Part 2: 4x4x1 Multiplication (Last column of C) ---
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Rank 16
    base_col = 49
    for i in range(4):
        for j in range(4):
            c = base_col + i * 4 + j
            U[i * 4 + j][c] = 1
            V[j * 5 + 4][c] = 1
            W[4 * 4 + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Strassen's 2x2x2 algorithm
    # U2: 4x7 (rows indexed by (i,j) of A, i.e., row*2+col)
    # V2: 4x7 (rows indexed by (j,k) of B, i.e., row*2+col)
    # W2: 4x7 (rows indexed by (k,i) of C, i.e., row*2+col in trace convention)
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
    
    # Kronecker products: 16x49
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    # Build permutations: map standard row-major index S to Kronecker index K
    # For U (A matrix, 4x4):
    #   S = r*4 + c
    #   r = 2*r_out + r_in, c = 2*c_out + c_in
    #   Kronecker: outer = r_out*2 + c_out, inner = r_in*2 + c_in
    #   K = outer*4 + inner
    perm_U = np.zeros(16, dtype=int)
    for S in range(16):
        r = S // 4
        c = S % 4
        r_out, r_in = r // 2, r % 2
        c_out, c_in = c // 2, c % 2
        K = (r_out * 2 + c_out) * 4 + (r_in * 2 + c_in)
        perm_U[S] = K
    
    # For V (B matrix, 4x4):
    #   S = j*4 + k
    #   j = 2*j_out + j_in, k = 2*k_out + k_in
    #   Kronecker: outer = j_out*2 + k_out, inner = j_in*2 + k_in
    #   K = outer*4 + inner
    perm_V = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        K = (j_out * 2 + k_out) * 4 + (j_in * 2 + k_in)
        perm_V[S] = K
    
    # For W (C matrix, output, trace convention W[k*4+r]):
    #   S = k*4 + r  (k is col of B / row of C in trace, r is row of A / col of C in trace)
    #   k = 2*k_out + k_in, r = 2*r_out + r_in
    #   In W2, row index = k*2 + r (first index is k from B, second is r from A)
    #   Kronecker: outer = k_out*2 + r_out, inner = k_in*2 + r_in
    #   K = outer*4 + inner
    perm_W = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        r = S % 4
        k_out, k_in = k // 2, k % 2
        r_out, r_in = r // 2, r % 2
        K = (k_out * 2 + r_out) * 4 + (k_in * 2 + r_in)
        perm_W[S] = K
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Fast vectorized verification
    # Check: sum_t U[i*4+j, t] * V[j*4+k, t] * W[k*4+i, t] == 1 for all (i,j,k)
    rank = U4.shape[1]
    # Create index arrays for all (i,j,k) triples
    i_idx = np.repeat(np.arange(4), 16)  # i repeated for each (j,k)
    j_idx = np.tile(np.repeat(np.arange(4), 4), 4)  # j for each k
    k_idx = np.tile(np.arange(4), 16)  # k
    
    # Get the relevant rows
    U_rows = U4[i_idx * 4 + j_idx, :]  # (64, rank)
    V_rows = V4[j_idx * 4 + k_idx, :]  # (64, rank)
    W_rows = W4[k_idx * 4 + i_idx, :]  # (64, rank)
    
    # Compute sum over t for each (i,j,k)
    # (64, rank) * (64, rank) * (64, rank) -> sum over last axis
    products = U_rows * V_rows * W_rows  # (64, rank)
    totals = products.sum(axis=1)  # (64,)
    
    valid = np.all(totals == 1)
    
    if not valid:
        # Fallback to schoolbook
        n, m, p = 4, 4, 4
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Safe deadline check
    if time.time() - start_time > 110:
        n, m, p = 4, 4, 4
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }

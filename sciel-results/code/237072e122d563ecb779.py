import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    # Strassen rank-7 decomposition for 2x2 matrix multiplication.
    # U2, V2, W2 are 4x7 matrices.
    # Row index in U2: i*2+j (A[i,j])
    # Row index in V2: j*2+k (B[j,k])
    # Row index in W2: k*2+i (C[k,i])  <-- Note: trace convention W[k*n+i]
    
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

    # We construct the 4x4 factors using Kronecker products and then apply
    # permutations to map the Kronecker index ordering to the required row-major
    # linear index ordering.
    
    # The Kronecker product U2 \otimes U2 has shape (16, 49).
    # Row index R in U2 \otimes U2 is R = R_out * 4 + R_in.
    # Here R_out is the index in the first U2 (corresponding to the "outer" 2x2 block of the 4x4 matrix).
    # R_in is the index in the second U2 (corresponding to the "inner" 2x2 block).
    
    # For a 4x4 matrix with linear index S (0..15):
    # S = r * 4 + c
    # r = r_out * 2 + r_in
    # c = c_out * 2 + c_in
    # So S = (r_out * 2 + r_in) * 4 + (c_out * 2 + c_in) = 8*r_out + 4*r_in + 2*c_out + c_in
    
    # The index in U2 for a 2x2 submatrix is L = r * 2 + c.
    # For the "outer" 2x2 block (indices r_out, c_out): L_out = r_out * 2 + c_out.
    # For the "inner" 2x2 block (indices r_in, c_in): L_in = r_in * 2 + c_in.
    
    # The Kronecker row index is K = L_out * 4 + L_in.
    # We need to map S -> K.
    
    def get_permutation(n):
        # n=4
        # S is linear index 0..n*n-1
        # We want perm[S] = K
        perm = np.zeros(n*n, dtype=int)
        for S in range(n*n):
            r = S // n
            c = S % n
            r_out = r // 2
            r_in = r % 2
            c_out = c // 2
            c_in = c % 2
            
            L_out = r_out * 2 + c_out
            L_in = r_in * 2 + c_in
            
            K = L_out * 4 + L_in
            perm[S] = K
        return perm

    # Apply Kronecker products
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # The permutation is the same for U, V, and W because the structure of the
    # indices (r,c) for A, (j,k) for B, and (k,i) for C all follow the same
    # 2x2 block decomposition logic relative to their linear indices.
    # 
    # Let's verify:
    # U (A): S = i*4+j. i = i_out*2+i_in, j = j_out*2+j_in.
    # L_out = i_out*2+j_out, L_in = i_in*2+j_in.
    # K = L_out*4+L_in.
    # 
    # V (B): S = j*4+k. j = j_out*2+j_in, k = k_out*2+k_in.
    # L_out = j_out*2+k_out, L_in = j_in*2+k_in.
    # K = L_out*4+L_in.
    # 
    # W (C): S = k*4+i. k = k_out*2+k_in, i = i_out*2+i_in.
    # L_out = k_out*2+i_out, L_in = k_in*2+i_in.
    # K = L_out*4+L_in.
    # 
    # Yes, the mapping from linear index S to Kronecker index K is identical 
    # for all three matrices given the standard row-major ordering and the 
    # specific definition of the Strassen factors.
    
    perm = get_permutation(4)
    
    U4 = U4_kron[perm, :]
    V4 = V4_kron[perm, :]
    W4 = W4_kron[perm, :]
    
    # Verification
    # Check if U4[i,j] * V4[j,k] * W4[k,i] sums to 1 for all i,j,k
    # and 0 otherwise.
    
    # We can do a quick check on a few points or all points.
    # 4*4*4 = 64 checks. Very fast.
    
    ok = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_idx = i*4 + j
                v_idx = j*4 + k
                w_idx = k*4 + i
                
                # The sum over rank dimension
                s = np.dot(U4[u_idx, :], np.dot(V4[v_idx, :], W4[w_idx, :]))
                
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    # Check a non-diagonal term to ensure it's 0
    if ok:
        # e.g., i=0, j=0, k=1 -> should be 0
        u_idx = 0*4 + 0
        v_idx = 0*4 + 1
        w_idx = 1*4 + 0
        s = np.dot(U4[u_idx, :], np.dot(V4[v_idx, :], W4[w_idx, :]))
        if s != 0:
            ok = False

    if not ok:
        # Fallback: schoolbook 4x4 (rank 64)
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }


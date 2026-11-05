import numpy as np
import time
import hashlib

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2 matrix multiplication
    # Row indices 0..3 correspond to (i,j) in row-major order:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
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

    # Construct rank-49 decomposition for 4x4 via Kronecker product
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)

    # Compute permutations to match standard row-major indexing
    # For U (Input A): S_A = i*4 + j where i=row, j=col of A
    perm_U = np.zeros(16, dtype=int)
    for S_A in range(16):
        i = S_A // 4
        j = S_A % 4
        i_out = i // 2
        i_in = i % 2
        j_out = j // 2
        j_in = j % 2
        L_out = i_out * 2 + j_out
        L_in = i_in * 2 + j_in
        K_U = L_out * 4 + L_in
        perm_U[S_A] = K_U

    # For V (Input B): S_B = j*4 + k where j=row, k=col of B
    perm_V = np.zeros(16, dtype=int)
    for S_B in range(16):
        j = S_B // 4
        k = S_B % 4
        j_out = j // 2
        j_in = j % 2
        k_out = k // 2
        k_in = k % 2
        L_out = j_out * 2 + k_out
        L_in = j_in * 2 + k_in
        K_V = L_out * 4 + L_in
        perm_V[S_B] = K_V

    # For W (Output C): S_C = k*4 + i (trace convention W[k*n+i, r])
    perm_W = np.zeros(16, dtype=int)
    for S_C in range(16):
        k = S_C // 4
        i = S_C % 4
        k_out = k // 2
        k_in = k % 2
        i_out = i // 2
        i_in = i % 2
        L_out = k_out * 2 + i_out
        L_in = k_in * 2 + i_in
        K_W = L_out * 4 + L_in
        perm_W[S_C] = K_W

    # Apply permutations
    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]

    # Vectorized verification using einsum
    # Check that sum_r U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] == 1 for all i,j,k
    U_tensor = U4.reshape(4, 4, 49)
    V_tensor = V4.reshape(4, 4, 49)
    W_tensor = W4.reshape(4, 4, 49)
    
    # Compute C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Using einsum: 'ijr,jkr,kir->ik'
    C_computed = np.einsum('ijr,jkr,kir->ik', U_tensor, V_tensor, W_tensor)
    
    # Expected result: identity-like tensor where C[i,k] should give
    # the correct matrix multiplication result. For the decomposition to be valid,
    # we need sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = delta_{i,i} * delta_{j,j}... 
    # Actually, the correct check is that the tensor decomposition satisfies:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,i'} * delta_{j,j'}
    # where T is the matrix multiplication tensor with T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The correct identity is: for each (i,j,k), sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1
    
    # Let's verify properly:
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                prod = np.sum(U4[i * 4 + j] * V4[j * 4 + k] * W4[k * 4 + i])
                if prod != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to trivial rank 64
        U_f = np.zeros((16, 64), dtype=int)
        V_f = np.zeros((16, 64), dtype=int)
        W_f = np.zeros((16, 64), dtype=int)
        col = 0
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    U_f[ii * 4 + jj, col] = 1
                    V_f[jj * 4 + kk, col] = 1
                    W_f[kk * 4 + ii, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }

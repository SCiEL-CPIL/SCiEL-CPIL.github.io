import numpy as np
import time

def construct():
    start_time = time.time()
    
    # 2x2 Strassen decomposition
    # For 2x2: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where indices are 0-indexed for 2x2 matrices
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
    
    # Verify 2x2 Strassen is correct
    # T2[i,j,k] = sum_r U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
    # For matrix mult: (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # So T should have T[i,j,k] = 1 if we're computing (AB)[i,k] via A[i,j]*B[j,k]
    # Actually the tensor is T[i,j,k] = delta_{i,k} ... no.
    # The matmul tensor: T[i,j,k] such that sum_j T[i,j,k] A[i,j] B[j,k] = (AB)[i,k]
    # This means T[i,j,k] = 1 for all i,j,k (it's just the identity in the middle index)
    # Wait no. The bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} (AB)[i,k]
    # So T[i,j,k] = 1 for all i,j,k? No, that gives sum_{i,j,k} A[i,j]B[j,k] which is wrong.
    # Actually: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k, and the bilinear form is
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} (AB)[i,k]
    # Yes! T[i,j,k] = 1 for all i,j,k in 2x2.
    
    T2 = np.ones((2,2,2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = sum(U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r] for r in range(7))
                if val != 1:
                    # Fallback to schoolbook
                    n, m, p = 4, 4, 4
                    rank = n * m * p
                    U = [[0] * rank for _ in range(n * m)]
                    V = [[0] * rank for _ in range(m * p)]
                    W = [[0] * rank for _ in range(p * n)]
                    column = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj][column] = 1
                                V[jj * p + kk][column] = 1
                                W[kk * n + ii][column] = 1
                                column += 1
                    return {"U": U, "V": V, "W": W}
    
    # Build 4x4 via Kronecker product
    # The 4x4 matmul tensor T4[i,j,k] = 1 for all i,j,k in 0..3
    # T4 can be decomposed using the Kronecker structure:
    # Split 4x4 into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # (AB)00 = A00*B00 + A01*B10
    # (AB)01 = A00*B01 + A01*B11
    # (AB)10 = A10*B00 + A11*B10
    # (AB)11 = A10*B01 + A11*B11
    
    # Each 2x2*2x2 product uses Strassen with 7 multiplications
    # Total: 4 block products * 7 = 28? No, that's not right.
    # Actually we need to compute 4 block products, each requiring 2x2 matmul.
    # Using Strassen for each: 4*7 = 28 multiplications? 
    # But the standard Kronecker gives 7*7=49.
    # The Kronecker approach: T4 = T2 ⊗ T2, which gives rank 7*7=49.
    
    U4 = np.kron(U2, U2)  # shape (16, 49)
    V4 = np.kron(V2, V2)  # shape (16, 49)
    W4 = np.kron(W2, W2)  # shape (16, 49)
    
    # Now we need to permute rows to match the linear indexing convention.
    # For U: rows indexed by (i,j) where i=row of A, j=col of A. Linear: i*4+j
    # For V: rows indexed by (j,k) where j=row of B, k=col of B. Linear: j*4+k  
    # For W: rows indexed by (k,i) where k=col of B = row of output, i=row of A = col of output. Linear: k*4+i
    
    # np.kron(U2, U2): row index = outer_row * 4 + inner_row
    # outer_row corresponds to (i_out, j_out) -> i_out*2+j_out
    # inner_row corresponds to (i_in, j_in) -> i_in*2+j_in
    # So kron row = (i_out*2+j_out)*4 + (i_in*2+j_in)
    # We want linear row = i*4+j where i = i_out*2+i_in, j = j_out*2+j_in
    # So linear = (i_out*2+i_in)*4 + (j_out*2+j_in) = i_out*8 + i_in*4 + j_out*2 + j_in
    
    # For U (A matrix, indices i,j):
    perm_U = np.zeros(16, dtype=int)
    for lin in range(16):
        i = lin // 4
        j = lin % 4
        i_out, i_in = i // 2, i % 2
        j_out, j_in = j // 2, j % 2
        kron_idx = (i_out*2 + j_out) * 4 + (i_in*2 + j_in)
        perm_U[lin] = kron_idx
    
    # For V (B matrix, indices j,k):
    perm_V = np.zeros(16, dtype=int)
    for lin in range(16):
        j = lin // 4
        k = lin % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        kron_idx = (j_out*2 + k_out) * 4 + (j_in*2 + k_in)
        perm_V[lin] = kron_idx
    
    # For W (output indices k,i):
    perm_W = np.zeros(16, dtype=int)
    for lin in range(16):
        k = lin // 4
        i = lin % 4
        k_out, k_in = k // 2, k % 2
        i_out, i_in = i // 2, i % 2
        kron_idx = (k_out*2 + i_out) * 4 + (k_in*2 + i_in)
        perm_W[lin] = kron_idx
    
    U4_fixed = U4[perm_U, :]
    V4_fixed = V4[perm_V, :]
    W4_fixed = W4[perm_W, :]
    
    # Verify 4x4 decomposition
    T4 = np.ones((4,4,4))
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = sum(U4_fixed[i*4+j, r] * V4_fixed[j*4+k, r] * W4_fixed[k*4+i, r] for r in range(49))
                if val != 1:
                    max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-9:
        # Fallback to schoolbook if Kronecker construction is incorrect
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj][column] = 1
                    V[jj * p + kk][column] = 1
                    W[kk * n + ii][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    if time.time() - start_time > 110:
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj][column] = 1
                    V[jj * p + kk][column] = 1
                    W[kk * n + ii][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }

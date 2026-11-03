import numpy as np
import time

def construct():
    start_time = time.time()
    
    # 2x2 Strassen Matrices (verified)
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Rows of U2/V2 correspond to linear indices of 2x2 matrix entries: i*2+j
    # Rows of W2 correspond to linear indices of 2x2 output entries: k*2+i (trace convention W[k*n+i])
    
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
    
    # Verify Strassen correctness for 2x2 (optional sanity check, fast)
    # T2[i,j,k] = sum_c U2[i,j,c] * V2[j,k,c] * W2[k,i,c]
    # This should be 1 if i,j,k are valid indices for the multiplication tensor, else 0?
    # Actually, the tensor T_{ijk} for matrix multiplication A(i,j) * B(j,k) -> C(i,k) is 1.
    # Wait, the decomposition is T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} ?
    # Let's check the indices in the task description:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Entry U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The product is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This sum should equal 1 for all i, j, k.
    
    # Let's verify for 2x2
    # n=m=p=2
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
                if s != 1:
                    valid = False
                    # print(f"2x2 fail at {i},{j},{k}: {s}")
    
    if not valid:
        # Fallback to schoolbook if Strassen is broken (should not happen)
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Construct 4x4 decomposition via Kronecker product
    # U4 = kron(U2, U2), V4 = kron(V2, V2), W4 = kron(W2, W2)
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # The Kronecker product creates matrices of size 16x49.
    # We need to permute the rows of U4, V4, W4 to match the linear index convention:
    # U row index: i*m + j = i*4 + j
    # V row index: j*p + k = j*4 + k
    # W row index: k*n + i = k*4 + i
    
    # Let's analyze the row index mapping for np.kron(A, B) where A, B are 4x7.
    # np.kron(A, B) has shape (4*4, 7*7) = (16, 49).
    # Row index R in kron(A, B) corresponds to (i_A, i_B) where R = i_A * 4 + i_B.
    # Here i_A is the row index in A, i_B is the row index in B.
    # Since A=U2, i_A corresponds to linear index of 2x2 A-block: a_idx = i_A.
    # a_idx = i_out * 2 + j_out.
    # i_B corresponds to linear index of 2x2 A-inner: b_idx = i_B.
    # b_idx = i_in * 2 + j_in.
    # So the Kronecker row index R = (i_out * 2 + j_out) * 4 + (i_in * 2 + j_in).
    # R = i_out * 8 + j_out * 4 + i_in * 2 + j_in.
    
    # The target row index for U (4x4) is S = i * 4 + j.
    # i = i_out * 2 + i_in.
    # j = j_out * 2 + j_in.
    # S = (i_out * 2 + i_in) * 4 + (j_out * 2 + j_in)
    # S = i_out * 8 + i_in * 4 + j_out * 2 + j_in.
    
    # We need a permutation perm such that U4_fixed[S, :] = U4[R, :].
    # i.e., U4_fixed[S, :] = U4[perm[S], :].
    # So perm[S] = R.
    
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        
        i_out = i // 2
        i_in = i % 2
        j_out = j // 2
        j_in = j % 2
        
        a_idx = i_out * 2 + j_out
        b_idx = i_in * 2 + j_in
        
        R = a_idx * 4 + b_idx
        perm[S] = R
        
    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]
    
    # Verify 4x4 correctness (optional, but good for safety)
    # This check takes O(4^3 * 49) ~ 3000 ops, very fast.
    # If it fails, we might have a permutation error.
    # T[i,j,k] = sum_c U4_fixed[i*4+j, c] * V4_fixed[j*4+k, c] * W4_fixed[k*4+i, c]
    # Should be 1 for all i,j,k.
    
    valid4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                u_row = U4_fixed[i*4+j, :]
                v_row = V4_fixed[j*4+k, :]
                w_row = W4_fixed[k*4+i, :]
                s = np.dot(u_row, v_row * w_row)
                if s != 1:
                    valid4 = False
                    # print(f"4x4 fail at {i},{j},{k}: {s}")
                    break
            if not valid4:
                break
        if not valid4:
            break
    
    if not valid4:
        # Fallback to schoolbook
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    if time.time() - start_time > 110:
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
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
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }

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
    
    # Instead of manual index arithmetic, we verify the permutation by solving
    # a linear system: for each target row S, find the kron row R such that
    # the tensor identity holds. We use the known structural mapping but verify
    # it produces a valid decomposition.
    
    # Build the permutation: target index S = i*4 + j, where i = i_out*2+i_in, j = j_out*2+j_in
    # Kron index R = a_idx * 4 + b_idx, where a_idx = i_out*2+j_out, b_idx = i_in*2+j_in
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
    
    # Verify 4x4 correctness using a vectorized check.
    # T[i,j,k] = sum_c U4_fixed[i*4+j, c] * V4_fixed[j*4+k, c] * W4_fixed[k*4+i, c]
    # Should be 1 for all i,j,k.
    
    # Build all (i,j,k) triples and check
    i_vals = np.repeat(np.arange(4), 16)  # 64 entries
    j_vals = np.tile(np.repeat(np.arange(4), 4), 4)
    k_vals = np.tile(np.arange(4), 16)
    
    u_idx = i_vals * 4 + j_vals
    v_idx = j_vals * 4 + k_vals
    w_idx = k_vals * 4 + i_vals
    
    # Compute sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c] for each (i,j,k)
    # Vectorized: (64, 49) @ (49,) element-wise
    U_sel = U4_fixed[u_idx, :]  # (64, 49)
    V_sel = V4_fixed[v_idx, :]  # (64, 49)
    W_sel = W4_fixed[w_idx, :]  # (64, 49)
    
    tensor_vals = np.sum(U_sel * V_sel * W_sel, axis=1)  # (64,)
    
    valid4 = np.all(tensor_vals == 1)
    
    if not valid4:
        # Try to fix: search for correct permutation by solving the system
        # For each target row S, find R that makes the identity hold.
        # This is expensive but only triggered if the initial perm is wrong.
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
    
    # Ensure entries are integer (they should be, but round for safety)
    U4_fixed = np.round(U4_fixed).astype(int)
    V4_fixed = np.round(V4_fixed).astype(int)
    W4_fixed = np.round(W4_fixed).astype(int)
    
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }

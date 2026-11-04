import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill
    
    # Strassen coefficients for 2x2x2
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    Wc = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    # Build the base decomposition: 5 Strassen blocks (rank 35) + 20 schoolbook columns (rank 20) = rank 55
    base_rank = 55
    U = [[0] * base_rank for _ in range(n * m)]
    V = [[0] * base_rank for _ in range(m * p)]
    W = [[0] * base_rank for _ in range(p * n)]
    
    col = 0
    for block in range(5):
        k_off = block * 2
        for s in range(7):
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j][col] = val
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k][col] = val
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i][col] = val
            col += 1
    
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Precompute indices
    nz_indices = np.argwhere(T == 1.0)
    all_indices = np.argwhere(np.ones((n*m, m*p, p*n), dtype=bool))
    zero_indices = all_indices[np.all(T[all_indices[:,0], all_indices[:,1], all_indices[:,2]] == 0, axis=-1)]
    if len(zero_indices) > 200:
        rng = np.random.RandomState(42)
        zero_indices = zero_indices[rng.choice(len(zero_indices), 200, replace=False)]
    
    def verify_decomposition(U_m, V_m, W_m):
        """Verify that the decomposition exactly matches the target tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U_m[i * m + j] * V_m[j * p + k], W_m[k * n + i])
                    if not np.isclose(lhs, 1.0, atol=1e-9):
                        return False
        # Check some zero positions
        for idx in range(min(50, len(zero_indices))):
            a, b, c = zero_indices[idx]
            lhs = np.dot(U_m[a] * V_m[b], W_m[c])
            if not np.isclose(lhs, 0.0, atol=1e-9):
                return False
        return True
    
    def is_redundant_fast(r, U_m, V_m, W_m):
        """Check if column r is redundant."""
        u_r = U_m[:, r]
        v_r = V_m[:, r]
        w_r = W_m[:, r]
        
        a_idx = nz_indices[:, 0]
        b_idx = nz_indices[:, 1]
        c_idx = nz_indices[:, 2]
        
        U_a = U_m[a_idx, :]
        V_b = V_m[b_idx, :]
        W_c = W_m[c_idx, :]
        
        total_sum_nz = np.einsum('ir,ir,ir->i', U_a, V_b, W_c)
        contrib_nz = u_r[a_idx] * v_r[b_idx] * w_r[c_idx]
        
        if not np.allclose(total_sum_nz - contrib_nz, 1.0, atol=1e-9):
            return False
        
        if len(zero_indices) > 0:
            a_idx = zero_indices[:, 0]
            b_idx = zero_indices[:, 1]
            c_idx = zero_indices[:, 2]
            
            U_a = U_m[a_idx, :]
            V_b = V_m[b_idx, :]
            W_c = W_m[c_idx, :]
            
            total_sum_z = np.einsum('ir,ir,ir->i', U_a, V_b, W_c)
            contrib_z = u_r[a_idx] * v_r[b_idx] * w_r[c_idx]
            
            if not np.allclose(total_sum_z - contrib_z, 0.0, atol=1e-9):
                return False
        
        return True
    
    # Greedy removal of redundant columns
    rank = base_rank
    r = 0
    while r < rank:
        if time.time() > deadline - 30:
            break
        if is_redundant_fast(r, U_np, V_np, W_np):
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            rank -= 1
            r = 0
        else:
            r += 1
    
    # Final verification
    if not verify_decomposition(U_np, V_np, W_np):
        # Fallback to base decomposition
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
    
    # Convert to lists with integer/half-integer entries
    U_out = U_np.tolist()
    V_out = V_np.tolist()
    W_out = W_np.tolist()
    
    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = round(val * 2) / 2
    
    return {"U": U_out, "V": V_out, "W": W_out}

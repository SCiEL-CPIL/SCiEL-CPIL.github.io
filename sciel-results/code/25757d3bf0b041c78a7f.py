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
    
    def verify_decomposition(U_m, V_m, W_m, check_all_zeros=True):
        """Verify that the decomposition exactly matches the target tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U_m[i * m + j] * V_m[j * p + k], W_m[k * n + i])
                    if not np.isclose(lhs, 1.0, atol=1e-9):
                        return False
        # Check zero positions
        if check_all_zeros:
            for idx in range(len(zero_indices)):
                a, b, c = zero_indices[idx]
                lhs = np.dot(U_m[a] * V_m[b], W_m[c])
                if not np.isclose(lhs, 0.0, atol=1e-9):
                    return False
        else:
            # Check a subset
            step = max(1, len(zero_indices) // 200)
            for idx in range(0, len(zero_indices), step):
                a, b, c = zero_indices[idx]
                lhs = np.dot(U_m[a] * V_m[b], W_m[c])
                if not np.isclose(lhs, 0.0, atol=1e-9):
                    return False
        return True
    
    def is_redundant_fast(r, U_m, V_m, W_m, check_all_zeros=True):
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
        
        if check_all_zeros:
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
        else:
            # Check a subset of zeros
            step = max(1, len(zero_indices) // 200)
            a_idx = zero_indices[::step, 0]
            b_idx = zero_indices[::step, 1]
            c_idx = zero_indices[::step, 2]
            
            U_a = U_m[a_idx, :]
            V_b = V_m[b_idx, :]
            W_c = W_m[c_idx, :]
            
            total_sum_z = np.einsum('ir,ir,ir->i', U_a, V_b, W_c)
            contrib_z = u_r[a_idx] * v_r[b_idx] * w_r[c_idx]
            
            if not np.allclose(total_sum_z - contrib_z, 0.0, atol=1e-9):
                return False
        
        return True
    
    # Greedy removal of redundant columns - try multiple passes with different orders
    rank = base_rank
    U_work = U_np.copy()
    V_work = V_np.copy()
    W_work = W_np.copy()
    
    best_U = U_work.copy()
    best_V = V_work.copy()
    best_W = W_work.copy()
    best_rank = rank
    
    # Try removal in original order
    r = 0
    while r < rank:
        if time.time() > deadline - 60:
            break
        if is_redundant_fast(r, U_work, V_work, W_work, check_all_zeros=False):
            U_work = np.delete(U_work, r, axis=1)
            V_work = np.delete(V_work, r, axis=1)
            W_work = np.delete(W_work, r, axis=1)
            rank -= 1
            r = 0
        else:
            r += 1
    
    if rank < best_rank:
        best_U = U_work.copy()
        best_V = V_work.copy()
        best_W = W_work.copy()
        best_rank = rank
    
    # Try removal in reverse order
    rank = base_rank
    U_work = U_np.copy()
    V_work = V_np.copy()
    W_work = W_np.copy()
    
    r = rank - 1
    while r >= 0:
        if time.time() > deadline - 45:
            break
        if is_redundant_fast(r, U_work, V_work, W_work, check_all_zeros=False):
            U_work = np.delete(U_work, r, axis=1)
            V_work = np.delete(V_work, r, axis=1)
            W_work = np.delete(W_work, r, axis=1)
            rank -= 1
        else:
            r -= 1
    
    if rank < best_rank:
        best_U = U_work.copy()
        best_V = V_work.copy()
        best_W = W_work.copy()
        best_rank = rank
    
    # Try removal in random order
    rng = np.random.RandomState(42)
    rank = base_rank
    U_work = U_np.copy()
    V_work = V_np.copy()
    W_work = W_np.copy()
    
    order = list(range(rank))
    rng.shuffle(order)
    for r in order:
        if time.time() > deadline - 30:
            break
        # Adjust index since columns may have been removed
        # Find the actual column index
        if r < rank:
            if is_redundant_fast(r, U_work, V_work, W_work, check_all_zeros=False):
                U_work = np.delete(U_work, r, axis=1)
                V_work = np.delete(V_work, r, axis=1)
                W_work = np.delete(W_work, r, axis=1)
                rank -= 1
                # Adjust remaining indices
                order = [x - 1 if x > r else x for x in order if x != r]
    
    if rank < best_rank:
        best_U = U_work.copy()
        best_V = V_work.copy()
        best_W = W_work.copy()
        best_rank = rank
    
    # Final verification with all zeros
    if not verify_decomposition(best_U, best_V, best_W, check_all_zeros=True):
        # Fallback to base decomposition
        best_U = U_np
        best_V = V_np
        best_W = W_np
        best_rank = base_rank
    
    # Convert to lists with integer/half-integer entries
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = round(val * 2) / 2
    
    return {"U": U_out, "V": V_out, "W": W_out}

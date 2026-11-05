"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Use Strassen-based block decomposition (rank 33) as baseline, then search for lower rank.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 1], [0, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 0]],
            [[1, 0], [0, 1]],
            [[-1, 0], [0, 1]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        for b in range(3):
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                col += 1
            
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                col += 1
        
        return U, V, W

    # Start with the known rank-33 decomposition
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    if not verify_decomposition(best_U, best_V, best_W):
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    # Try to find a lower-rank decomposition using alternating least squares
    # The idea: fix two matrices, solve for the third via linear system
    def try_rank(r, max_iters=50, seed=42):
        """Try to find a rank-r decomposition using ALS."""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        # Initialize with random matrices
        U = rng.randint(-2, 3, size=(dim_a, r))
        V = rng.randint(-2, 3, size=(dim_b, r))
        W = rng.randint(-2, 3, size=(dim_c, r))
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update W: for each column c, solve (U[:,c] ⊗ V[:,c])^T W = T_flat
            # Actually, the constraint is: sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
            # This is linear in W if U,V are fixed.
            # Flatten: for each (a,b,c_idx), sum_t U[a,t]*V[b,t]*W[c_idx,t] = T[a,b,c_idx]
            # For fixed U,V, this is a linear system in W.
            
            # Build the linear system: A_W * W_flat = T_flat
            # A_W is (dim_a*dim_b*dim_c) x (dim_c*r)
            # This is too large. Instead, use the fact that for each output index c_idx,
            # sum_t W[c_idx,t] * (U[:,t] ⊗ V[:,t]) = T[:,:,c_idx]
            
            # For each c_idx, we have: sum_t W[c_idx,t] * outer(U[:,t], V[:,t]) = T[:,:,c_idx]
            # This is a linear system: M_c * W[c_idx,:] = T[:,:,c_idx].flatten()
            # where M_c[t, (a,b)] = U[a,t] * V[b,t]
            
            # M is (r, dim_a*dim_b) for each c_idx (same M for all c_idx)
            M = np.zeros((r, dim_a * dim_b), dtype=np.float64)
            for t in range(r):
                for a in range(dim_a):
                    for b in range(dim_b):
                        M[t, a * dim_b + b] = U[a, t] * V[b, t]
            
            # Now for each c_idx, solve: M * W[c_idx,:] = T[:,:,c_idx].flatten()
            # M is r x (dim_a*dim_b), W[c_idx,:] is r x 1, T[:,:,c_idx].flatten() is (dim_a*dim_b) x 1
            # This is an underdetermined system if r < dim_a*dim_b, which it is.
            # Use least squares.
            
            T_flat = T.reshape(dim_a * dim_b, dim_c)  # (dim_a*dim_b, dim_c)
            
            # W_new[c_idx, t] such that sum_t M[t, :] * W_new[c_idx, t] = T_flat[:, c_idx]
            # i.e., M^T * W_new[:, c_idx] = T_flat[:, c_idx]
            # Wait, let me re-derive.
            # sum_t U[a,t] * V[b,t] * W[c_idx,t] = T[a,b,c_idx]
            # Let M[t, (a,b)] = U[a,t] * V[b,t]
            # Then sum_t M[t, (a,b)] * W[c_idx,t] = T[a,b,c_idx]
            # In matrix form: M^T @ W[c_idx,:]^T = T[:,:,c_idx].flatten()
            # So W[c_idx,:]^T = (M^T)^+ @ T[:,:,c_idx].flatten()  (pseudoinverse)
            # Or equivalently, solve M^T X = T_flat[:, c_idx] for X = W[c_idx,:]^T
            
            M_T = M.T  # (dim_a*dim_b, r)
            W_new = np.zeros((dim_c, r), dtype=np.float64)
            for c_idx in range(dim_c):
                # Solve M_T * x = T_flat[:, c_idx]
                # M_T is (dim_a*dim_b, r), x is (r, 1), T_flat[:, c_idx] is (dim_a*dim_b, 1)
                try:
                    x, residuals, rank_s, sv = np.linalg.lstsq(M_T, T_flat[:, c_idx], rcond=None)
                    W_new[c_idx, :] = x
                except:
                    return None
            
            W = W_new
            
            # Similarly update U and V
            # For U: sum_t U[a,t] * (V[:,t] ⊗ W[:,t]) = T[a,:,:]
            # Let N[t, (b,c_idx)] = V[b,t] * W[c_idx,t]
            # Then sum_t U[a,t] * N[t, (b,c_idx)] = T[a,b,c_idx]
            # N^T * U[a,:]^T = T[a,:,:].flatten()
            
            N = np.zeros((r, dim_b * dim_c), dtype=np.float64)
            for t in range(r):
                for b in range(dim_b):
                    for c_idx in range(dim_c):
                        N[t, b * dim_c + c_idx] = V[b, t] * W[c_idx, t]
            
            N_T = N.T  # (dim_b*dim_c, r)
            T_flat2 = T.reshape(dim_a, dim_b * dim_c)
            U_new = np.zeros((dim_a, r), dtype=np.float64)
            for a in range(dim_a):
                try:
                    x, residuals, rank_s, sv = np.linalg.lstsq(N_T, T_flat2[a, :], rcond=None)
                    U_new[a, :] = x
                except:
                    return None
            U = U_new
            
            # For V: sum_t V[b,t] * (U[:,t] ⊗ W[:,t]) = T[:,b,:]
            # Let P[t, (a,c_idx)] = U[a,t] * W[c_idx,t]
            # Then sum_t V[b,t] * P[t, (a,c_idx)] = T[a,b,c_idx]
            # P^T * V[b,:]^T = T[:,b,:].flatten()
            
            P = np.zeros((r, dim_a * dim_c), dtype=np.float64)
            for t in range(r):
                for a in range(dim_a):
                    for c_idx in range(dim_c):
                        P[t, a * dim_c + c_idx] = U[a, t] * W[c_idx, t]
            
            P_T = P.T  # (dim_a*dim_c, r)
            T_flat3 = T.reshape(dim_b, dim_a * dim_c)
            V_new = np.zeros((dim_b, r), dtype=np.float64)
            for b in range(dim_b):
                try:
                    x, residuals, rank_s, sv = np.linalg.lstsq(P_T, T_flat3[b, :], rcond=None)
                    V_new[b, :] = x
                except:
                    return None
            V = V_new
            
            # Check convergence
            if it % 5 == 0:
                # Round to nearest integers/half-integers and check
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify_decomposition(U_r.astype(np.int64) * 2, V_r.astype(np.int64) * 2, W_r.astype(np.int64) * 2):
                    # Scale back
                    U_f = U_r / 2
                    V_f = V_r / 2
                    W_f = W_r / 2
                    return U_f, V_f, W_f
        
        return None

    # Try ranks from 32 down to 25
    for r in range(32, 24, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, max_iters=30, seed=42 + r)
        if result is not None:
            U_try, V_try, W_try = result
            if verify_decomposition(U_try, V_try, W_try):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = r
                break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END

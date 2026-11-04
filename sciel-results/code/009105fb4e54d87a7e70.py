import numpy as np
import time
import random

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Build the target tensor T
    # T is a tensor of shape (n*m, m*p, p*n)
    # Indices: (i*m+j, j*p+k, k*n+i)
    # T[idx1, idx2, idx3] = 1 if the indices correspond to a valid multiplication term
    # Specifically, T[i*m+j, j*p+k, k*n+i] = 1 for all i, j, k
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
                
    # Function to check if a set of factors U, V, W exactly decomposes T
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    def check_decomposition(U, V, W):
        # Compute the tensor sum
        # T_reconstructed[i, j, k] = sum_t U[i, t] * V[j, t] * W[k, t]
        # This is a trilinear form.
        # We can compute this efficiently using einsum or loops.
        # For small dimensions, loops are fine.
        R = U.shape[1]
        T_rec = np.zeros((n*m, m*p, p*n), dtype=float)
        for t in range(R):
            u_t = U[:, t]
            v_t = V[:, t]
            w_t = W[:, t]
            # Outer product of u_t, v_t, w_t
            T_rec += np.einsum('i,j,k->ijk', u_t, v_t, w_t)
        
        # Check if T_rec matches T exactly (within floating point tolerance)
        # Since we want integer/half-integer, we check for exact match after rounding
        # But first, let's check if the float error is small
        err = np.linalg.norm(T_rec - T)
        if err > 1e-5:
            return False, T_rec, err
        
        # Try to round to nearest integer or half-integer
        # The problem states "integer/half-integer entries"
        # This usually means entries are in {0, 1, -1, 0.5, -0.5, ...}
        # Let's assume integers first, then half-integers.
        
        # Check integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        T_rec_int = np.zeros((n*m, m*p, p*n), dtype=float)
        for t in range(R):
            T_rec_int += np.einsum('i,j,k->ijk', U_int[:, t], V_int[:, t], W_int[:, t])
        if np.allclose(T_rec_int, T, atol=1e-9):
            return True, T_rec_int, 0.0
            
        # Check half-integers (multiply by 2, round, divide by 2)
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        T_rec_half = np.zeros((n*m, m*p, p*n), dtype=float)
        for t in range(R):
            T_rec_half += np.einsum('i,j,k->ijk', U_half[:, t], V_half[:, t], W_half[:, t])
        if np.allclose(T_rec_half, T, atol=1e-9):
            return True, T_rec_half, 0.0
            
        return False, T_rec, err

    # Function to find a decomposition of rank R
    def find_decomposition(R, max_iters=5000):
        # Initialize U, V, W with random integers in {-1, 0, 1}
        # Or small integers
        U = np.random.randint(-1, 2, (n*m, R))
        V = np.random.randint(-1, 2, (m*p, R))
        W = np.random.randint(-1, 2, (p*n, R))
        
        best_err = np.inf
        best_U, best_V, best_W = U, V, W
        
        # Local search: coordinate descent
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random factor and a random column
            factor_idx = random.randint(0, 2)
            col_idx = random.randint(0, R - 1)
            
            if factor_idx == 0:
                # Update U[:, col_idx]
                # We want to minimize || T - sum_t u_t v_t w_t ||^2
                # This is a linear least squares problem for u_t
                # T = u_t (v_t w_t)^T + rest
                # We can compute the residual
                T_rec = np.zeros((n*m, m*p, p*n), dtype=float)
                for t in range(R):
                    T_rec += np.einsum('i,j,k->ijk', U[:, t], V[:, t], W[:, t])
                residual = T - T_rec + np.einsum('i,j,k->ijk', U[:, col_idx], V[:, col_idx], W[:, col_idx])
                
                # We want to find u_t such that u_t (v_t w_t)^T approximates residual
                # Let X = (v_t w_t) reshaped to (m*p*p*n, 1)? No.
                # The term is sum_i u_i * (v_t w_t)_{j,k}
                # This is a linear system: A u = b
                # Where A_{(j,k), i} = (v_t w_t)_{j,k}
                # b_{j,k} = residual_{i,j,k} ? No, residual is a tensor.
                # We want to minimize sum_{i,j,k} (residual_{i,j,k} - sum_i' u_{i'} A_{(j,k), i'})^2 ?
                # Actually, the term is sum_i u_i * M_{i,j,k} where M_{i,j,k} = V_{j,t} W_{k,t}
                # So for each (j,k), we have a vector u and a vector M_{:,j,k}.
                # We want u dot M_{:,j,k} to approximate residual_{i,j,k}?
                # No, the sum is over i.
                # C_{i,k} = sum_j A_{i,j} B_{j,k}.
                # The tensor equation is T_{i,j,k} = sum_t U_{i,t} V_{j,t} W_{k,t}.
                # So for fixed t, we have U_{:,t} V_{:,t} W_{:,t}.
                # This is a rank-1 tensor.
                # We want to update U_{:,t} to minimize the error.
                # Error = || T - sum_{t'} U_{:,t'} V_{:,t'} W_{:,t'} ||_F
                # This is a non-convex problem.
                # A simple heuristic: update U_{i,t} by a small random step and check if error decreases.
                
                # Let's just do a random perturbation
                step = random.uniform(-0.5, 0.5)
                U_orig = U[:, col_idx].copy()
                U[:, col_idx] += step
                
                T_rec_new = np.zeros((n*m, m*p, p*n), dtype=float)
                for t in range(R):
                    T_rec_new += np.einsum('i,j,k->ijk', U[:, t], V[:, t], W[:, t])
                err_new = np.linalg.norm(T_rec_new - T)
                
                if err_new < best_err:
                    best_err = err_new
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err < 1e-5:
                        return best_U, best_V, best_W
                else:
                    U[:, col_idx] = U_orig
                    
            elif factor_idx == 1:
                V_orig = V[:, col_idx].copy()
                step = random.uniform(-0.5, 0.5)
                V[:, col_idx] += step
                
                T_rec_new = np.zeros((n*m, m*p, p*n), dtype=float)
                for t in range(R):
                    T_rec_new += np.einsum('i,j,k->ijk', U[:, t], V[:, t], W[:, t])
                err_new = np.linalg.norm(T_rec_new - T)
                
                if err_new < best_err:
                    best_err = err_new
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err < 1e-5:
                        return best_U, best_V, best_W
                else:
                    V[:, col_idx] = V_orig
                    
            else:
                W_orig = W[:, col_idx].copy()
                step = random.uniform(-0.5, 0.5)
                W[:, col_idx] += step
                
                T_rec_new = np.zeros((n*m, m*p, p*n), dtype=float)
                for t in range(R):
                    T_rec_new += np.einsum('i,j,k->ijk', U[:, t], V[:, t], W[:, t])
                err_new = np.linalg.norm(T_rec_new - T)
                
                if err_new < best_err:
                    best_err = err_new
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err < 1e-5:
                        return best_U, best_V, best_W
                else:
                    W[:, col_idx] = W_orig
                    
        return best_U, best_V, best_W

    # Try to find a rank 56 decomposition
    # If found, return it. Otherwise, return schoolbook rank 64.
    
    # First, check if rank 56 is possible with a few random restarts
    for _ in range(10):
        if time.time() > deadline:
            break
        U, V, W = find_decomposition(56, max_iters=1000)
        valid, _, _ = check_decomposition(U, V, W)
        if valid:
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # If not found, return schoolbook
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

# EVOLVE-BLOCK-END

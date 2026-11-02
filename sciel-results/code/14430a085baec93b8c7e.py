import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Define the target tensor T
    # T[i, j, k] = 1 for all i, j, k
    # The condition is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # We try to find lower rank decompositions
    # We start from rank 71 down to 50
    # Using alternating minimization
    
    # Precompute indices for faster access
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
    
    num_eqs = n * m * p
    
    for target_rank in range(71, 49, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Alternating minimization
            converged = False
            for iteration in range(20):
                if time.time() > deadline - 10:
                    break
                
                # Solve for W
                # For each (i, j, k), we have sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is linear in W.
                # We can set up a linear system A_w W = b_w
                # But it's overdetermined and coupled.
                # Instead, we can use a least squares approach.
                
                # Construct A_w
                # Rows correspond to (i, j, k)
                # Columns correspond to (k, i, r)
                # A_w[(i,j,k), (k,i,r)] = U[i*m+j, r] * V[j*p+k, r]
                
                A_w = np.zeros((num_eqs, p * n * target_rank))
                b_w = np.ones(num_eqs)
                
                idx = 0
                for i, j, k in indices:
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    for r in range(target_rank):
                        A_w[idx, c_idx * target_rank + r] = U[a_idx, r] * V[b_idx, r]
                    idx += 1
                
                W_flat, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                W = W_flat.reshape(p * n, target_rank)
                
                # Solve for U
                A_u = np.zeros((num_eqs, n * m * target_rank))
                b_u = np.ones(num_eqs)
                idx = 0
                for i, j, k in indices:
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    for r in range(target_rank):
                        A_u[idx, a_idx * target_rank + r] = V[b_idx, r] * W[c_idx, r]
                    idx += 1
                
                U_flat, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_flat.reshape(n * m, target_rank)
                
                # Solve for V
                A_v = np.zeros((num_eqs, m * p * target_rank))
                b_v = np.ones(num_eqs)
                idx = 0
                for i, j, k in indices:
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    for r in range(target_rank):
                        A_v[idx, b_idx * target_rank + r] = U[a_idx, r] * W[c_idx, r]
                    idx += 1
                
                V_flat, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_flat.reshape(m * p, target_rank)
                
                # Check error
                error = 0
                for i, j, k in indices:
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                    error += (val - 1) ** 2
                
                if error < 1e-10:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    error_r = 0
                    for i, j, k in indices:
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        val = np.dot(U_r[a_idx, :], V_r[b_idx, :] * W_r[c_idx, :])
                        error_r += (val - 1) ** 2
                    
                    if error_r < 1e-10:
                        best_rank = target_rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        converged = True
                        break
                    else:
                        # Keep unrounded if it's close, but we need exact
                        # The problem requires integer/half-integer entries
                        # If rounding fails, we might need to continue or accept unrounded if allowed?
                        # "integer/half-integer entries"
                        # So we must round.
                        pass
            
            if converged:
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

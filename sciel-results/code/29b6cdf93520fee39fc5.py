"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 by 3x8 matrix multiplication.
    
    Uses iterative hard thresholding (IHT) on a block-sparse tensor representation
    to minimize the number of non-zero rank-1 terms. The approach treats the factor
    matrices as a single block-sparse tensor and applies IHT to sparsify the solution.
    
    Strategy:
    1. Start from the schoolbook decomposition (rank 72)
    2. Reformulate as a sparse tensor completion problem
    3. Apply IHT to iteratively threshold small entries in the factor matrices
    4. Verify exactness after each iteration
    5. Return the lowest-rank valid decomposition found
    """
    n, m, p = DIMENSIONS
    
    import numpy as np
    
    # Build the target tensor T[a, b, c] where:
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # T[a, b, c] = 1 iff j==j' and k==k' and i==i'
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def validate_decomposition(U, V, W):
        """Check if U*V*W decomposition is valid."""
        rank = U.shape[1]
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            T_recon += np.einsum('i,j,k->ijk', u, v, w)
        return np.allclose(T_recon, T, atol=1e-10)
    
    def get_nonzero_rank(U, V, W):
        """Count the number of non-zero rank-1 terms."""
        # A rank-1 term is non-zero if any of its three vectors has a non-zero entry
        rank = U.shape[1]
        nonzero_count = 0
        for r in range(rank):
            if np.any(U[:, r] != 0) or np.any(V[:, r] != 0) or np.any(W[:, r] != 0):
                nonzero_count += 1
        return nonzero_count
    
    # Start from schoolbook decomposition
    rank_init = n * m * p
    U = np.zeros((n*m, rank_init))
    V = np.zeros((m*p, rank_init))
    W = np.zeros((p*n, rank_init))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Apply IHT to sparsify the decomposition
    # We'll try to reduce the effective rank by thresholding small entries
    
    best_rank = rank_init
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Try different threshold values and iteration counts
    for threshold in [0.01, 0.05, 0.1, 0.2, 0.5]:
        for max_iter in [50, 100, 200]:
            U_curr = U.copy()
            V_curr = V.copy()
            W_curr = W.copy()
            
            for iteration in range(max_iter):
                # Compute residual
                T_recon = np.zeros((n*m, m*p, p*n))
                for r in range(U_curr.shape[1]):
                    u = U_curr[:, r]
                    v = V_curr[:, r]
                    w = W_curr[:, r]
                    if np.all(u == 0) and np.all(v == 0) and np.all(w == 0):
                        continue
                    T_recon += np.einsum('i,j,k->ijk', u, v, w)
                
                residual = T - T_recon
                
                # If residual is small, check if we can reduce rank
                if np.max(np.abs(residual)) < 1e-8:
                    # Try to sparsify
                    U_sp = U_curr.copy()
                    V_sp = V_curr.copy()
                    W_sp = W_curr.copy()
                    
                    # Threshold small entries
                    U_sp[np.abs(U_sp) < threshold] = 0
                    V_sp[np.abs(V_sp) < threshold] = 0
                    W_sp[np.abs(W_sp) < threshold] = 0
                    
                    # Check if still valid
                    if validate_decomposition(U_sp, V_sp, W_sp):
                        curr_rank = get_nonzero_rank(U_sp, V_sp, W_sp)
                        if curr_rank < best_rank:
                            best_rank = curr_rank
                            best_U = U_sp.copy()
                            best_V = V_sp.copy()
                            best_W = W_sp.copy()
                    break
                
                # Gradient descent step with small learning rate
                lr = 0.01
                for r in range(U_curr.shape[1]):
                    u = U_curr[:, r]
                    v = V_curr[:, r]
                    w = W_curr[:, r]
                    if np.all(u == 0) and np.all(v == 0) and np.all(w == 0):
                        continue
                    
                    # Gradient for U[:, r]
                    grad_U = np.zeros(n*m)
                    for b in range(m*p):
                        for c in range(p*n):
                            grad_U += residual[:, b, c] * v[b] * w[c]
                    U_curr[:, r] -= lr * grad_U
                    
                    # Gradient for V[:, r]
                    grad_V = np.zeros(m*p)
                    for a in range(n*m):
                        for c in range(p*n):
                            grad_V += residual[a, :, c] * u[a] * w[c]
                    V_curr[:, r] -= lr * grad_V
                    
                    # Gradient for W[:, r]
                    grad_W = np.zeros(p*n)
                    for a in range(n*m):
                        for b in range(m*p):
                            grad_W += residual[a, b, :] * u[a] * v[b]
                    W_curr[:, r] -= lr * grad_W
                
                # Hard threshold: keep only top-k nonzeros per column
                # This helps enforce sparsity
                for r in range(U_curr.shape[1]):
                    # Keep top 3 nonzeros in U[:, r]
                    u = U_curr[:, r]
                    if np.count_nonzero(u) > 3:
                        idx = np.argsort(np.abs(u))[::-1][:3]
                        u_new = np.zeros_like(u)
                        u_new[idx] = u[idx]
                        U_curr[:, r] = u_new
                    
                    v = V_curr[:, r]
                    if np.count_nonzero(v) > 3:
                        idx = np.argsort(np.abs(v))[::-1][:3]
                        v_new = np.zeros_like(v)
                        v_new[idx] = v[idx]
                        V_curr[:, r] = v_new
                    
                    w = W_curr[:, r]
                    if np.count_nonzero(w) > 3:
                        idx = np.argsort(np.abs(w))[::-1][:3]
                        w_new = np.zeros_like(w)
                        w_new[idx] = w[idx]
                        W_curr[:, r] = w_new
            
            # After IHT iterations, check if we found a valid lower-rank decomposition
            if validate_decomposition(U_curr, V_curr, W_curr):
                curr_rank = get_nonzero_rank(U_curr, V_curr, W_curr)
                if curr_rank < best_rank:
                    best_rank = curr_rank
                    best_U = U_curr.copy()
                    best_V = V_curr.copy()
                    best_W = W_curr.copy()
    
    # If no improvement found, return schoolbook
    if best_rank == rank_init:
        U = np.zeros((n*m, rank_init))
        V = np.zeros((m*p, rank_init))
        W = np.zeros((p*n, rank_init))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Trim to actual rank
    # Find columns that are non-zero
    nonzero_cols = []
    for r in range(best_U.shape[1]):
        if np.any(best_U[:, r] != 0) or np.any(best_V[:, r] != 0) or np.any(best_W[:, r] != 0):
            nonzero_cols.append(r)
    
    U_final = best_U[:, nonzero_cols]
    V_final = best_V[:, nonzero_cols]
    W_final = best_W[:, nonzero_cols]
    
    # Round to clean values if close to integers or half-integers
    U_final = np.round(U_final, 8)
    V_final = np.round(V_final, 8)
    W_final = np.round(W_final, 8)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END

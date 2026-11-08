"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    tensor = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Get lower bounds from matrix flattening
    T0 = tensor.reshape(n*m, -1)
    rank0 = np.linalg.matrix_rank(T0)
    T1 = tensor.transpose(1,0,2).reshape(m*p, -1)
    rank1 = np.linalg.matrix_rank(T1)
    T2 = tensor.transpose(2,0,1).reshape(p*n, -1)
    rank2 = np.linalg.matrix_rank(T2)
    lower_bound = max(rank0, rank1, rank2)
    
    # Start with the trivial rank-42 solution
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to find lower rank decompositions using ALS
    for trial_rank in range(max(lower_bound, 1), rank):
        if time.time() > deadline - 40:
            break
        
        # Multiple random initializations
        for init in range(5):
            if time.time() > deadline - 40:
                break
            
            U_curr = np.random.randn(n*m, trial_rank)
            V_curr = np.random.randn(m*p, trial_rank)
            W_curr = np.random.randn(p*n, trial_rank)
            
            # Alternating least squares
            for iteration in range(100):
                if time.time() > deadline - 40:
                    break
                
                # Update U
                # For each a: minimize ||tensor[a] - U[a] * (V*W)||_F
                # (V*W)[b,c] = sum_r V[b,r]*W[c,r]
                # Linear system: (V*W)^T * U[a]^T = tensor[a].flatten()
                # This is (m*p*p*n, trial_rank) system - too large
                # Instead: sum_b (V*W)[b,c] * U[a,r] * V[b,r'] ... 
                
                # Better approach: 
                # Minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # = sum_{b,c} T[a,b,c]^2 - 2 sum_r U[a,r] sum_{b,c} T[a,b,c] V[b,r] W[c,r] + sum_{r,r'} U[a,r] U[a,r'] (sum_b V[b,r] V[b,r']) (sum_c W[c,r] W[c,r'])
                
                # Gradient: 2 sum_{b,c} (sum_r U[a,r] V[b,r] W[c,r] - T[a,b,c]) V[b,r'] W[c,r'] = 0
                # => sum_{r,r'} U[a,r] (sum_b V[b,r] V[b,r']) (sum_c W[c,r'] W[c,r'']) ... this is getting messy
                
                # Let me use a different formulation. Define:
                # M_U = V @ W.T  # shape (m*p, p*n)
                # Then T[a] should be approximated by U[a] * M_U (outer product in a sense)
                # Actually: sum_r U[a,r] * M_U[:, r] where M_U[:, r] = V[:,r] * W[:,r] (element-wise, shape m*p*p*n)
                
                # So we're approximating tensor[a].flatten() (length m*p*p*n) by U[a] @ (V*W).flatten()
                # where (V*W)[b,c,r] = V[b,r]*W[c,r]
                
                # This is: U[a] @ M where M is (m*p*p*n, trial_rank)
                # M[:, r] = (V[:,r] * W[:,r]).flatten()  -- but V[:,r] has length m*p, W[:,r] has length p*n
                # So M[:, r] = np.kron(V[:,r], W[:,r])  -- length m*p*p*n
                
                # This matrix is huge. Let me use a different approach.
                
                # Use the fact that the objective is quadratic in U[a]. The Hessian is:
                # H[r,r'] = sum_{b,c} V[b,r] V[b,r'] W[c,r] W[c,r'] = (V[:,r] . V[:,r']) * (W[:,r] . W[:,r'])
                # = (V.T @ V)[r,r'] * (W.T @ W)[r,r']
                
                # The gradient is:
                # g[r] = 2 * (sum_{b,c} T[a,b,c] V[b,r] W[c,r] - sum_{r'} U[a,r'] (V.T@V)[r,r'] (W.T@W)[r',r])
                
                # Wait, let me redo. Objective: f(U[a]) = ||tensor[a] - sum_r U[a,r] * (V[:,r] ⊗ W[:,r])||^2
                # where ⊗ is the Kronecker product (element-wise outer product flattened)
                
                # Let z_r = V[:,r] ⊗ W[:,r] (length m*p*p*n)
                # f = ||t - U[a] @ Z||^2 where Z is (m*p*p*n, trial_rank) with Z[:,r] = z_r
                # = t^T t - 2 U[a] @ Z^T @ t + U[a] @ Z^T @ Z @ U[a]
                
                # Gradient: -2 Z^T t + 2 Z^T Z U[a] = 0
                # => Z^T Z U[a] = Z^T t
                
                # Z^T Z is trial_rank x trial_rank:
                # (Z^T Z)[r,r'] = z_r . z_r' = sum_{b,c} V[b,r] W[c,r] V[b,r'] W[c,r']
                # = (sum_b V[b,r] V[b,r']) * (sum_c W[c,r] W[c,r'])
                # = (V.T @ V)[r,r'] * (W.T @ W)[r,r']
                
                # Z^T t is trial_rank:
                # (Z^T t)[r] = sum_{b,c} V[b,r] W[c,r] T[a,b,c]
                
                VT_V = V_curr.T @ V_curr  # (trial_rank, trial_rank)
                WT_W = W_curr.T @ W_curr  # (trial_rank, trial_rank)
                H_U = VT_V * WT_W  # element-wise product, (trial_rank, trial_rank)
                
                for a in range(n*m):
                    rhs = np.zeros(trial_rank)
                    for r in range(trial_rank):
                        rhs[r] = np.sum(V_curr[:, r] * W_curr[:, r] * tensor[a])
                    try:
                        U_curr[a] = np.linalg.solve(H_U, rhs)
                    except np.linalg.LinAlgError:
                        U_curr[a] = np.linalg.lstsq(H_U, rhs, rcond=None)[0]
                
                # Update V
                UT_U = U_curr.T @ U_curr
                WT_W = W_curr.T @ W_curr
                H_V = UT_U * WT_W
                
                for b in range(m*p):
                    rhs = np.zeros(trial_rank)
                    for r in range(trial_rank):
                        rhs[r] = np.sum(U_curr[:, r] * W_curr[:, r] * tensor[:, b, :].reshape(-1))
                    try:
                        V_curr[b] = np.linalg.solve(H_V, rhs)
                    except np.linalg.LinAlgError:
                        V_curr[b] = np.linalg.lstsq(H_V, rhs, rcond=None)[0]
                
                # Update W
                UT_U = U_curr.T @ U_curr
                VT_V = V_curr.T @ V_curr
                H_W = UT_U * VT_W if False else UT_U * VT_V  # fix
                
                for c in range(p*n):
                    rhs = np.zeros(trial_rank)
                    for r in range(trial_rank):
                        rhs[r] = np.sum(U_curr[:, r] * V_curr[:, r] * tensor[:, :, c].reshape(-1))
                    try:
                        W_curr[c] = np.linalg.solve(H_W, rhs)
                    except np.linalg.LinAlgError:
                        W_curr[c] = np.linalg.lstsq(H_W, rhs, rcond=None)[0]
                
                # Check convergence
                recon = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
                error = np.linalg.norm(recon - tensor)
                if error < 1e-8:
                    break
            
            # Check if decomposition is valid
            recon = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
            error = np.linalg.norm(recon - tensor)
            
            if error < 1e-6:
                # Try to round
                for factor in [1, 2]:
                    U_r = np.round(U_curr * factor) / factor
                    V_r = np.round(V_curr * factor) / factor
                    W_r = np.round(W_curr * factor) / factor
                    
                    recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    error_r = np.linalg.norm(recon_r - tensor)
                    
                    if error_r < 1e-6:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = trial_rank
                        break
                if best_rank == trial_rank:
                    break
        
        if best_rank == trial_rank:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Check if the decomposition exactly represents matrix multiplication."""
    # Tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition should satisfy: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    # Actually, the tensor for matmul is: T[i,j,k] = delta_{i,j,k} meaning the coefficient
    # Let me think about this more carefully.
    
    # The matrix multiplication tensor has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right either. Let me reconsider.
    
    # Actually for matrix multiplication C = A*B where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 (it's just the identity in the j-index sense)
    # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Let's verify by reconstructing the tensor
    # Create a 3D array of shape (n, m, p)
    T_reconstructed = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u_col = np.array(U)[:, c]  # shape (n*m,)
        v_col = np.array(V)[:, c]  # shape (m*p,)
        w_col = np.array(W)[:, c]  # shape (p*n,)
        
        # Reshape u_col to (n, m), v_col to (m, p), w_col to (p, n)
        u_2d = u_col.reshape(n, m)
        v_2d = v_col.reshape(m, p)
        w_2d = w_col.reshape(p, n)
        
        # The contribution to T[i,j,k] from rank c is u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    
    # Check if T_reconstructed is all ones
    if not np.allclose(T_reconstructed, 1.0, atol=1e-9):
        return False
    return True

def _verify_fast(U, V, W, n, m, p, rank):
    """Faster verification using vectorized operations."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # Vectorized: create index arrays
    # This is O(n*m*p*rank) which for n=2,m=5,p=6 is 60*rank operations
    
    # Alternative: use einsum-like approach
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Let's just do it directly but vectorized over c
    # For each c, we have outer products
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u = U_arr[:, c].reshape(n, m)
        v = V_arr[:, c].reshape(m, p)
        w = W_arr[:, c].reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # This is a tensor contraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
    
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    deadline = time.time() + 140  # Leave 10s margin
    
    best_rank = schoolbook_rank
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    
    # Initialize with schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Strategy: Start with a random initialization at lower rank and try to fit
    # The tensor we're decomposing is T[i,j,k] = 1 for all i,j,k
    
    # Try various ranks from low to high
    for rank in range(20, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Multiple random restarts for this rank
        max_restarts = 5 if rank < 40 else 3
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Scale to make entries reasonable
            scale = 1.0 / (rank ** 0.5)
            U *= scale
            V *= scale
            W *= scale
            
            # Gradient-free local search: try to minimize ||T_reconstructed - 1||^2
            # Using a simple coordinate descent or random perturbation approach
            
            best_loss = float('inf')
            best_U_local = U.copy()
            best_V_local = V.copy()
            best_W_local = W.copy()
            
            def compute_loss(U, V, W):
                T = np.zeros((n, m, p), dtype=float)
                for c in range(rank):
                    u = U[:, c].reshape(n, m)
                    v = V[:, c].reshape(m, p)
                    w = W[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                return np.sum((T - 1.0) ** 2)
            
            initial_loss = compute_loss(U, V, W)
            best_loss = initial_loss
            
            # Local search with random perturbations
            max_iter = 200
            step_size = 0.5
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Randomly perturb one of the matrices
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    U_new = U + np.random.randn(n*m, rank) * step_size
                elif mat_idx == 1:
                    V_new = V + np.random.randn(m*p, rank) * step_size
                else:
                    W_new = W + np.random.randn(p*n, rank) * step_size
                
                loss_new = compute_loss(U_new if mat_idx == 0 else U,
                                       V_new if mat_idx == 1 else V,
                                       W_new if mat_idx == 2 else W)
                
                if loss_new < best_loss:
                    best_loss = loss_new
                    best_U_local = (U_new if mat_idx == 0 else U).copy()
                    best_V_local = (V_new if mat_idx == 1 else V).copy()
                    best_W_local = (W_new if mat_idx == 2 else W).copy()
                    # Accept
                    if mat_idx == 0:
                        U = U_new
                    elif mat_idx == 1:
                        V = V_new
                    else:
                        W = W_new
                    # Reduce step size slightly
                    step_size *= 0.99
                else:
                    # Reject, increase step size occasionally
                    step_size *= 1.01
                    step_size = min(step_size, 2.0)
                
                if best_loss < 1e-12:
                    break
            
            if best_loss < 1e-9:
                # Found a valid decomposition at this rank
                best_rank = rank
                best_U = best_U_local.tolist()
                best_V = best_V_local.tolist()
                best_W = best_W_local.tolist()
                break
        
        if best_rank < schoolbook_rank:
            break
    
    # Final polish: if we found a solution, try gradient-based refinement for higher precision
    if best_rank < schoolbook_rank:
        U_polish = np.array(best_U, dtype=float)
        V_polish = np.array(best_V, dtype=float)
        W_polish = np.array(best_W, dtype=float)
        rank_p = best_rank
        
        lr = 0.01
        for polish_iter in range(100):
            if time.time() > deadline:
                break
            
            T = _compute_T(U_polish, V_polish, W_polish, n, m, p, rank_p)
            err = T - 1.0
            loss = np.sum(err ** 2)
            if loss < 1e-14:
                break
            
            # Compute gradients via finite differences on a few elements or use analytic
            # For speed, use a small number of gradient evaluations
            # dL/dU[i*m+j, c] = 2 * sum_{i,j,k} err[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
            # dL/dV[j*p+k, c] = 2 * sum_{i,j,k} err[i,j,k] * U[i*m+j,c] * W[k*n+i,c]
            # dL/dW[k*n+i, c] = 2 * sum_{i,j,k} err[i,j,k] * U[i*m+j,c] * V[j*p+k,c]
            
            # Build index arrays for efficient gradient computation
            # For each c, compute:
            # grad_U[c] (flattened n*m) where grad_U[i,j,c] = 2 * sum_k err[i,j,k]*V[j,k,c]*W[k,i,c]
            # grad_V[c] (flattened m*p) where grad_V[j,k,c] = 2 * sum_i err[i,j,k]*U[i,j,c]*W[k,i,c]
            # grad_W[c] (flattened p*n) where grad_W[k,i,c] = 2 * sum_j err[i,j,k]*U[i,j,c]*V[j,k,c]
            
            grad_U = np.zeros((n*m, rank_p))
            grad_V = np.zeros((m*p, rank_p))
            grad_W = np.zeros((p*n, rank_p))
            
            for c in range(rank_p):
                u = U_polish[:, c].reshape(n, m)
                v = V_polish[:, c].reshape(m, p)
                w = W_polish[:, c].reshape(p, n)
                # grad_U[i,j,c] = 2 * sum_k err[i,j,k] * v[j,k] * w[k,i]
                grad_U[:, c] = 2.0 * np.einsum('ijk,jk,ki->ij', err, v, w).reshape(n*m)
                # grad_V[j,k,c] = 2 * sum_i err[i,j,k] * u[i,j] * w[k,i]
                grad_V[:, c] = 2.0 * np.einsum('ijk,ij,ki->jk', err, u, w).reshape(m*p)
                # grad_W[k,i,c] = 2 * sum_j err[i,j,k] * u[i,j] * v[j,k]
                grad_W[:, c] = 2.0 * np.einsum('ijk,ij,jk->ki', err, u, v).reshape(p*n)
            
            U_polish -= lr * grad_U
            V_polish -= lr * grad_V
            W_polish -= lr * grad_W
            
            # Reduce learning rate
            lr *= 0.995
        
        # Final check
        T_final = _compute_T(U_polish, V_polish, W_polish, n, m, p, rank_p)
        if np.allclose(T_final, 1.0, atol=1e-9):
            best_U = U_polish.tolist()
            best_V = V_polish.tolist()
            best_W = W_polish.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

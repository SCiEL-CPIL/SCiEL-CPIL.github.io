"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Known lower bounds: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Schoolbook gives 63. We'll search for better ranks.
    # Try ranks from lower bound upward
    
    best_result = None
    best_rank = None
    
    # Start with a reasonable upper bound and work down
    # For 3x3 * 3x7, known best is around 21-25
    # Let's try ranks from 21 to 35
    
    for rank in range(21, 36):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations for this rank
        max_restarts = 50 if rank <= 25 else 20
        if time.time() > deadline - 10:
            max_restarts = 5
            
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            try:
                result = _try_rank(rank, n, m, p, restart, deadline)
                if result is not None:
                    found = True
                    if best_result is None or rank < best_rank:
                        best_rank = rank
                        best_result = result
                    break
            except Exception:
                continue
        
        if found and best_rank == rank:
            # Found a valid decomposition at this rank, try lower ranks
            continue
        elif not found:
            # If we can't find at this rank, higher ranks might be harder
            # But let's continue searching upward as a fallback
            pass
    
    # If nothing found, return schoolbook
    if best_result is None:
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
    
    return best_result

def _try_rank(rank, n, m, p, restart_idx, deadline):
    """Try to find a rank-rank decomposition using random init + gradient descent."""
    np.random.seed(42 + restart_idx * 1000)
    
    # Initialize U, V, W randomly
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Scale to reasonable magnitude
    scale = 1.0 / np.sqrt(rank)
    U *= scale
    V *= scale
    W *= scale
    
    # Use gradient descent to minimize ||tensor - sum_c U_c ⊗ V_c ⊗ W_c||
    # The multiplication tensor T[i,j,k] = delta(i,j,k) where the product entry
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if we're computing (A*B)[i,k] from A[i,j]*B[j,k]
    # In our factorization: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,k) for all i,k
    # Wait, let me reconsider the tensor structure.
    
    # The matrix multiplication tensor M has entries M[i,j,k] where:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So M[i,j,k] = 1 for all valid i,j,k (it's the standard multiplication tensor)
    
    # Our decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all (i,j,k)
    
    # Actually, let me re-examine. The tensor T is defined such that:
    # T[i,j,k] = 1 for all i in [0,n), j in [m), k in [p)
    # And we want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is a tensor completion/ranking problem.
    
    # Let's use a simple gradient-based approach
    lr = 0.01
    max_iter = 200
    
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
            
        # Compute current tensor estimate
        # T_est[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Efficient computation using einsum
        # First compute U^T * V for each c: gives (n*m, m*p)
        # Then contract with W
        
        # Let's compute the loss and gradients
        # T_est[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # For efficiency, compute in steps
        # Step 1: For each c, compute outer product structure
        # Actually, let's just compute the full tensor
        
        T_est = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            # T_est[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
            # This is a rank-1 tensor: u_c[i,j] * v_c[j,k] * w_c[k,i]
            # We can compute this as: for each i,j,k: u_c[i,j]*v_c[j,k]*w_c[k,i]
            
            # Efficient way: 
            # temp[i,j,k] = u_c[i,j] * v_c[j,k] * w_c[k,i]
            # = (u_c[i,j] * w_c[k,i]) * v_c[j,k]
            
            # Let's compute: A[i,k] = u_c[i,j] * w_c[k,i] but j is free...
            # Actually: temp[i,j,k] = u_c[i,j] * (v_c[j,k] * w_c[k,i])
            
            # v_c[j,k] * w_c[k,i] = sum over k? No, k is fixed.
            # = v_c[j,k] * w_c[k,i]
            # Let B[j,i] = sum_k v_c[j,k] * w_c[k,i] = (v_c @ w_c)[j,i]
            # Then temp[i,j,k] = u_c[i,j] * v_c[j,k] * w_c[k,i]
            # But B[j,i] = sum_k v_c[j,k]*w_c[k,i], so we can't separate.
            
            # Let's just do it directly but vectorized:
            # temp[i,j,k] = u_c[i,j] * v_c[j,k] * w_c[k,i]
            
            # Create 3D array:
            # u3d[i,j,k] = u_c[i,j] (broadcast over k)
            # v3d[i,j,k] = v_c[j,k] (broadcast over i)  
            # w3d[i,j,k] = w_c[k,i] (broadcast over j)
            
            u3d = u_c[:, :, np.newaxis]  # (n, m, 1)
            v3d = v_c[np.newaxis, :, :]  # (1, m, p)
            w3d = np.transpose(w_c, (1, 0))[np.newaxis, np.newaxis, :]  # (1, 1, p) but indexed as [k,i] -> w_c[k,i]
            # w3d[i,j,k] should be w_c[k,i]
            # w_c is (p, n), so w_c[k,i] where k in [0,p), i in [0,n)
            # We want a 3D array where index [i,j,k] = w_c[k,i]
            # So shape (n, m, p): for each i, j, k: value = w_c[k,i]
            # = np.tile(w_c.T, (m,1)).T reshaped? 
            # w_c.T is (n, p), w_c.T[i,k] = w_c[k,i]
            # So w3d[i,j,k] = w_c.T[i,k]
            # = np.broadcast_to(w_c.T[np.newaxis, :, :], (n, 1, p))
            
            w3d = np.broadcast_to(w_c.T[np.newaxis, :, :], (n, 1, p))
            
            T_est += u3d * v3d * w3d
        
        # Target tensor: all ones
        T_target = np.ones((n, m, p))
        
        # Loss
        diff = T_est - T_target
        loss = np.sum(diff ** 2)
        
        # Check convergence
        if loss < 1e-10:
            # Verify and return
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Compute gradients
        # dL/dU[i*m+j, c] = 2 * sum_{i,j,k} diff[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
        # dL/dV[j*p+k, c] = 2 * sum_{i,j,k} diff[i,j,k] * U[i*m+j, c] * W[k*n+i, c]
        # dL/dW[k*n+i, c] = 2 * sum_{i,j,k} diff[i,j,k] * U[i*m+j, c] * V[j*p+k, c]
        
        dU = np.zeros((n * m, rank))
        dV = np.zeros((m * p, rank))
        dW = np.zeros((p * n, rank))
        
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            # dU[i*m+j, c] = 2 * sum_k diff[i,j,k] * v_c[j,k] * w_c[k,i]
            # = 2 * sum_k diff[i,j,k] * v_c[j,k] * w_c[k,i]
            # Let C[i,j] = sum_k diff[i,j,k] * v_c[j,k] * w_c[k,i]
            # = sum_k diff[i,j,k] * v_c[j,k] * w_c[k,i]
            # Note: v_c[j,k] * w_c[k,i] = (v_c @ w_c)[j,i]
            # But diff[i,j,k] varies with k, so we can't just use matrix product.
            
            # C[i,j] = sum_k diff[i,j,k] * v_c[j,k] * w_c[k,i]
            # For fixed j: C[i,j] = v_c[j,:] @ (diff[:,j,:] * w_c.T)[:, i]? 
            # Let's think differently.
            
            # C[i,j] = sum_k diff[i,j,k] * v_c[j,k] * w_c[k,i]
            # = sum_k (diff[i,j,k] * w_c[k,i]) * v_c[j,k]
            # Let D[i,j,k] = diff[i,j,k] * w_c[k,i]
            # Then C[i,j] = sum_k D[i,j,k] * v_c[j,k]
            # = (D[:,:,i] @ v_c.T)[j]? No...
            
            # D[i,j,k] has shape (n, m, p)
            # C[i,j] = sum_k D[i,j,k] * v_c[j,k]
            # For fixed i: C[i,j] = sum_k D[i,j,k] * v_c[j,k]
            # This is like: for each i, compute sum_k D[i,j,k] * v_c[j,k]
            # = np.einsum('ijk,jk->ij', D, v_c)
            
            D = diff * w_c.T[np.newaxis, np.newaxis, :]  # (n, m, p), D[i,j,k] = diff[i,j,k]*w_c[k,i]
            C_u = np.einsum('ijk,jk->ij', D, v_c)  # (n, m)
            dU[c] = 2 * C_u.ravel()
            
            # dV[j*p+k, c] = 2 * sum_{i,j,k} diff[i,j,k] * u_c[i,j] * w_c[k,i]
            # For fixed k: dV[j,k] = 2 * sum_i diff[i,j,k] * u_c[i,j] * w_c[k,i]
            # Let E[i,j,k] = diff[i,j,k] * u_c[i,j]
            # dV[j,k] = 2 * sum_i E[i,j,k] * w_c[k,i]
            # = 2 * np.einsum('ijk,ki->jk', E, w_c)
            
            E = diff * u_c[:, :, np.newaxis]  # (n, m, p), E[i,j,k] = diff[i,j,k]*u_c[i,j]
            C_v = np.einsum('ijk,ki->jk', E, w_c)  # (m, p)
            dV[c] = 2 * C_v.ravel()
            
            # dW[k*n+i, c] = 2 * sum_{i,j,k} diff[i,j,k] * u_c[i,j] * v_c[j,k]
            # For fixed i: dW[k,i] = 2 * sum_j diff[i,j,k] * u_c[i,j] * v_c[j,k]
            # Let F[i,j,k] = diff[i,j,k] * v_c[j,k]
            # dW[k,i] = 2 * sum_j F[i,j,k] * u_c[i,j]
            # = 2 * np.einsum('ijk,ij->ki', F, u_c)
            
            F = diff * v_c[np.newaxis, :, :]  # (n, m, p), F[i,j,k] = diff[i,j,k]*v_c[j,k]
            C_w = np.einsum('ijk,ij->ki', F, u_c)  # (p, n)
            dW[c] = 2 * C_w.ravel()
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Final check
    T_est = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        u3d = u_c[:, :, np.newaxis]
        v3d = v_c[np.newaxis, :, :]
        w3d = np.broadcast_to(w_c.T[np.newaxis, :, :], (n, 1, p))
        T_est += u3d * v3d * w3d
    
    loss = np.sum((T_est - 1.0) ** 2)
    if loss < 1e-8:
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END

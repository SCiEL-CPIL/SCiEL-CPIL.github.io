import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 21
    
    # Target tensor T[i, j, k] = 1
    # We want U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r to equal 1
    
    # Initialize
    rank = target_rank
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Gauge fixing: Set U[0, r] = 1 for all r?
    # This might be too restrictive if U[0,r] is 0 in the true solution.
    # But we can permute indices.
    # Let's just run optimization.
    
    start_time = time.time()
    deadline = start_time + 100 # Leave 20s for verification and formatting
    
    # Simple gradient descent
    # Error = sum_{i,j,k} (sum_r U_imj_r V_jpk_r W_kni_r - 1)^2
    # Grad U_imj_r = 2 * (sum_{i',j',k'} T_err_{i',j',k'} V_j'pk'_r W_k'n i'_r) ... wait, indices are tricky.
    
    # Let's use a simpler approach: 
    # We know the solution exists with half-integers.
    # We can try to solve the linear system for a fixed set of "active" terms? No, it's non-linear.
    
    # Given the complexity and risk of failure, and the fact that I don't have the explicit coefficients memorized,
    # I will return the schoolbook decomposition (rank 63) to ensure a valid output, 
    # unless I can quickly find a known decomposition.
    
    # Actually, there is a known rank-21 decomposition for 3x3x7.
    # It is derived from the 7-way decomposition of 2x2x2? No.
    # Let's try to construct it.
    
    # If I cannot guarantee finding it, I should stick to a valid solution.
    # However, the score 1.0 is rank 63.
    # If I return rank 63, I get score 1.0.
    # If I find rank 21, I get score 3.0.
    
    # Let's try a randomized search with local optimization for a short time.
    
    best_error = np.inf
    best_U, best_V, best_W = None, None, None
    
    for _ in range(50): # 50 restarts
        if time.time() > deadline: break
        
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to prevent divergence
        for r in range(rank):
            norm_U = np.linalg.norm(U[:, r])
            norm_V = np.linalg.norm(V[:, r])
            norm_W = np.linalg.norm(W[:, r])
            if norm_U > 1e-9 and norm_V > 1e-9 and norm_W > 1e-9:
                scale = 1.0 / (norm_U * norm_V * norm_W) ** (1/3)
                U[:, r] *= scale
                V[:, r] *= scale
                W[:, r] *= scale
            else:
                # Reset this term
                U[:, r] = np.random.randn(n * m)
                V[:, r] = np.random.randn(m * p)
                W[:, r] = np.random.randn(p * n)
                
        # Gradient Descent
        lr = 0.01
        for step in range(2000):
            if time.time() > deadline: break
            
            # Compute current tensor
            # T_curr[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # This is expensive to compute fully every step.
            # We can compute gradients analytically.
            
            # Error
            T_curr = np.zeros((n, m, p))
            for r in range(rank):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                T_curr += u_r[:, :, None] * v_r[None, :, :] * w_r[None, None, :] # Broadcasting might be wrong shape
                # u_r is (n, m), v_r is (m, p), w_r is (p, n)
                # We need T[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
                # This is a tensor contraction.
                # np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
                
            T_curr = np.einsum('ijr,jkr,kir->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
            err = T_curr - 1.0
            loss = np.sum(err**2)
            
            if loss < best_error:
                best_error = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if loss < 1e-6:
                    break
            
            # Gradients
            # dL/dU_imj_r = 2 * err[i,j,k] * V[j,k,r] * W[k,i,r] summed over k? No, k is fixed in err index.
            # err is (n, m, p).
            # dL/dU[i,m,j,r] = 2 * sum_k err[i,j,k] V[j,k,r] W[k,i,r]
            # dL/dV[j,p,k,r] = 2 * sum_i err[i,j,k] U[i,m,j,r] W[k,i,r]
            # dL/dW[k,n,i,r] = 2 * sum_j err[i,j,k] U[i,m,j,r] V[j,k,r]
            
            # Using einsum for gradients
            # dU = 2 * np.einsum('ijk,jkr,kir->ijr', err, V.reshape(m,p,rank), W.reshape(p,n,rank))
            # dV = 2 * np.einsum('ijk,ijr,kir->jkr', err, U.reshape(n,m,rank), W.reshape(p,n,rank))
            # dW = 2 * np.einsum('ijk,ijr,jkr->kir', err, U.reshape(n,m,rank), V.reshape(m,p,rank))
            
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', err, V.reshape(m,p,rank), W.reshape(p,n,rank))
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', err, U.reshape(n,m,rank), W.reshape(p,n,rank))
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', err, U.reshape(n,m,rank), V.reshape(m,p,rank))
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Re-normalize to keep values in check
            for r in range(rank):
                norm_U = np.linalg.norm(U[:, r])
                norm_V = np.linalg.norm(V[:, r])
                norm_W = np.linalg.norm(W[:, r])
                if norm_U > 1e-9 and norm_V > 1e-9 and norm_W > 1e-9:
                    scale = 1.0 / (norm_U * norm_V * norm_W) ** (1/3)
                    U[:, r] *= scale
                    V[:, r] *= scale
                    W[:, r] *= scale

    if best_error < 1e-4:
        # Snap to half-integers
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        
        # Verify
        T_check = np.einsum('ijr,jkr,kir->ijk', U_snap.reshape(n,m,rank), V_snap.reshape(m,p,rank), W_snap.reshape(p,n,rank))
        if np.allclose(T_check, 1.0, atol=1e-9):
            return {
                "U": U_snap.tolist(),
                "V": V_snap.tolist(),
                "W": W_snap.tolist()
            }
            
    # Fallback to schoolbook
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

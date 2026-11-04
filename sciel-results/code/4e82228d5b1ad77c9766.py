import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    
    # Schoolbook decomposition
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try ranks from 44 down to 37 (or lower)
    # It is unlikely that rank 45 is not minimal for 3x3x5, but let's check 44, 43, etc.
    # We will attempt to find a decomposition of rank R.
    # If we fail, we keep the best found (which might be 45).
    
    # We will try a few specific ranks below 45.
    # If the true rank is 45, these will fail, and we return 45.
    # If the true rank is 44, we might find it.
    
    target_ranks = [44, 43, 42, 41, 40]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Try multiple restarts for this rank
        max_restarts = 3
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            try:
                result = search_rank_gradient(n, m, p, target_rank, deadline)
                if result is not None:
                    return result
            except Exception:
                pass

    return best_result

def search_rank_gradient(n, m, p, rank, deadline):
    """
    Uses Gradient Descent with Momentum to minimize the Frobenius norm of the tensor decomposition error.
    """
    # Initialization
    # We use random initialization with small values.
    # To help snapping, we might initialize with values close to 0 or 0.5 or 1.
    # Let's try a mix.
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Reshape for einsum
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    
    # Learning rate and momentum
    lr = 0.01
    momentum = 0.9
    
    U_vel = np.zeros_like(U)
    V_vel = np.zeros_like(V)
    W_vel = np.zeros_like(W)
    
    max_iter = 2000
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Compute T = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.sum((T - 1.0)**2)
        
        if error < 1e-10:
            # Try snapping
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            U3s = U_snap.reshape(n, m, rank)
            V3s = V_snap.reshape(m, p, rank)
            W3s = W_snap.reshape(p, n, rank)
            Ts = np.einsum('ijr,jkr,kir->ijk', U3s, V3s, W3s)
            snap_error = np.sum((Ts - 1.0)**2)
            
            if snap_error < 1e-8:
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        # Compute Gradients
        # dE/dU[i,j,r] = 2 * sum_k (T[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
        # dE/dV[j,k,r] = 2 * sum_i (T[i,j,k] - 1) * U[i,j,r] * W[k,i,r]
        # dE/dW[k,i,r] = 2 * sum_j (T[i,j,k] - 1) * U[i,j,r] * V[j,k,r]
        
        # Note: T[i,j,k] corresponds to the (i,j) entry of A*B.
        # The tensor is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        # The target is 1 for all i,j,k? 
        # Wait, the matrix multiplication tensor M[i,j,k] is 1 if the term A[i,j] B[j,k] contributes to C[i,k].
        # In the standard decomposition, the tensor T[i,j,k] should equal 1 for all i,j,k?
        # Let's check the seed code.
        # Seed code:
        # U_sb[i * m + j, col] = 1
        # V_sb[j * p + k, col] = 1
        # W_sb[k * n + i, col] = 1
        # T[i,j,k] = sum_col U[i*m+j, col] V[j*p+k, col] W[k*n+i, col]
        # For a specific (i,j,k), only one column (the one corresponding to i,j,k) is non-zero in all three?
        # No, U[i*m+j, col] is 1 only if col corresponds to (i,j,*).
        # V[j*p+k, col] is 1 only if col corresponds to (*,j,k).
        # W[k*n+i, col] is 1 only if col corresponds to (k,*,i).
        # So for a fixed (i,j,k), the column col = i*m*p + j*p + k has U=1, V=1, W=1.
        # All other columns have at least one 0.
        # So T[i,j,k] = 1 for all i,j,k.
        # Yes, the target tensor is all ones.
        
        # Gradients:
        # dE/dU[i,j,r] = 2 * sum_k (T[i,j,k] - 1) * V3[j,k,r] * W3[k,i,r]
        # dE/dV[j,k,r] = 2 * sum_i (T[i,j,k] - 1) * U3[i,j,r] * W3[k,i,r]
        # dE/dW[k,i,r] = 2 * sum_j (T[i,j,k] - 1) * U3[i,j,r] * V3[j,k,r]
        
        # We can compute these using einsum as well.
        # Let D[i,j,k] = 2 * (T[i,j,k] - 1)
        
        D = 2.0 * (T - 1.0)
        
        # dU[i,j,r] = sum_k D[i,j,k] * V3[j,k,r] * W3[k,i,r]
        # dU = np.einsum('ijk,jkr,kir->ijr', D, V3, W3)
        # dV[j,k,r] = sum_i D[i,j,k] * U3[i,j,r] * W3[k,i,r]
        # dV = np.einsum('ijk,ijr,kir->jkr', D, U3, W3)
        # dW[k,i,r] = sum_j D[i,j,k] * U3[i,j,r] * V3[j,k,r]
        # dW = np.einsum('ijk,ijr,jkr->kir', D, U3, V3)
        
        dU = np.einsum('ijk,jkr,kir->ijr', D, V3, W3).reshape(n*m, rank)
        dV = np.einsum('ijk,ijr,kir->jkr', D, U3, W3).reshape(m*p, rank)
        dW = np.einsum('ijk,ijr,jkr->kir', D, U3, V3).reshape(p*n, rank)
        
        # Update with momentum
        U_vel = momentum * U_vel - lr * dU
        V_vel = momentum * V_vel - lr * dV
        W_vel = momentum * W_vel - lr * dW
        
        U += U_vel
        V += V_vel
        W += W_vel
        
        # Optional: Project or snap periodically? No, let it converge.
        
    # Final check
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    error = np.sum((T - 1.0)**2)
    
    if error < 1e-8:
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        U3s = U_snap.reshape(n, m, rank)
        V3s = V_snap.reshape(m, p, rank)
        W3s = W_snap.reshape(p, n, rank)
        Ts = np.einsum('ijr,jkr,kir->ijk', U3s, V3s, W3s)
        snap_error = np.sum((Ts - 1.0)**2)
        
        if snap_error < 1e-8:
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
    return None

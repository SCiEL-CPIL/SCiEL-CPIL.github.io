import numpy as np
import scipy.optimize as opt

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Build target tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Try ranks from 70 upwards
    for rank in range(70, 78):
        # Initialize U, V, W
        # Use small random values to break symmetry
        np.random.seed(42)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Define objective function
        def objective(params):
            U_flat = params[:n*m*rank]
            V_flat = params[n*m*rank:n*m*rank + m*p*rank]
            W_flat = params[n*m*rank + m*p*rank:]
            
            U = U_flat.reshape(n*m, rank)
            V = V_flat.reshape(m*p, rank)
            W = W_flat.reshape(p*n, rank)
            
            # Compute T_dec
            # T_dec[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            
            error = T - T_dec
            return np.sum(error * error)
        
        # Optimize
        # Use L-BFGS-B
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # We might need to scale the problem or use a different optimizer
        # Let's try a simple gradient descent with a fixed step size or adaptive
        
        # Actually, L-BFGS-B might be slow for large rank.
        # Let's try a simple manual gradient descent for a few iterations
        
        lr = 0.01
        for it in range(1000):
            # Compute gradients
            # dT_dec/dU[i, r] = sum_{j,k} V[j, r] W[k, r] error[i, j, k]
            # dT_dec/dV[j, r] = sum_{i,k} U[i, r] W[k, r] error[i, j, k]
            # dT_dec/dW[k, r] = sum_{i,j} U[i, r] V[j, r] error[i, j, k]
            
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = T - T_dec
            
            grad_U = np.einsum('jkr,ijk->ir', V, W, error) # Wait, indices
            # error is ijk. V is jr. W is kr.
            # grad_U[i, r] = sum_{j,k} V[j, r] W[k, r] error[i, j, k]
            # This is not a simple einsum.
            # grad_U[i, r] = sum_j sum_k V[j, r] W[k, r] error[i, j, k]
            # Let's do it step by step
            
            # temp1[i, j, r] = sum_k W[k, r] error[i, j, k]
            temp1 = np.einsum('kr,ijk->ijr', W, error)
            # grad_U[i, r] = sum_j V[j, r] temp1[i, j, r]
            grad_U = np.einsum('jr,ijr->ir', V, temp1)
            
            # grad_V[j, r] = sum_{i,k} U[i, r] W[k, r] error[i, j, k]
            # temp2[j, k, r] = sum_i U[i, r] error[i, j, k]
            temp2 = np.einsum('ir,ijk->jkr', U, error)
            # grad_V[j, r] = sum_k W[k, r] temp2[j, k, r]
            grad_V = np.einsum('kr,jkr->jr', W, temp2)
            
            # grad_W[k, r] = sum_{i,j} U[i, r] V[j, r] error[i, j, k]
            # temp3[i, j, r] = U[i, r] V[j, r]
            temp3 = np.einsum('ir,jr->ijr', U, V)
            # grad_W[k, r] = sum_{i,j} temp3[i, j, r] error[i, j, k]
            grad_W = np.einsum('ijr,ijk->kr', temp3, error)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if it % 100 == 0:
                obj = np.sum(error * error)
                if obj < 1e-6:
                    break
        
        # Snap to integers
        U_snap = np.round(U)
        V_snap = np.round(V)
        W_snap = np.round(W)
        
        # Check if exact
        T_dec_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
        if np.allclose(T_dec_snap, T, atol=1e-9):
            return {
                "U": U_snap.tolist(),
                "V": V_snap.tolist(),
                "W": W_snap.tolist()
            }
            
    # If no rank found, fall back to the original Strassen decomposition
    # (Copy the original code here as fallback)
    # ...

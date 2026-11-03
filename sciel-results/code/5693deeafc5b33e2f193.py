import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if u, v, w correspond to a valid multiplication term
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Function to verify a decomposition
    def verify(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Compute sum_r U[:,r] V[:,r] W[:,r]
        # This is a tensor of shape (nm, mp, pn)
        # We check if it matches T
        # For efficiency, we can check the flattened tensor
        # But R is small, so we can just loop or use einsum
        # np.einsum('ir,jr,kr->ijk', U, V, W)
        try:
            decomposed = np.einsum('ir,jr,kr->ijk', U, V, W)
            return np.allclose(decomposed, T, atol=1e-9)
        except:
            return False

    # Function to round to half-integers
    def round_half_int(x):
        return np.round(x * 2) / 2

    # Search for low rank decomposition
    # We try ranks from 30 up to 35
    # If we find one, we return it
    # Otherwise we return schoolbook 36
    
    deadline = time.time() + 110  # Leave 10s margin
    
    for R in range(30, 36):
        if time.time() > deadline:
            break
            
        # Initialize randomly
        np.random.seed(42 + R)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Optimize
        # We minimize the sum of squares of the difference
        # def loss(params):
        #     U, V, W = reshape params
        #     T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
        #     return np.sum((T_dec - T)**2)
        
        # Use L-BFGS-B
        from scipy.optimize import minimize
        
        def loss_and_grad(params):
            U_flat = params[:n*m*R].reshape(n*m, R)
            V_flat = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W_flat = params[n*m*R + m*p*R:].reshape(p*n, R)
            
            T_dec = np.einsum('ir,jr,kr->ijk', U_flat, V_flat, W_flat)
            diff = T_dec - T
            loss = np.sum(diff**2)
            
            # Gradient
            # dL/dU = sum_k diff * V * W
            # grad_U[i, r] = sum_{j,k} diff[i,j,k] V[j,r] W[k,r]
            grad_U = np.einsum('ijk,jr,kr->ir', diff, V_flat, W_flat)
            grad_V = np.einsum('ijk,ir,kr->jr', diff, U_flat, W_flat)
            grad_W = np.einsum('ijk,ir,jr->kr', diff, U_flat, V_flat)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return loss, grad

        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Run optimizer
        res = minimize(loss_and_grad, params0, method='L-BFGS-B', jac=True, options={'maxiter': 50, 'maxfun': 50})
        
        if res.success or res.fun < 1e-6:
            U_opt = res.x[:n*m*R].reshape(n*m, R)
            V_opt = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W_opt = res.x[n*m*R + m*p*R:].reshape(p*n, R)
            
            # Round to half-integers
            U_rounded = round_half_int(U_opt)
            V_rounded = round_half_int(V_opt)
            W_rounded = round_half_int(W_opt)
            
            if verify(U_rounded, V_rounded, W_rounded):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
    
    # Fallback: Schoolbook rank 36
    R = 36
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

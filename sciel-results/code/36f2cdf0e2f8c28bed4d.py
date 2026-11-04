import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    
    # Build target tensor
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try rank 17 with explicit Strassen-like construction
    # For 2x3 * 3x3, we can use a rank-17 decomposition
    # The key insight: group the 3x3 multiplication into blocks and use Strassen-like tricks
    
    target_rank = 17
    U = np.zeros((dim_u, target_rank))
    V = np.zeros((dim_v, target_rank))
    W = np.zeros((dim_w, target_rank))
    
    # Standard decomposition has 18 terms: for each (i,j,k), U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # We need to merge two terms. Let's try merging terms that share structure.
    # 
    # Terms are indexed by (i,j,k) where i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # Let's try to merge (i=0,j=0,k=0) and (i=1,j=0,k=0) using a Strassen-like trick
    #
    # Actually, for 2x3 * 3x3, a known rank-17 decomposition exists.
    # Let me construct it explicitly.
    
    # The 18 standard terms:
    # (0,0,0), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,1,2), (0,2,0), (0,2,1), (0,2,2),
    # (1,0,0), (1,0,1), (1,0,2), (1,1,0), (1,1,1), (1,1,2), (1,2,0), (1,2,1), (1,2,2)
    
    # Let's try a different approach: use the fact that for 2x3*3x3,
    # we can write C = A*B where A is 2x3 and B is 3x3.
    # 
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # Let's try to find a rank-17 decomposition by solving a system of equations.
    # We'll use the approach of starting from rank 18 and trying to eliminate one term.
    
    # Alternative: use a known construction. For 2x3*3x3, the rank is at most 17.
    # Let me try to construct it.
    
    # Let's use a different strategy: try all pairs of terms to merge and see if we can
    # find a valid rank-17 decomposition.
    
    # For each pair of standard terms (i1,j1,k1) and (i2,j2,k2), try to replace them
    # with two new terms that capture both contributions.
    
    # The standard term (i,j,k) contributes:
    # U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
    
    # To merge two terms, we need to find U', V', W' such that:
    # U'[i1*m+j1] + U'[i2*m+j2] captures both U contributions
    # etc.
    
    # This is getting complex. Let me try a more direct approach:
    # Use scipy to solve for a rank-17 decomposition.
    
    start_time = time.time()
    deadline = 140.0
    
    # Build the standard rank-18 decomposition
    U_std = np.zeros((dim_u, 18))
    V_std = np.zeros((dim_v, 18))
    W_std = np.zeros((dim_w, 18))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_std[i * m + j, col] = 1
                V_std[j * p + k, col] = 1
                W_std[k * n + i, col] = 1
                col += 1
    
    # Try to find a rank-17 decomposition by optimization
    from scipy.optimize import minimize
    
    best_err = float('inf')
    best_params = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        np.random.seed(restart * 13 + 7)
        
        # Initialize: take standard 18, drop one column, add noise
        U_init = np.zeros((dim_u, target_rank))
        V_init = np.zeros((dim_v, target_rank))
        W_init = np.zeros((dim_w, target_rank))
        
        # Drop column 'restart % 18' and redistribute
        drop_col = restart % 18
        for t in range(target_rank):
            src = t if t < drop_col else t + 1
            U_init[:, t] = U_std[:, src]
            V_init[:, t] = V_std[:, src]
            W_init[:, t] = W_std[:, src]
        
        # Add perturbation
        U_init += np.random.randn(dim_u, target_rank) * 0.05
        V_init += np.random.randn(dim_v, target_rank) * 0.05
        W_init += np.random.randn(dim_w, target_rank) * 0.05
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(params):
            U = params[0:dim_u*target_rank].reshape((dim_u, target_rank))
            V = params[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
            W = params[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
            T_rec = np.einsum('at,bt,ct->abc', U, V, W)
            E = T - T_rec
            return np.sum(E**2)
        
        def gradient(params):
            U = params[0:dim_u*target_rank].reshape((dim_u, target_rank))
            V = params[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
            W = params[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
            T_rec = np.einsum('at,bt,ct->abc', U, V, W)
            E = T - T_rec
            
            grad_U = -2 * np.einsum('abc,bt,ct->at', E, V, W)
            grad_V = -2 * np.einsum('abc,at,ct->bt', E, U, W)
            grad_W = -2 * np.einsum('abc,at,bt->ct', E, U, V)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        try:
            res = minimize(objective, params_init, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 200, 'gtol': 1e-12, 'ftol': 1e-16, 'disp': False})
            
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x.copy()
            
            if best_err < 1e-10:
                U_sol = best_params[0:dim_u*target_rank].reshape((dim_u, target_rank))
                V_sol = best_params[dim_u*target_rank:dim_u*target_rank + dim_v*target_rank].reshape((dim_v, target_rank))
                W_sol = best_params[dim_u*target_rank + dim_v*target_rank:].reshape((dim_w, target_rank))
                
                # Round to half-integers
                U_rounded = np.round(2.0 * U_sol) / 2.0
                V_rounded = np.round(2.0 * V_sol) / 2.0
                W_rounded = np.round(2.0 * W_sol) / 2.0
                
                T_check = np.einsum('at,bt,ct->abc', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_check, T, atol=1e-9):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
        except Exception:
            pass
    
    # Return standard rank 18
    return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}

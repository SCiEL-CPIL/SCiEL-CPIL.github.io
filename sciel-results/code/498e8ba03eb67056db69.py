"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank_17(T, deadline):
    n, m, p = 2, 3, 3
    dims = (n*m, m*p, p*n)
    rank = 17
    
    # Initialize
    best_loss = float('inf')
    best_params = None
    
    # Try a few random initializations
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(42 + restart)
        # Scale initialization to avoid huge gradients
        U = np.random.randn(*dims[0:1] + (rank,)) * 0.5
        V = np.random.randn(*dims[1:2] + (rank,)) * 0.5
        W = np.random.randn(*dims[2:3] + (rank,)) * 0.5
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            U_flat, V_flat, W_flat = np.split(params, [U.size, U.size + V.size])
            U_r = U_flat.reshape(dims[0], rank)
            V_r = V_flat.reshape(dims[1], rank)
            W_r = W_flat.reshape(dims[2], rank)
            
            # Compute T_est
            # T_est[a,b,c] = sum_r U_r[a,r] V_r[b,r] W_r[c,r]
            # Efficiently:
            # UV = U @ V.T -> (a, b, r) ? No.
            # Let's use einsum or loops for clarity/speed in numpy
            # T_est = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            
            T_est = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            loss = np.sum((T - T_est)**2)
            return loss
            
        def gradient(params):
            U_flat, V_flat, W_flat = np.split(params, [U.size, U.size + V.size])
            U_r = U_flat.reshape(dims[0], rank)
            V_r = V_flat.reshape(dims[1], rank)
            W_r = W_flat.reshape(dims[2], rank)
            
            T_est = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            err = T - T_est
            
            # dJ/dU[a,r] = -2 * sum_{b,c} err[a,b,c] V[b,r] W[c,r]
            # = -2 * (err * V.T * W.T) ?
            # err is (a,b,c). V is (b,r). W is (c,r).
            # grad_U = -2 * np.einsum('abc,br,cr->ar', err, V_r, W_r)
            # grad_V = -2 * np.einsum('abc,ar,cr->br', err, U_r, W_r)
            # grad_W = -2 * np.einsum('abc,ar,br->cr', err, U_r, V_r)
            
            grad_U = -2 * np.einsum('abc,br,cr->ar', err, V_r, W_r)
            grad_V = -2 * np.einsum('abc,ar,cr->br', err, U_r, W_r)
            grad_W = -2 * np.einsum('abc,ar,br->cr', err, U_r, V_r)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
        from scipy.optimize import minimize
        res = minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-10})
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
            
            if best_loss < 1e-10:
                # Check if half-integers
                U_flat, V_flat, W_flat = np.split(best_params, [U.size, U.size + V.size])
                U_r = U_flat.reshape(dims[0], rank)
                V_r = V_flat.reshape(dims[1], rank)
                W_r = W_flat.reshape(dims[2], rank)
                
                # Round to nearest 0.5
                U_r_half = np.round(U_r * 2) / 2
                V_r_half = np.round(V_r * 2) / 2
                W_r_half = np.round(W_r * 2) / 2
                
                # Verify exactness
                T_est = np.einsum('ar,br,cr->abc', U_r_half, V_r_half, W_r_half)
                if np.allclose(T, T_est, atol=1e-9):
                    return {"U": U_r_half.tolist(), "V": V_r_half.tolist(), "W": W_r_half.tolist()}
                    
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140
    
    # Build T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1.0
                
    # Try rank 17
    res = _try_rank_17(T, deadline)
    if res is not None:
        return res
        
    # Fallback
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END

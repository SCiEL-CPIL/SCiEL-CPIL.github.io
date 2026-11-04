import scipy
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    n, m, p = 3, 3, 6
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Construct target tensor T_target
    # T_target[a, b, c] = 1 if a,b,c form a valid multiplication term (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def snap_and_verify(U, V, W):
        # Snap to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        # Check exactness
        R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        # Check if R matches T_target exactly
        if np.allclose(R, T_target, atol=1e-9):
            # Convert to lists
            U_list = U_s.tolist()
            V_list = V_s.tolist()
            W_list = W_s.tolist()
            return True, U_list, V_list, W_list
        return False, None, None, None

    def get_schoolbook(rank):
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Pre-compute indices for faster objective function
    # We want to compute R = U @ V^T @ W^T (conceptually)
    # R[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    
    # We can vectorize the objective function calculation
    # Let's create a function that computes the residual efficiently
    
    def compute_residual(params, rank):
        U_p = params[:dim_U*rank].reshape(dim_U, rank)
        V_p = params[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
        W_p = params[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
        
        # Efficiently compute R
        # R[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # This is a tensor product
        # We can use einsum
        R = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
        
        return np.sum((R - T_target)**2)

    # We will iterate ranks from 46 to 53.
    # If we find a valid decomposition, return it.
    # If not, return schoolbook (54).
    
    # Strategy:
    # 1. Use multiple random initializations
    # 2. Use L-BFGS-B for optimization
    # 3. Try to snap to half-integers
    # 4. Verify exactness
    
    # To improve success rate, we can:
    # - Use better initialization (e.g., based on schoolbook structure)
    # - Use more iterations
    # - Try different optimization methods
    # - Use a two-stage approach: coarse optimization then fine-tuning
    
    # Let's try a structured initialization based on Strassen-like patterns
    # For 3x3x3, Strassen uses 7 multiplications
    # For 3x3x6, we can view it as two 3x3x3 multiplications sharing the middle dimension
    
    # Actually, let's just improve the random search with better parameters
    
    for rank in range(46, 54):
        if time.time() > deadline - 30: break
        
        # Try multiple random initializations
        # More attempts for lower ranks as they are harder.
        num_attempts = 15 if rank < 48 else 10
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 15: break
            
            # Initialize with different strategies
            if attempt % 3 == 0:
                # Standard random
                np.random.seed(42 + rank * 100 + attempt)
                U = np.random.randn(dim_U, rank) * 0.5
                V = np.random.randn(dim_V, rank) * 0.5
                W = np.random.randn(dim_W, rank) * 0.5
            elif attempt % 3 == 1:
                # Sparse random (more zeros)
                np.random.seed(43 + rank * 100 + attempt)
                U = np.random.choice([0, 0.5, 1, -0.5, -1], size=(dim_U, rank))
                V = np.random.choice([0, 0.5, 1, -0.5, -1], size=(dim_V, rank))
                W = np.random.choice([0, 0.5, 1, -0.5, -1], size=(dim_W, rank))
            else:
                # Based on schoolbook structure with perturbations
                np.random.seed(44 + rank * 100 + attempt)
                U_base, V_base, W_base = get_schoolbook(rank)
                # Add small random perturbations
                U = U_base + np.random.randn(dim_U, rank) * 0.1
                V = V_base + np.random.randn(dim_V, rank) * 0.1
                W = W_base + np.random.randn(dim_W, rank) * 0.1
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def obj(params, rank=rank):
                return compute_residual(params, rank)
            
            # L-BFGS-B
            try:
                # Use tighter tolerance and more iterations
                res = scipy.optimize.minimize(
                    obj, params, method='L-BFGS-B', 
                    options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                U_opt = res.x[:dim_U*rank].reshape(dim_U, rank)
                V_opt = res.x[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                W_opt = res.x[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return {"U": U_s, "V": V_s, "W": W_s}
                    
                # If snap fails, try a local polish with tighter constraints
                # Use Nelder-Mead for final polish
                if res.fun < 1e-6:
                    try:
                        res2 = scipy.optimize.minimize(
                            obj, res.x, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                        )
                        U_opt2 = res2.x[:dim_U*rank].reshape(dim_U, rank)
                        V_opt2 = res2.x[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                        W_opt2 = res2.x[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                        
                        success2, U_s2, V_s2, W_s2 = snap_and_verify(U_opt2, V_opt2, W_opt2)
                        if success2:
                            return {"U": U_s2, "V": V_s2, "W": W_s2}
                    except Exception:
                        pass
                        
            except Exception:
                pass

    # Fallback to schoolbook
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

import scipy
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T
    # T[i, j, k] = 1 if valid, else 0
    # Actually, the tensor is defined on the grid.
    # We represent the decomposition as U (nm x r), V (mp x r), W (pn x r)
    # T_flat[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We want to find r such that there exist U, V, W with entries in Z/2
    # minimizing the error.
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # We try to find a better rank.
    # Known: M_2,2,2 has rank 7.
    # M_2,3,3 has rank 14?
    # M_2,3,5:
    # Let's try ranks from 20 to 30.
    
    best_result = None
    best_rank = rank_schoolbook
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    # Fill schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
                
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Helper to compute error
    def compute_error(U, V, W, r):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # Tensor product
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # We can compute this efficiently.
        # Loop over c
        # T = sum_c U[:,c,None] * V[None,:,c] * W[None,None,c] ? No.
        # U is (nm, r), V is (mp, r), W is (pn, r).
        # Index i*m+j, j*p+k, k*n+i.
        # Let's just do it with loops for simplicity, or vectorize.
        
        # Vectorized:
        # Reshape U to (n, m, r)
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # We want sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
        # This is a contraction.
        # Sum over c and j.
        # For each i, k:
        # T[i,k] = sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        
        # Let's compute the full tensor T_calc of size (n, m, p)
        # T_calc[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        
        # Use einsum
        # U: (n, m, r)
        # V: (m, p, r)
        # W: (p, n, r)
        # We want T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Note: W is indexed by (k,i).
        
        T_calc = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Target T is all ones for valid i,j,k
        T_target = np.ones((n, m, p))
        
        err = np.sum((T_calc - T_target) ** 2)
        return err

    # Optimization function
    def optimize(rank, max_iters=500, lr=0.1, restarts=5):
        best_err = float('inf')
        best_params = None
        
        for restart in range(restarts):
            # Initialize
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Simple SGD
            for it in range(max_iters):
                err = compute_error(U, V, W, rank)
                if err < 1e-6:
                    return U, V, W, err
                
                # Gradient calculation is expensive to do symbolically.
                # We can use numerical gradient or just a simple heuristic.
                # Given time constraints, let's try a few random perturbations?
                # Or use a library? No, only stdlib/numpy/scipy.
                # Let's use scipy.optimize.minimize if available?
                # "Python standard library, NumPy and SciPy are available."
                
                # We can flatten U,V,W and use scipy.optimize.
                # But 500 iters * 5 restarts * 30 ranks * scipy overhead might be too slow.
                # Let's stick to a simple manual gradient step or random search.
                
                # Actually, for tensor rank, the landscape is very rugged.
                # A better strategy: Use the known structure.
                # For M_2,3,5, we can try to combine schoolbook terms.
                
                break # Skip inner loop for now, just return random
            
        return None

    # Since random search is unlikely to find the exact integer solution quickly,
    # we should look for specific constructions.
    
    # One known construction for M_2,3,3 is rank 14.
    # For M_2,3,5, we can pad M_2,3,3 with M_2,0,2? No.
    
    # Let's try a simple local search from the schoolbook solution.
    # Start with rank 30.
    # Try to set some entries to 0 or 0.5 to reduce effective rank?
    # No, rank is the number of columns.
    
    # Alternative: Try fixed ranks like 28, 27, 26 with random initialization and a few steps of gradient descent.
    
    def run_optimization(rank):
        # Flatten U, V, W
        size = (n*m + m*p + p*n) * rank
        x0 = np.random.randn(size) * 0.5
        
        def func(x):
            U = x[:n*m*rank].reshape(n*m, rank)
            V = x[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
            W = x[(n*m+m*p)*rank:].reshape(p*n, rank)
            return compute_error(U, V, W, rank)
        
        # Use Nelder-Mead or BFGS?
        # BFGS might be faster if gradient is cheap.
        # But gradient of einsum is easy.
        
        def grad(x):
            U = x[:n*m*rank].reshape(n*m, rank)
            V = x[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
            W = x[(n*m+m*p)*rank:].reshape(p*n, rank)
            
            U_r = U.reshape(n, m, rank)
            V_r = V.reshape(m, p, rank)
            W_r = W.reshape(p, n, rank)
            
            # T_calc[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # dT/dU[i,j,c] = V[j,k,c] W[k,i,c]
            # dT/dV[j,k,c] = U[i,j,c] W[k,i,c]
            # dT/dW[k,i,c] = U[i,j,c] V[j,k,c]
            
            # Error = sum (T_calc - 1)^2
            # dE/dU[i,j,c] = 2 * sum_{i,k} (T_calc[i,j,k] - 1) * V[j,k,c] W[k,i,c]
            
            # We can compute this with einsum.
            # Let D[i,j,k] = T_calc[i,j,k] - 1
            T_calc = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            D = T_calc - 1
            
            # grad_U[i,j,c] = 2 * sum_k D[i,j,k] * V[j,k,c] * W[k,i,c]
            grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', D, V_r, W_r)
            # grad_V[j,k,c] = 2 * sum_i D[i,j,k] * U[i,j,c] * W[k,i,c]
            grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', D, U_r, W_r)
            # grad_W[k,i,c] = 2 * sum_j D[i,j,k] * U[i,j,c] * V[j,k,c]
            grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', D, U_r, V_r)
            
            gU = grad_U.reshape(-1)
            gV = grad_V.reshape(-1)
            gW = grad_W.reshape(-1)
            return np.concatenate([gU, gV, gW])
        
        # Run BFGS
        try:
            res = scipy.optimize.minimize(func, x0, method='BFGS', jac=grad, options={'maxiter': 50, 'disp': False})
            if res.success and res.fun < 1e-6:
                U = res.x[:n*m*rank].reshape(n*m, rank)
                V = res.x[n*m*rank:(n*m+m*p)*rank].reshape(m*p, rank)
                W = res.x[(n*m+m*p)*rank:].reshape(p*n, rank)
                return U, V, W
        except:
            pass
        return None

    # Try ranks
    for r in range(20, 30):
        if time.time() > deadline:
            break
        for attempt in range(3):
            if time.time() > deadline:
                break
            res = run_optimization(r)
            if res is not None:
                U, V, W = res
                # Round to half integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                err = compute_error(U_rounded, V_rounded, W_rounded, r)
                if err < 1e-9:
                    # Found a solution!
                    # Convert to lists
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
    
    # Fallback to schoolbook
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }

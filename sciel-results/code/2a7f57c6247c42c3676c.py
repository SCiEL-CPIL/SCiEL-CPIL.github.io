import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 150s limit, 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    rank_sb = n * m * p   # 42
    
    # 1. Build the schoolbook decomposition (Ground Truth / Fallback)
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_residual = 0.0
    
    # 2. Define the target tensor T
    # T[i, j, k] = 1 if it corresponds to a valid multiplication term
    # Indices: 
    #   U index: i*m + j
    #   V index: j*p + k
    #   W index: k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # 3. Optimization Strategy
    # We want to find U, V, W of rank R such that sum_r U[:,r] v_r w_r^T approx T.
    # We flatten U, V, W into a single vector x.
    # Objective: || T - sum_r u_r v_r^T w_r ||_F^2
    
    def objective_and_grad(x, R):
        # Unpack x into U, V, W
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = x[0:size_u*R].reshape(size_u, R)
        V = x[size_u*R:size_u*R + size_v*R].reshape(size_v, R)
        W = x[size_u*R + size_v*R:].reshape(size_w, R)
        
        # Compute residual
        # Res = T - sum_r U[:,r] outer V[:,r] outer W[:,r]
        # Efficient computation:
        # Let M_{uv} be the matrix of coefficients for U and V? No, simpler to loop over R for small R.
        # R is at most 42.
        
        Res = T.copy()
        for r in range(R):
            # u_r is (n*m), v_r is (m*p), w_r is (p*n)
            # Tensor term: u_r[:,None,None] * v_r[None,:,None] * w_r[None,None,:]
            Res -= np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
        residual = np.sum(Res**2)
        
        # Gradient computation (Analytical for speed)
        # d/dU[i,r] (sum (T - sum U V W)^2)
        # = -2 * sum_{j,k} Res[i,j,k] V[j,r] W[k,r]
        # d/dV[j,r] = -2 * sum_{i,k} Res[i,j,k] U[i,r] W[k,r]
        # d/dW[k,r] = -2 * sum_{i,j} Res[i,j,k] U[i,r] V[j,r]
        
        grad_U = -2.0 * np.einsum('ijk,jr,kr->ir', Res, V, W)
        grad_V = -2.0 * np.einsum('ijk,ir,kr->jr', Res, U, W)
        grad_W = -2.0 * np.einsum('ijk,ir,jr->kr', Res, U, V)
        
        grad = np.concatenate([
            grad_U.flatten(), 
            grad_V.flatten(), 
            grad_W.flatten()
        ])
        
        return residual, grad

    # Try ranks from 41 down to 35 (Lower bound is max(6, 21, 14)=21, but likely close to 42)
    # We prioritize finding *any* rank < 42.
    
    np.random.seed(42)
    
    for R in range(41, 34, -1):
        if time.time() > deadline - 20:
            break
            
        # Multiple random starts for robustness
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
                
            # Initialize with random values, scaled
            # Scale matters for convergence. Schoolbook has entries 0/1.
            # Let's start with small random noise around 0, or structured init.
            # Structured init: take schoolbook, remove R columns, keep rest? 
            # No, that's rank 42. 
            # Let's try random init with magnitude ~1.
            
            U_init = np.random.randn(n*m, R) * 0.5
            V_init = np.random.randn(m*p, R) * 0.5
            W_init = np.random.randn(p*n, R) * 0.5
            
            x0 = np.concatenate([
                U_init.flatten(), 
                V_init.flatten(), 
                W_init.flatten()
            ])
            
            # Use L-BFGS-B
            try:
                res = minimize(
                    objective_and_grad, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=True, # Return (f, grad)
                    options={
                        'maxiter': 200, 
                        'ftol': 1e-12, 
                        'gtol': 1e-10,
                        'maxls': 20
                    }
                )
                
                if res.success or res.fun < 1e-8:
                    # Check if the solution is actually valid (residual is tiny)
                    if res.fun < 1e-6:
                        # Extract solution
                        size_u = n * m
                        size_v = m * p
                        size_w = p * n
                        
                        U_sol = res.x[0:size_u*R].reshape(size_u, R)
                        V_sol = res.x[size_u*R:size_u*R + size_v*R].reshape(size_v, R)
                        W_sol = res.x[size_u*R + size_v*R:].reshape(size_w, R)
                        
                        # Verify
                        Res = T.copy()
                        for r in range(R):
                            Res -= np.einsum('i,j,k->ijk', U_sol[:, r], V_sol[:, r], W_sol[:, r])
                        final_res = np.sum(Res**2)
                        
                        if final_res < 1e-5:
                            # Valid decomposition found!
                            best_rank = R
                            best_U = U_sol
                            best_V = V_sol
                            best_W = W_sol
                            best_residual = final_res
                            print(f"Found rank {R} decomposition with residual {final_res}")
                            # We can break out of attempts, but keep trying lower ranks?
                            # The task asks for rank-minimal. So we should try lower.
                            # But if we found 41, we should try 40, 39...
                            # The loop above does this.
                            break # Break attempts for this R, move to next (lower) R
                            
            except Exception as e:
                print(f"Optimization failed for R={R}, attempt={attempt}: {e}")
                continue
            
    # 4. Return the best result found
    # If we found a better rank, return it. Otherwise return schoolbook.
    
    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

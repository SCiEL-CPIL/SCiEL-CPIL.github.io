import time
import numpy as np
from scipy.optimize import minimize
import random

TIME_LIMIT = 120.0
START_TIME = time.time()

def construct():
    global START_TIME, TIME_LIMIT
    n, m, p = 2, 4, 7
    
    # Trivial Rank = 56
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1

    best_result = {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }
    best_rank = rank_trivial

    # Target tensor T is all ones
    # We want to find R < 56 such that there exist U, V, W with:
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    def objective(params, R):
        # Unpack
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        # Compute tensor product
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We can use einsum
        # U is (i,j,r) -> reshape to (n, m, R)
        # V is (j,k,r) -> reshape to (m, p, R)
        # W is (k,i,r) -> reshape to (p, n, R)
        
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # Contract over r
        # Result shape (n, m, p)
        # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        
        # Efficient computation:
        # temp[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r]
        # This is a batched matrix multiplication?
        # U_3d is (n, m, R). V_3d is (m, p, R).
        # We want sum over last dim.
        # np.einsum('ijr,jkr->ijk', U_3d, V_3d)
        
        UV = np.einsum('ijr,jkr->ijk', U_3d, V_3d)
        
        # Now sum over r with W_3d[k,i,r]
        # W_3d is (p, n, R). We need W_3d[k,i,r].
        # UV is (i, j, k).
        # We need sum_r UV[i,j,k] * W_3d[k,i,r]
        # np.einsum('ijk,kir->ij', UV, W_3d) ? No.
        # W_3d[k, i, r].
        # So sum_r UV[i,j,k] * W_3d[k,i,r]
        
        T_computed = np.einsum('ijk,kir->ij', UV, W_3d) # Wait, result should be scalar? No.
        # The identity is for each i,j,k.
        # So T_computed should be (n, m, p).
        # Let's re-check indices.
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # UV[i,j,k] = sum_r U[i,j,r] V[j,k,r]
        # Then T[i,j,k] = sum_r UV[i,j,k] * W[k,i,r] ?? No.
        # UV already summed over r. We can't sum over r again.
        # The term is U * V * W.
        # We must keep r open until the end.
        
        # Let's do it in one step:
        # np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        T_computed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Target is 1 everywhere
        error = T_computed - 1.0
        return np.sum(error**2)

    def get_grad(params, R):
        # Numerical gradient or analytic?
        # Analytic is better.
        # d/dU[i,j,r] = 2 * sum_{i,j,k} (T_c - 1) * V[j,k,r] * W[k,i,r]
        # This is expensive to compute explicitly for every step.
        # L-BFGS can use numerical gradient if not provided, but it's slow.
        # Let's provide analytic gradient.
        
        U = params[:n*m*R].reshape(n, m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m, p, R)
        W = params[n*m*R + m*p*R:].reshape(p, n, R)
        
        T_c = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        err = T_c - 1.0
        
        # Grad U[i,j,r] = 2 * sum_k err[i,j,k] * V[j,k,r] * W[k,i,r]
        # Grad V[j,k,r] = 2 * sum_i err[i,j,k] * U[i,j,r] * W[k,i,r]
        # Grad W[k,i,r] = 2 * sum_j err[i,j,k] * U[i,j,r] * V[j,k,r]
        
        # Note: einsum indices
        # err is (i, j, k)
        # V is (j, k, r)
        # W is (k, i, r)
        
        # Grad U (i, j, r)
        # sum_k err[i,j,k] * V[j,k,r] * W[k,i,r]
        # np.einsum('ijk,jkr,kir->ijr', err, V, W)
        
        grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', err, V, W)
        grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', err, U, W)
        grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', err, U, V)
        
        # Flatten
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return grad

    # Search for lower rank
    # We try ranks from 40 down to 28
    # Limit time per rank
    
    for R in range(40, 27, -1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        found = False
        # Try a few random restarts
        for restart in range(5):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
                
            np.random.seed(restart * 1000 + R)
            # Initialize with small random values
            # Scale: since product is 1, maybe init around 1/sqrt(R)?
            init_scale = 1.0 / np.sqrt(R)
            params0 = np.random.uniform(-init_scale, init_scale, size=n*m*R + m*p*R + p*n*R)
            
            res = minimize(
                objective, 
                params0, 
                method='L-BFGS-B', 
                jac=get_grad,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.fun < 1e-8:
                # Check if we can snap to half-integers
                U = res.x[:n*m*R].reshape(n*m, R)
                V = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Snap to nearest half-integer
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify snapped solution
                U_3d = U_snapped.reshape(n, m, R)
                V_3d = V_snapped.reshape(m, p, R)
                W_3d = W_snapped.reshape(p, n, R)
                T_snapped = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                
                if np.allclose(T_snapped, 1.0, atol=1e-9):
                    best_result = {
                        "U": U_snapped.tolist(),
                        "V": V_snapped.tolist(),
                        "W": W_snapped.tolist()
                    }
                    best_rank = R
                    found = True
                    break # Found for this R, try lower R
        
        if not found:
            # If not found for this R, we could continue to lower R, 
            # but if we fail for 40, we might fail for 39.
            # However, sometimes lower rank is easier to find? Unlikely.
            # We'll continue searching.
            pass

    return best_result

if __name__ == "__main__":
    construct()

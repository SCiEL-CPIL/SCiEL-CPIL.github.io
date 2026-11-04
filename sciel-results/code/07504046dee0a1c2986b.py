"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 54
    
    # Build the target tensor T[a, b, c] where:
    # a = i*m + j (row i, col j of A), i in [0,n), j in [0,m)
    # b = j*p + k (row j, col k of B), j in [0,m), k in [0,p)
    # c = k*n + i (output index), k in [0,p), i in [0,n)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    
    def to_list(mat):
        # Convert to list of lists, rounding to nearest half-integer
        result = []
        for row in mat:
            r = []
            for val in row:
                # Round to nearest 0.5
                rv = round(val * 2) / 2
                r.append(rv)
            result.append(r)
        return result
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        R = U.shape[1]
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(R))
                    if abs(val - T[a, b, c]) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using matrix operations."""
        R = U.shape[1]
        # Compute T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # For each r, the contribution is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # We can check by computing for each (a,b): sum_r U[a,r]*V[b,r]*W[c,r] should match T[a,b,c]
        # This is like: for each (a,b), the vector W_recon[a,b,:] = sum_r U[a,r]*V[b,r]*W[:,r]
        # should equal T[a,b,:]
        # Equivalently, T[a,b,:] = W @ (U[:,r] * V[b,r])^T ... 
        # Let's do: T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol
    
    best_rank = n * m * p
    best_result = (to_list(U_sb), to_list(V_sb), to_list(W_sb))
    
    # Try to find lower rank decompositions
    # Start from rank 27 (theoretical lower bound) up to 53
    for R in range(27, min(54, n * m * p)):
        if time.time() > deadline - 30:
            break
        
        # Initialize with random or structured initialization
        # Use a structured initialization based on the tensor structure
        np.random.seed(R)  # Reproducible
        
        # Try multiple initializations
        found = False
        for init in range(3):
            if time.time() > deadline - 25:
                break
            
            if init == 0:
                # Random initialization
                U = np.random.randn(n * m, R) * 0.5
                V = np.random.randn(m * p, R) * 0.5
                W = np.random.randn(p * n, R) * 0.5
            elif init == 1:
                # Structured: try to use the natural structure
                # Each rank-1 term should connect one (i,j) to one (j,k) to one (k,i)
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                # Assign columns cyclically
                for r in range(R):
                    # Pick a random (i,j,k)
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                # Add some random perturbation
                U += np.random.randn(n * m, R) * 0.1
                V += np.random.randn(m * p, R) * 0.1
                W += np.random.randn(p * n, R) * 0.1
            else:
                # Use partial schoolbook and zero out some columns
                U = U_sb[:, :R].copy()
                V = V_sb[:, :R].copy()
                W = W_sb[:, :R].copy()
                # Perturb
                U += np.random.randn(n * m, R) * 0.1
                V += np.random.randn(m * p, R) * 0.1
                W += np.random.randn(p * n, R) * 0.1
            
            # Optimize using gradient descent
            # Objective: ||T - sum_r U[:,r]⊗V[:,r]⊗W[:,r]||_F^2
            def objective(params):
                U_flat = params[:n * m * R].reshape(n * m, R)
                V_flat = params[n * m * R:n * m * R + m * p * R].reshape(m * p, R)
                W_flat = params[n * m * R + m * p * R:].reshape(p * n, R)
                T_recon = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
                return np.sum((T_recon - T) ** 2)
            
            def gradient(params):
                U_flat = params[:n * m * R].reshape(n * m, R)
                V_flat = params[n * m * R:n * m * R + m * p * R].reshape(m * p, R)
                W_flat = params[n * m * R + m * p * R:].reshape(p * n, R)
                T_recon = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
                diff = T_recon - T
                # d/dU[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
                dU = np.einsum('abc,br,cr->ar', diff, V_flat, W_flat)
                dV = np.einsum('abc,ar,cr->br', diff, U_flat, W_flat)
                dW = np.einsum('abc,ar,br->cr', diff, U_flat, V_flat)
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B with limited iterations
            try:
                result = minimize(objective, params0, method='L-BFGS-B', jac=gradient,
                                  options={'maxiter': 200, 'gtol': 1e-8, 'maxfun': 500})
                
                U_opt = result.x[:n * m * R].reshape(n * m, R)
                V_opt = result.x[n * m * R:n * m * R + m * p * R].reshape(m * p, R)
                W_opt = result.x[n * m * R + m * p * R:].reshape(p * n, R)
                
                # Check if close to integer/half-integer
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify the rounded solution
                if verify_fast(U_rounded, V_rounded, W_rounded, tol=1e-6):
                    best_rank = R
                    best_result = (to_list(U_rounded), to_list(V_rounded), to_list(W_rounded))
                    found = True
                    break
                
                # Also check if the unrounded solution is very close
                if result.fun < 1e-10:
                    # Very close, try harder rounding
                    for scale in [2, 4, 8]:
                        U_r = np.round(U_opt * scale) / scale
                        V_r = np.round(V_opt * scale) / scale
                        W_r = np.round(W_opt * scale) / scale
                        if verify_fast(U_r, V_r, W_r, tol=1e-6):
                            best_rank = R
                            best_result = (to_list(U_r), to_list(V_r), to_list(W_r))
                            found = True
                            break
                    if found:
                        break
                        
            except Exception:
                continue
        
        if found:
            # Try to go even lower
            continue
    
    # Final verification
    U_final, V_final, W_final = best_result
    R_final = len(U_final[0]) if U_final else 0
    
    # Double-check with exact verification
    U_np = np.array(U_final, dtype=float)
    V_np = np.array(V_final, dtype=float)
    W_np = np.array(W_final, dtype=float)
    
    if not verify_fast(U_np, V_np, W_np, tol=1e-9):
        # Fall back to schoolbook
        U_np, V_np, W_np = U_sb, V_sb, W_sb
        U_final, V_final, W_final = to_list(U_np), to_list(V_np), to_list(W_np)
        R_final = n * m * p
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END

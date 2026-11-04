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
    
    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using matrix operations."""
        R = U.shape[1]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol
    
    best_rank = n * m * p
    best_result = (to_list(U_sb), to_list(V_sb), to_list(W_sb))
    
    # Try to find lower rank decompositions using structured approach
    # The key insight: for each (i,j,k) triple, we need T[i*m+j, j*p+k, k*n+i] = 1
    # A rank-1 term can cover multiple (i,j,k) triples if the factors have multiple non-zeros
    
    # Try ranks from 30 to 53 (skip very low ranks that are unlikely)
    for R in range(30, min(54, n * m * p)):
        if time.time() > deadline - 20:
            break
        
        np.random.seed(R * 7 + 13)  # Reproducible
        
        # Try multiple structured initializations
        found = False
        for init in range(4):
            if time.time() > deadline - 15:
                break
            
            if init == 0:
                # Random initialization with moderate scale
                U = np.random.randn(n * m, R) * 1.0
                V = np.random.randn(m * p, R) * 1.0
                W = np.random.randn(p * n, R) * 1.0
            elif init == 1:
                # Structured: group terms by i
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                terms_per_i = R // n
                for i in range(n):
                    for t in range(terms_per_i):
                        r = i * terms_per_i + t
                        j = t % m
                        k = (t // m) % p
                        U[i * m + j, r] = 1.0
                        V[j * p + k, r] = 1.0
                        W[k * n + i, r] = 1.0
                # Add perturbation
                U += np.random.randn(n * m, R) * 0.5
                V += np.random.randn(m * p, R) * 0.5
                W += np.random.randn(p * n, R) * 0.5
            elif init == 2:
                # Use schoolbook subset with perturbation
                U = U_sb[:, :R].copy()
                V = V_sb[:, :R].copy()
                W = W_sb[:, :R].copy()
                U += np.random.randn(n * m, R) * 0.3
                V += np.random.randn(m * p, R) * 0.3
                W += np.random.randn(p * n, R) * 0.3
            else:
                # Block-structured: each block of R/n terms handles one i
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                R_per_i = R // n
                for i in range(n):
                    for r in range(R_per_i):
                        col = i * R_per_i + r
                        # Distribute across j and k
                        j = r % m
                        k = (r // m) % p
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                # Perturb
                U += np.random.randn(n * m, R) * 0.2
                V += np.random.randn(m * p, R) * 0.2
                W += np.random.randn(p * n, R) * 0.2
            
            # Optimize
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
                dU = np.einsum('abc,br,cr->ar', diff, V_flat, W_flat)
                dV = np.einsum('abc,ar,cr->br', diff, U_flat, W_flat)
                dW = np.einsum('abc,ar,br->cr', diff, U_flat, V_flat)
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                # Use L-BFGS-B with more iterations
                result = minimize(objective, params0, method='L-BFGS-B', jac=gradient,
                                  options={'maxiter': 300, 'gtol': 1e-10, 'maxfun': 1000})
                
                U_opt = result.x[:n * m * R].reshape(n * m, R)
                V_opt = result.x[n * m * R:n * m * R + m * p * R].reshape(m * p, R)
                W_opt = result.x[n * m * R + m * p * R:].reshape(p * n, R)
                
                # Try rounding to half-integers
                for scale in [2, 4]:
                    U_rounded = np.round(U_opt * scale) / scale
                    V_rounded = np.round(V_opt * scale) / scale
                    W_rounded = np.round(W_opt * scale) / scale
                    
                    if verify_fast(U_rounded, V_rounded, W_rounded, tol=1e-6):
                        best_rank = R
                        best_result = (to_list(U_rounded), to_list(V_rounded), to_list(W_rounded))
                        found = True
                        break
                    if verify_fast(U_opt, V_opt, W_opt, tol=1e-9):
                        # Check if close to half-integers
                        if np.max(np.abs(U_opt * 2 - np.round(U_opt * 2))) < 0.1 and \
                           np.max(np.abs(V_opt * 2 - np.round(V_opt * 2))) < 0.1 and \
                           np.max(np.abs(W_opt * 2 - np.round(W_opt * 2))) < 0.1:
                            best_rank = R
                            best_result = (to_list(U_opt), to_list(V_opt), to_list(W_opt))
                            found = True
                            break
                    if found:
                        break
                        
            except Exception:
                continue
        
        if found:
            continue
    
    # Final verification
    U_final, V_final, W_final = best_result
    
    U_np = np.array(U_final, dtype=float)
    V_np = np.array(V_final, dtype=float)
    W_np = np.array(W_final, dtype=float)
    
    if not verify_fast(U_np, V_np, W_np, tol=1e-9):
        # Fall back to schoolbook
        U_np, V_np, W_np = U_sb, V_sb, W_sb
        U_final, V_final, W_final = to_list(U_np), to_list(V_np), to_list(W_np)
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END

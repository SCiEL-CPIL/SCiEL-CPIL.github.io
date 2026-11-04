"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Start with schoolbook rank 56 and try to reduce
    # First, let's build the known rank-50 solution as a starting point
    rank0 = 50
    U0 = np.zeros((nm, rank0))
    V0 = np.zeros((mp, rank0))
    W0 = np.zeros((pn, rank0))
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        U0[i0*m + j0, col] = 1
        U0[(i0+1)*m + (j0+1), col] = 1
        V0[j0*p + k0, col] = 1
        V0[(j0+1)*p + (k0+1), col] = 1
        W0[k0*n + i0, col] = 1
        W0[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # M2: (A21+A22)B11 -> C21 += M2, C22 -= M2
        U0[(i0+1)*m + j0, col] = 1
        U0[(i0+1)*m + (j0+1), col] = 1
        V0[j0*p + k0, col] = 1
        W0[k0*n + (i0+1), col] = 1
        W0[(k0+1)*n + (i0+1), col] = -1
        col += 1
        
        # M3: A11(B12-B22) -> C12 += M3, C22 += M3
        U0[i0*m + j0, col] = 1
        V0[j0*p + (k0+1), col] = 1
        V0[(j0+1)*p + (k0+1), col] = -1
        W0[(k0+1)*n + i0, col] = 1
        W0[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # M4: A22(B21-B11) -> C11 += M4, C21 += M4
        U0[(i0+1)*m + (j0+1), col] = 1
        V0[(j0+1)*p + k0, col] = 1
        V0[j0*p + k0, col] = -1
        W0[k0*n + i0, col] = 1
        W0[k0*n + (i0+1), col] = 1
        col += 1
        
        # M5: (A11+A12)B22 -> C11 -= M5, C12 += M5
        U0[i0*m + j0, col] = 1
        U0[i0*m + (j0+1), col] = 1
        V0[(j0+1)*p + (k0+1), col] = 1
        W0[k0*n + i0, col] = -1
        W0[(k0+1)*n + i0, col] = 1
        col += 1
        
        # M6: (A21-A11)(B11+B12) -> C22 += M6
        U0[(i0+1)*m + j0, col] = 1
        U0[i0*m + j0, col] = -1
        V0[j0*p + k0, col] = 1
        V0[j0*p + (k0+1), col] = 1
        W0[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        # M7: (A12-A22)(B21+B22) -> C11 += M7
        U0[i0*m + (j0+1), col] = 1
        U0[(i0+1)*m + (j0+1), col] = -1
        V0[(j0+1)*p + k0, col] = 1
        V0[(j0+1)*p + (k0+1), col] = 1
        W0[k0*n + i0, col] = 1
        col += 1

    # 2x4x4: 4 Strassen blocks
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # 2x4x2: 2 Strassen blocks
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 2x4x1: 8 rank-1 terms
    for i in range(2):
        for j in range(4):
            U0[i*m + j, col] = 1
            V0[j*p + 6, col] = 1
            W0[6*n + i, col] = 1
            col += 1
    
    # Verify the rank-50 solution
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(lhs - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U0, V0, W0, rank0):
        # Fallback: return the rank-50 solution as-is
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    # Try to reduce rank using optimization
    # Strategy: try to express the tensor with fewer rank-1 terms
    # We'll use a gradient-based approach to minimize rank
    
    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0
    
    # Try ranks from 48 down to 44
    for target_rank in range(48, 43, -1):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Initialize with random perturbation of the first target_rank columns
        np.random.seed(42 + target_rank)
        
        for attempt in range(20):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            # Start from the rank-50 solution, take first target_rank columns
            # and add some random noise
            U_init = U0[:, :target_rank].copy() + 0.01 * np.random.randn(nm, target_rank)
            V_init = V0[:, :target_rank].copy() + 0.01 * np.random.randn(mp, target_rank)
            W_init = W0[:, :target_rank].copy() + 0.01 * np.random.randn(pn, target_rank)
            
            # Define the objective: we want the tensor decomposition to match T[i,j,k] = 1
            # The constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # We'll minimize the sum of squared errors + penalty for non-zero entries outside
            
            def objective(params):
                U_p = params[:nm*target_rank].reshape(nm, target_rank)
                V_p = params[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_p = params[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            lhs = sum(U_p[i*m+j, r] * V_p[j*p+k, r] * W_p[k*n+i, r] for r in range(target_rank))
                            error += (lhs - 1.0) ** 2
                
                # L1 penalty to encourage sparsity (optional)
                # l1 = 0.1 * (np.abs(U_p).sum() + np.abs(V_p).sum() + np.abs(W_p).sum())
                l1 = 0.0
                
                return error + l1
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                
                if result.fun < 1e-8:
                    U_opt = result.x[:nm*target_rank].reshape(nm, target_rank)
                    V_opt = result.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                    W_opt = result.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                    
                    # Verify
                    if verify(U_opt, V_opt, W_opt, target_rank):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        break  # Found a valid lower-rank decomposition
            except Exception:
                pass
        
        if best_rank < target_rank + 1:
            break  # Found a better rank, stop
    
    # Convert to list format
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

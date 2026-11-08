import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def _build_rank17():
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Part 1: Strassen for 2x2x2
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3: A11(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5: (A10-A00)(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2]
    # r=7: A02 B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02 B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12 B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12 B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2]
    # r=11: A00 B02 -> C02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A01 B12 -> C02
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A02 B22 -> C02
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # r=14: A10 B02 -> C12
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # r=15: A11 B12 -> C12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # r=16: A12 B22 -> C12
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W form a valid decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Calculate sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # result[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    reconstructed = np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(reconstructed, 1.0, atol=1e-9):
        return False
    return True

def _attempt_rank16(U17, V17, W17, n, m, p, time_limit_per_attempt=4.0):
    """
    Attempts to find a rank 16 decomposition by removing one term and optimizing.
    """
    start_time = time.time()
    
    # We will try removing each term index
    for remove_idx in range(17):
        if time.time() - start_time > 100.0: # Global safety margin
            break
            
        # Create initial guess: Rank 17 with one column zeroed
        U = U17.copy()
        V = V17.copy()
        W = W17.copy()
        U[:, remove_idx] = 0
        V[:, remove_idx] = 0
        W[:, remove_idx] = 0
        
        # Flatten the remaining 16 columns
        # We keep the structure of 16 columns.
        U_16 = U[:, :remove_idx].copy() if remove_idx == 0 else np.hstack([U[:, :remove_idx], U[:, remove_idx+1:]])
        V_16 = V[:, :remove_idx].copy() if remove_idx == 0 else np.hstack([V[:, :remove_idx], V[:, remove_idx+1:]])
        W_16 = W[:, :remove_idx].copy() if remove_idx == 0 else np.hstack([W[:, :remove_idx], W[:, remove_idx+1:]])
        
        # Flatten to vector
        x0 = np.concatenate([U_16.flatten(), V_16.flatten(), W_16.flatten()])
        
        # Objective function
        def objective(x):
            U_flat = x[:6*16]
            V_flat = x[6*16:12*16]
            W_flat = x[12*16:]
            
            U_curr = U_flat.reshape(6, 16)
            V_curr = V_flat.reshape(9, 16)
            W_curr = W_flat.reshape(6, 16)
            
            U_3d = U_curr.reshape(n, m, 16)
            V_3d = V_curr.reshape(m, p, 16)
            W_3d = W_curr.reshape(p, n, 16)
            
            rec = np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d)
            err = rec - 1.0
            return np.sum(err**2)
        
        # Gradient (optional, but helps L-BFGS-B)
        def gradient(x):
            U_flat = x[:6*16]
            V_flat = x[6*16:12*16]
            W_flat = x[12*16:]
            
            U_curr = U_flat.reshape(6, 16)
            V_curr = V_flat.reshape(9, 16)
            W_curr = W_flat.reshape(6, 16)
            
            U_3d = U_curr.reshape(n, m, 16)
            V_3d = V_curr.reshape(m, p, 16)
            W_3d = W_curr.reshape(p, n, 16)
            
            rec = np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d)
            err = rec - 1.0
            
            # Gradient with respect to U_3d
            # d/dU[i,j,r] sum_{i,j,k} err[i,j,k]^2 = 2 * err[i,j,k] * V[j,k,r] * W[k,i,r]
            grad_U = 2 * np.einsum('ijk,jkr,kin->ijr', err, V_3d, W_3d)
            grad_V = 2 * np.einsum('ijk,ijr,kin->jkr', err, U_3d, W_3d)
            grad_W = 2 * np.einsum('ijk,ijr,jkr->kin', err, U_3d, V_3d)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        # Optimize
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False})
            
            if res.fun < 1e-6:
                # Extract solution
                x_sol = res.x
                U_sol = x_sol[:6*16].reshape(6, 16)
                V_sol = x_sol[6*16:12*16].reshape(9, 16)
                W_sol = x_sol[12*16:].reshape(6, 16)
                
                # Verify with high precision
                if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p):
                    return U_sol, V_sol, W_sol
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Baseline Rank 17
    U17, V17, W17 = _build_rank17()
    
    # Try Rank 16
    # We have ~120s. Baseline is instant.
    # Allocate 100s for search.
    if time.time() - start_time < 100.0:
        result_16 = _attempt_rank16(U17, V17, W17, n, m, p)
        if result_16 is not None:
            U, V, W = result_16
            # Final sanity check
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Rank 17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}

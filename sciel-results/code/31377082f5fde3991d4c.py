"""Find rank-minimal tensor decomposition for 3x3 * 3x7 matrix multiplication."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check: for all i, j, k, the sum over alpha of U[i*m+j, alpha] * V[j*p+k, alpha] * W[k*n+i, alpha]
    # should equal 1 if this is a valid "multiplication" term, 0 otherwise.
    # Actually the tensor T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[a,b,c] = 1 iff j == j' and k == k' and i == i' ... no.
    # The multiplication tensor: T[i,j,k] = delta_{j, j'} where we compute C[i,k] = sum_j A[i,j] B[j,k]
    # In the 3-way tensor formulation with indices (a, b, c):
    # a = i*m + j (row i, col j of A)
    # b = j*p + k (row j, col k of B)  
    # c = k*n + i (col k, row i of C)
    # T[a, b, c] = 1 if the j in a equals the j in b, AND the k in b equals the k in c, AND the i in c equals the i in a
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1, and 0 otherwise.
    
    # Reconstruct the tensor from U, V, W
    # T_recon[a, b, c] = sum_alpha U[a, alpha] * V[b, alpha] * W[c, alpha]
    # We need T_recon to match T_target.
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Compute T_recon
    # T_recon[a, b, c] = sum_alpha U[a, alpha] * V[b, alpha] * W[c, alpha]
    # This is a 3-tensor product. We can compute it as:
    # For each alpha, outer product U[:,alpha] * V[:,alpha] * W[:,alpha]
    # Sum over alpha.
    
    # Efficient: T_recon = sum_alpha np.einsum('a,b,c->abc', U[:,alpha], V[:,alpha], W[:,alpha])
    # But that's slow for large rank. Use:
    # T_recon[a,b,c] = sum_alpha U[a,alpha] * (V[b,alpha] * W[c,alpha])
    # = sum_alpha U[a,alpha] * (V * W^T)[b,c]... no, V and W have different shapes.
    
    # Actually: define M_alpha = V[:,alpha] * W[:,alpha] (element-wise, shape (m*p, p*n))
    # Then T_recon[a,b,c] = sum_alpha U[a,alpha] * M_alpha[b,c]
    
    # Let's compute it directly for verification
    T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
    for alpha in range(rank):
        ua = U[:, alpha]
        va = V[:, alpha]
        wa = W[:, alpha]
        # outer product
        T_recon += np.einsum('a,b,c->abc', ua, va, wa)
    
    err = np.max(np.abs(T_recon - T_target))
    return err < tol

def _to_half_integer(U, V, W, tol=1e-4):
    """Try to snap entries to half-integers and verify."""
    U2 = np.round(np.array(U) * 2) / 2.0
    V2 = np.round(np.array(V) * 2) / 2.0
    W2 = np.round(np.array(W) * 2) / 2.0
    return U2, V2, W2

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Start with schoolbook rank 63
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    best_err = 0.0
    
    # Try to reduce rank by zeroing out columns and optimizing
    # Strategy: start from rank 63, try to set some columns to 0 and reoptimize
    
    # Alternative: use a known reduced-rank construction
    # For 3x3 x 3x3, use rank 21 (Strassen-like)
    # For 3x3 x 3x4, use rank 12 (schoolbook)
    # Total: 33
    
    # Let me try to construct a rank-33 solution using block structure.
    # Split B into B1 (3x3) and B2 (3x4)
    # C1 = A * B1 (3x3), C2 = A * B2 (3x4)
    # Use rank-21 for C1, rank-12 for C2
    # But the U factors must be consistent...
    
    # Actually, let's try a different approach: numerical optimization
    # Start with a random low-rank initialization and optimize
    
    # Try ranks from 25 up to 62
    for target_rank in range(25, min(63, rank0)):
        if time.time() > deadline - 30:
            break
        
        # Initialize
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Optimize
        from scipy.optimize import minimize
        
        def objective(params):
            if time.time() > deadline - 10:
                return 1e10
            U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_flat = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            
            # Compute error
            T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
            for alpha in range(target_rank):
                T_recon += np.einsum('a,b,c->abc', U_flat[:,alpha], V_flat[:,alpha], W_flat[:,alpha])
            
            T_target = np.zeros((n*m, m*p, p*n), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m+j, j*p+k, k*n+i] = 1.0
            
            err = np.sum((T_recon - T_target)**2)
            return err
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'gtol': 1e-8})
            if result.fun < 1e-6:
                U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Try to snap to half-integers
                U_hi, V_hi, W_hi = _to_half_integer(U_opt, V_opt, W_opt)
                if _verify(U_hi, V_hi, W_hi, n, m, p):
                    best_U, best_V, best_W = U_hi, V_hi, W_hi
                    best_rank = target_rank
                    break
        except Exception:
            pass
    
    # Return best found
    U_ret = best_U.tolist()
    V_ret = best_V.tolist()
    W_ret = best_W.tolist()
    return {"U": U_ret, "V": V_ret, "W": W_ret}
